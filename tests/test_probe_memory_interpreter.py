import asyncio
import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest

from grounded_mortgage_assistant.guardrails.privacy import ingest
from grounded_mortgage_assistant.llm.conversation import (
    LOCAL_MODEL,
    InterpretationResult,
)
from grounded_mortgage_assistant.state.memory import (
    FactOperation,
    Memory,
    TurnInterpretation,
    assertion_unit_ranges,
)

_SPEC = importlib.util.spec_from_file_location(
    "probe_memory_interpreter",
    Path(__file__).parents[1] / "scripts" / "probe_memory_interpreter.py",
)
assert _SPEC and _SPEC.loader
_PROBE = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _PROBE
_SPEC.loader.exec_module(_PROBE)
CanaryTransport = _PROBE.CanaryTransport
CanaryAudit = _PROBE.CanaryAudit
GateFailure = _PROBE.GateFailure
_assert_ltv = _PROBE._assert_ltv


def test_canary_transport_retains_earlier_turn_canaries():
    async def exercise():
        audit = CanaryAudit({"earlier-private", "later-private"})
        transport = CanaryTransport(
            httpx.MockTransport(lambda request: httpx.Response(200)), audit
        )
        with pytest.raises(Exception, match="PRIVATE_CANARY_IN_REQUEST"):
            await transport.handle_async_request(
                httpx.Request("POST", "http://127.0.0.1", content="earlier-private")
            )

    asyncio.run(exercise())


def test_each_probe_transport_uses_a_fresh_nonreusable_delegate():
    class SingleUse(httpx.AsyncBaseTransport):
        def __init__(self):
            self.closed = False

        async def handle_async_request(self, request):
            if self.closed:
                raise RuntimeError("reused after close")
            return httpx.Response(200)

        async def aclose(self):
            if self.closed:
                raise RuntimeError("closed twice")
            self.closed = True

    async def exercise():
        audit = CanaryAudit()
        first = CanaryTransport(SingleUse(), audit)
        await first.handle_async_request(httpx.Request("POST", "http://127.0.0.1"))
        await first.aclose()
        second = CanaryTransport(SingleUse(), audit)
        await second.handle_async_request(httpx.Request("POST", "http://127.0.0.1"))
        await second.aclose()
        assert audit.calls == 2

    asyncio.run(exercise())


def test_ltv_gate_rejects_missing_or_stale_calculation_state():
    with pytest.raises(GateFailure, match="CALCULATION_STATE_ASSERTION_FAILED"):
        _assert_ltv(Memory(), "70.00")


def test_probe_evidence_keeps_only_safe_schema_diagnostics(monkeypatch):
    class Tokenizer:
        pass

    async def invalid_interpretation(*args, **kwargs):
        raise _PROBE.InterpretationError(
            "INVALID_TOOL_ARGUMENTS",
            schema_diagnostics=(("requested_action", "literal_error"),),
        )

    monkeypatch.setattr(
        _PROBE.AutoTokenizer, "from_pretrained", lambda *args, **kwargs: Tokenizer()
    )
    monkeypatch.setattr(_PROBE, "interpret_turn", invalid_interpretation)
    with pytest.raises(GateFailure) as raised:
        asyncio.run(
            _PROBE.run(
                SimpleNamespace(
                    tokenizer_path=Path("unused"), capacity=262144, max_calls=12
                )
            )
        )
    evidence = raised.value.evidence
    failed = evidence[-1]
    assert failed["schema_diagnostics"] == [
        {"path": "requested_action", "code": "literal_error"}
    ]
    rendered = repr(evidence)
    assert "refinance my existing mortgage" not in rendered
    assert "development-client" not in rendered


def test_original_natural_language_probe_cases_now_have_distinct_safe_tokens():
    assert (
        "NEEDED" in ingest("What information is needed for refinance?").model_projection
    )
    assert (
        "NEEDS"
        in ingest("My friend needs to refinance their mortgage.").model_projection
    )
    assert "VALUE(" in " ".join(
        ingest("If my mortgage is SEK 1,820,000, calculate LTV.").model_projection
    )


@pytest.mark.parametrize(
    ("bad_control", "hypothetical_action", "passes"),
    [
        (None, "clarify", True),
        (None, "calculate", True),
        ("third-party", "clarify", False),
        ("hypothetical", "clarify", False),
        (None, "forget", False),
        (None, "review", False),
    ],
)
def test_full_mocked_probe_run_preserves_original_order_and_rejects_wrong_control_scope(
    monkeypatch, bad_control, hypothetical_action, passes
):
    calls = []

    class Tokenizer:
        pass

    def proposal(turn):
        span = assertion_unit_ranges(turn)[0]
        base = {
            "subject": "self",
            "conditional": False,
            "operations": [],
            "goal": "unchanged",
            "borrowing": "unchanged",
            "ownership": "unchanged",
            "requested_action": "acknowledge",
            "unresolved": [],
        }
        text = turn.original
        if text.startswith("I want to refinance"):
            base.update(
                goal="refinance",
                borrowing="same",
                ownership="own_existing",
                scope_source_spans=[span],
            )
        elif text.startswith(("My mortgage is SEK", "Actually")):
            base["operations"] = [
                FactOperation(
                    field="existing_loan_amount",
                    operation="set",
                    value_refs=[turn.protected_values[0].value_ref],
                    source_span=span,
                )
            ]
        elif text.startswith("SEK 2,600"):
            base["operations"] = [
                FactOperation(
                    field="property_value",
                    operation="set",
                    value_refs=[turn.protected_values[0].value_ref],
                    source_span=span,
                )
            ]
        elif text == "Calculate LTV.":
            base["requested_action"] = "calculate"
        elif text.startswith("What information"):
            base.update(
                requested_action="knowledge",
                knowledge_question=["INFORMATION", "REFINANCE"],
            )
        elif text == "House.":
            base["operations"] = [
                FactOperation(
                    field="property_type",
                    operation="set",
                    enum_value="house",
                    source_span=span,
                )
            ]
        elif text.startswith("I would like a better rate"):
            base.update(
                goal="refinance", ownership="own_existing", scope_source_spans=[span]
            )
        elif text.startswith("My friend"):
            base.update(
                subject="self" if bad_control == "third-party" else "third_party",
                requested_action="clarify",
            )
        elif text.startswith("If my"):
            base.update(
                subject="hypothetical",
                conditional=bad_control != "hypothetical",
                requested_action=hypothetical_action,
            )
        else:
            base["requested_action"] = "clarify"
        return TurnInterpretation(**base)

    async def fake_interpret(state, turn, **kwargs):
        calls.append(turn.original)
        kwargs["transport"].audit.calls += 1
        return InterpretationResult(
            proposal(turn), "call", LOCAL_MODEL, 1, 1, 1, 1, "a" * 64, "b" * 64
        )

    monkeypatch.setattr(
        _PROBE.AutoTokenizer, "from_pretrained", lambda *args, **kwargs: Tokenizer()
    )
    monkeypatch.setattr(_PROBE, "interpret_turn", fake_interpret)
    if not passes:
        with pytest.raises(GateFailure):
            asyncio.run(
                _PROBE.run(
                    SimpleNamespace(
                        tokenizer_path=Path("unused"), capacity=262144, max_calls=12
                    )
                )
            )
        return
    evidence, complete = asyncio.run(
        _PROBE.run(
            SimpleNamespace(
                tokenizer_path=Path("unused"), capacity=262144, max_calls=12
            )
        )
    )
    assert complete is True
    assert len(calls) == 12
    assert [item["case_id"] for item in evidence[:2]] == [
        "sensitive-ingress",
        "unknown-ingress",
    ]
    assert evidence[-1]["case_id"] == "hypothetical"
    assert (
        evidence[2]
        | {
            "requested_action": "acknowledge",
            "subject": "self",
            "conditional": False,
            "operation_count": 0,
            "scope_span_count": 1,
        }
        == evidence[2]
    )
    rendered = str(evidence)
    assert "1,820,000" not in rendered and "development-client" not in rendered
    assert calls == [
        "I want to refinance my existing mortgage without extra borrowing.",
        "My mortgage is SEK 1,820,000.",
        "SEK 2,600,000.",
        "Calculate LTV.",
        "Actually, my mortgage is SEK 1,560,000.",
        "Calculate LTV.",
        "What information is needed for refinance?",
        "House.",
        "I would like a better rate on my existing mortgage.",
        "My mortgage is not SEK 1,820,000.",
        "My friend needs to refinance their mortgage.",
        "If my mortgage is SEK 1,820,000, calculate LTV.",
    ]
