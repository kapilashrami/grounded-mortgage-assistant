"""Bounded exposed-development gate for the local native turn interpreter."""

from __future__ import annotations

import argparse
import asyncio
import json
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

import httpx
from transformers import AutoTokenizer

from grounded_mortgage_assistant.guardrails.privacy import (
    IngressError,
    SanitizedTurn,
    ingest,
)
from grounded_mortgage_assistant.llm.conversation import (
    InterpretationError,
    InterpretationResult,
    interpret_turn,
)
from grounded_mortgage_assistant.state.memory import (
    FactStatus,
    Memory,
    MemoryError,
    choose_action,
    choose_pending_question,
    stage_turn,
    stage_uninterpreted_turn,
)
from grounded_mortgage_assistant.tools.calculations import calculate_ltv

MAX_CALLS = 16

# Pre-run safe classifications for the exposed controls. These are deliberately
# narrower than the general schema and do not authorize facts or confirmation.
CONTROL_ACTIONS = {
    "better-rate": frozenset({"acknowledge", "clarify", "knowledge"}),
    "negated-loan": frozenset({"acknowledge", "clarify"}),
    "third-party": frozenset({"acknowledge", "clarify", "knowledge"}),
    "hypothetical": frozenset({"calculate", "clarify"}),
}


class GateFailure(RuntimeError):
    def __init__(self, category: str):
        self.category = category
        super().__init__(category)


@dataclass(frozen=True)
class ProbeCase:
    case_id: str
    text: str


CASE_IDS = frozenset(
    {
        "scope",
        "loan",
        "property-value",
        "ltv-before-correction",
        "loan-correction",
        "ltv-after-correction",
        "knowledge-interruption",
        "property-type",
        "better-rate",
        "negated-loan",
        "third-party",
        "hypothetical",
        "sensitive-ingress",
        "unknown-ingress",
        "final-readiness",
        "gate",
    }
)


@dataclass
class CanaryAudit:
    forbidden: set[str] = field(default_factory=set)
    calls: int = 0


class CanaryTransport(httpx.AsyncBaseTransport):
    """Inspect actual post-adapter payloads transiently; never retain them."""

    def __init__(self, transport: httpx.AsyncBaseTransport, audit: CanaryAudit):
        self.transport, self.audit, self.closed = transport, audit, False

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        payload = request.content.decode("utf-8", errors="replace")
        if any(value and value in payload for value in self.audit.forbidden):
            raise InterpretationError("PRIVATE_CANARY_IN_REQUEST")
        self.audit.calls += 1
        return await self.transport.handle_async_request(request)

    async def aclose(self) -> None:
        if not self.closed:
            self.closed = True
            await self.transport.aclose()


def _canaries(turn: SanitizedTurn, client_id: str) -> set[str]:
    return {
        turn.original,
        client_id,
        *(x.original_proof.original for x in turn.protected_values),
        *(str(x.value) for x in turn.protected_values),
    }


def _record(
    evidence: list[dict[str, Any]],
    case_id: str,
    outcome: str,
    result: InterpretationResult | None,
    calls: int,
    *,
    schema_diagnostics: tuple[tuple[str, str], ...] = (),
) -> None:
    if case_id not in CASE_IDS:
        raise GateFailure("INVALID_CASE_ID")
    record: dict[str, Any] = {"case_id": case_id, "outcome": outcome, "calls": calls}
    if result:
        record.update(
            model=result.model,
            prompt_hash=result.prompt_hash,
            schema_hash=result.schema_hash,
            input_tokens=result.input_tokens,
            output_tokens=result.output_tokens,
            provider_input_tokens=result.provider_input_tokens,
            elapsed_ms=result.elapsed_ms,
            requested_action=result.interpretation.requested_action,
            subject=result.interpretation.subject,
            conditional=result.interpretation.conditional,
            operation_count=len(result.interpretation.operations),
            scope_span_count=len(result.interpretation.scope_source_spans),
        )
    if schema_diagnostics:
        record["schema_diagnostics"] = [
            {"path": path, "code": code} for path, code in schema_diagnostics
        ]
    evidence.append(record)


async def _step(
    state: Memory,
    text: str,
    case_id: str,
    tokenizer: Any,
    capacity: int,
    audit: CanaryAudit,
    evidence: list[dict[str, Any]],
    *,
    allow_rejected: bool = False,
) -> tuple[Memory, InterpretationResult]:
    client_id = f"development-client-{case_id}"
    try:
        turn = ingest(text)
    except IngressError as error:
        _record(evidence, case_id, error.code, None, 0)
        raise GateFailure("UNEXPECTED_INGRESS") from error
    audit.forbidden.update(_canaries(turn, client_id))
    before = audit.calls
    transport = CanaryTransport(httpx.AsyncHTTPTransport(), audit)
    try:
        result = await interpret_turn(
            state,
            turn,
            tokenizer=tokenizer,
            provider_capacity=capacity,
            transport=transport,
        )
    except InterpretationError as error:
        # Retain the real fail-closed state even though a native failure stops.
        failed = stage_uninterpreted_turn(
            state,
            turn,
            client_message_id=client_id,
            rejection_code="INTERPRETATION_FAILED",
        )
        _record(
            evidence,
            case_id,
            error.category,
            None,
            audit.calls - before,
            schema_diagnostics=error.schema_diagnostics,
        )
        failure = GateFailure("INTERPRETATION_FAILURE")
        failure.failed_state = failed.state
        raise failure from error
    finally:
        await transport.aclose()
    if audit.calls - before != 1:
        raise GateFailure("CALL_COUNT_MISMATCH")
    try:
        staged = stage_turn(
            state, turn, result.interpretation, client_message_id=client_id
        )
    except MemoryError as error:
        _record(evidence, case_id, "INVALID_PROPOSAL", result, 1)
        raise GateFailure("REJECTED_CONTROL_PROPOSAL") from error
    _record(evidence, case_id, staged.rejection_code or "interpreted", result, 1)
    if staged.proposal_rejected and not allow_rejected:
        raise GateFailure("UNEXPECTED_PROPOSAL_REJECTION")
    return staged.state, result


async def _run_case(
    state: Memory,
    case: ProbeCase,
    tokenizer: Any,
    capacity: int,
    audit: CanaryAudit,
    evidence: list[dict[str, Any]],
    *,
    allow_rejected: bool = False,
) -> tuple[Memory, InterpretationResult]:
    return await _step(
        state,
        text=case.text,
        case_id=case.case_id,
        tokenizer=tokenizer,
        capacity=capacity,
        audit=audit,
        evidence=evidence,
        allow_rejected=allow_rejected,
    )


def _fact(state: Memory, field: str, revision: int) -> None:
    fact = state.facts.get(field)
    if (
        not fact
        or fact.status != FactStatus.KNOWN
        or fact.field_revision != revision
        or fact.original_proof is None
    ):
        raise GateFailure("FACT_OR_PROVENANCE_ASSERTION_FAILED")
    if fact.value_ref is not None:
        value = state.protected_values.get(fact.value_ref)
        if (
            value is None
            or value.original_proof.turn_ref != fact.original_proof.turn_ref
        ):
            raise GateFailure("FACT_OR_PROVENANCE_ASSERTION_FAILED")


def _display_pending(
    state: Memory, field: str, *, allowed_action: str | None = None
) -> Memory:
    pending = choose_pending_question(state, allowed_action or choose_action(state))
    if not pending or pending.field != field:
        raise GateFailure("PENDING_QUESTION_ASSERTION_FAILED")
    return replace(state, pending_question=pending)


def _assert_ltv(state: Memory, expected_percent: str) -> None:
    if choose_action(state) != "calculate":
        raise GateFailure("CALCULATION_STATE_ASSERTION_FAILED")
    try:
        loan = state.facts["existing_loan_amount"]
        value = state.facts["property_value"]
        result = calculate_ltv(
            state.protected_values[loan.value_ref].value,
            state.protected_values[value.value_ref].value,
        )
    except Exception as error:
        raise GateFailure("PRIVATE_CALCULATION_ASSERTION_FAILED") from error
    if result.get("percent") != expected_percent:
        raise GateFailure("PRIVATE_CALCULATION_ASSERTION_FAILED")


async def run(args: argparse.Namespace) -> tuple[list[dict[str, Any]], bool]:
    evidence: list[dict[str, Any]] = []
    sequence = (
        ProbeCase(
            "scope", "I want to refinance my existing mortgage without extra borrowing."
        ),
        ProbeCase("loan", "My mortgage is SEK 1,820,000."),
        ProbeCase("property-value", "SEK 2,600,000."),
        ProbeCase("ltv-before-correction", "Calculate LTV."),
        ProbeCase("loan-correction", "Actually, my mortgage is SEK 1,560,000."),
        ProbeCase("ltv-after-correction", "Calculate LTV."),
        ProbeCase(
            "knowledge-interruption", "What information is needed for refinance?"
        ),
        ProbeCase("property-type", "House."),
    )
    controls = (
        ProbeCase("better-rate", "I would like a better rate on my existing mortgage."),
        ProbeCase("negated-loan", "My mortgage is not SEK 1,820,000."),
        ProbeCase("third-party", "My friend needs to refinance their mortgage."),
        ProbeCase("hypothetical", "If my mortgage is SEK 1,820,000, calculate LTV."),
    )
    if args.max_calls < len(sequence) + len(controls):
        return evidence, False
    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer_path, local_files_only=True, trust_remote_code=False
    )
    audit, state = CanaryAudit(), Memory()
    for case_id, text in (
        ("sensitive-ingress", "Contact me at development@example.invalid."),
        ("unknown-ingress", "unreviewed vocabulary"),
    ):
        try:
            ingest(text)
        except IngressError as error:
            _record(evidence, case_id, error.code, None, 0)
        else:
            raise GateFailure("INGRESS_CONTROL_ASSERTION_FAILED")
    try:
        state, result = await _run_case(
            state, sequence[0], tokenizer, args.capacity, audit, evidence
        )
        if result.interpretation.requested_action not in {"acknowledge", "clarify"}:
            raise GateFailure("REQUESTED_ACTION_ASSERTION_FAILED")
        if (state.goal, state.scope, state.borrowing) != (
            "refinance",
            "own_existing",
            "same",
        ):
            raise GateFailure("SCOPE_ASSERTION_FAILED")
        state, _ = await _run_case(
            state, sequence[1], tokenizer, args.capacity, audit, evidence
        )
        _fact(state, "existing_loan_amount", 1)
        state = _display_pending(state, "property_value")
        state, _ = await _run_case(
            state, sequence[2], tokenizer, args.capacity, audit, evidence
        )
        _fact(state, "property_value", 1)
        state, result = await _run_case(
            state, sequence[3], tokenizer, args.capacity, audit, evidence
        )
        if result.interpretation.requested_action != "calculate":
            raise GateFailure("FIRST_LTV_ASSERTION_FAILED")
        _assert_ltv(state, "70.00")
        state, _ = await _run_case(
            state, sequence[4], tokenizer, args.capacity, audit, evidence
        )
        _fact(state, "existing_loan_amount", 2)
        _fact(state, "property_value", 1)
        state, result = await _run_case(
            state, sequence[5], tokenizer, args.capacity, audit, evidence
        )
        if result.interpretation.requested_action != "calculate":
            raise GateFailure("SECOND_LTV_ASSERTION_FAILED")
        _assert_ltv(state, "60.00")
        state = _display_pending(state, "property_type", allowed_action="clarify")
        pending_id = state.pending_question.question_id
        state, result = await _run_case(
            state, sequence[6], tokenizer, args.capacity, audit, evidence
        )
        if (
            result.interpretation.requested_action != "knowledge"
            or not state.pending_question
            or state.pending_question.question_id != pending_id
        ):
            raise GateFailure("PENDING_PRESERVATION_ASSERTION_FAILED")
        state, _ = await _run_case(
            state, sequence[7], tokenizer, args.capacity, audit, evidence
        )
        _fact(state, "property_type", 1)
        if any(
            field not in state.facts
            for field in ("existing_loan_amount", "property_value", "property_type")
        ):
            raise GateFailure("READINESS_ASSERTION_FAILED")
        _record(evidence, "final-readiness", "ready", None, 0)
        for case in controls:
            controlled, result = await _run_case(
                Memory(),
                case,
                tokenizer,
                args.capacity,
                audit,
                evidence,
            )
            if any(
                fact.status == FactStatus.KNOWN for fact in controlled.facts.values()
            ):
                raise GateFailure("NEGATIVE_CONTROL_ASSERTION_FAILED")
            expected_scope = {
                "better-rate": ("self", False),
                "negated-loan": ("self", False),
                "third-party": ("third_party", False),
                "hypothetical": ("hypothetical", True),
            }[case.case_id]
            if (
                result.interpretation.subject,
                result.interpretation.conditional,
            ) != expected_scope:
                raise GateFailure("CONTROL_SCOPE_ASSERTION_FAILED")
            if (
                result.interpretation.requested_action
                not in CONTROL_ACTIONS[case.case_id]
            ):
                raise GateFailure("REQUESTED_ACTION_ASSERTION_FAILED")
            if case.case_id == "better-rate" and (
                controlled.goal,
                controlled.scope,
                controlled.borrowing,
            ) != ("refinance", "own_existing", "unknown"):
                raise GateFailure("SCOPE_ASSERTION_FAILED")
    except GateFailure as error:
        error.evidence = evidence
        raise
    return evidence, True


def main() -> None:
    parser = argparse.ArgumentParser(
        description="bounded exposed-development interpreter gate"
    )
    parser.add_argument("--tokenizer-path", required=True, type=Path)
    parser.add_argument("--capacity", required=True, type=int)
    parser.add_argument("--max-calls", type=int, default=12)
    parser.add_argument("--evidence", required=True, type=Path)
    args = parser.parse_args()
    if args.capacity <= 0 or not 1 <= args.max_calls <= MAX_CALLS:
        raise SystemExit("invalid bounded probe options")
    results: list[dict[str, Any]] = []
    complete, failed = False, False
    try:
        results, complete = asyncio.run(run(args))
        failed = not complete
    except GateFailure as error:
        results = getattr(error, "evidence", results)
        results.append({"case_id": "gate", "outcome": error.category, "calls": 0})
        failed = True
    finally:
        safe = {
            "development_examples_exposed": True,
            "complete": complete,
            "native_call_count": sum(item["calls"] for item in results),
            "results": results,
        }
        args.evidence.write_text(
            json.dumps(safe, separators=(",", ":")), encoding="utf-8"
        )
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
