from __future__ import annotations

import asyncio

import pytest

from grounded_mortgage_assistant.guardrails.privacy import ingest
from grounded_mortgage_assistant.llm.conversation import InterpretationResult
from grounded_mortgage_assistant.state.memory import (
    FactOperation,
    Memory,
    TurnInterpretation,
)
from grounded_mortgage_assistant.tracing import TRACE_EVENTS, clarify_decision
from grounded_mortgage_assistant.workflow.graph import build_graph


def result(proposal: TurnInterpretation) -> InterpretationResult:
    return InterpretationResult(
        proposal, "call", "local", 12, 2, 260344, 1, "a" * 64, "b" * 64
    )


def proposal(turn, *, action="acknowledge", operations=()):
    return TurnInterpretation(
        subject="self",
        conditional=False,
        operations=list(operations),
        goal="unchanged",
        borrowing="unchanged",
        ownership="unchanged",
        requested_action=action,
        unresolved=[],
    )


def op(turn, field, value_index):
    return FactOperation(
        field=field,
        operation="set",
        value_refs=[turn.protected_values[value_index].value_ref],
        source_span={"token_start": 0, "token_end": len(turn.tokens) - 1},
    )


def test_empty_native_knowledge_query_fails_closed_without_positive_commit():
    turn = ingest("What information about mortgage?", turn_ref="t")
    invalid = TurnInterpretation(
        subject="self",
        conditional=False,
        operations=[],
        goal="unchanged",
        borrowing="unchanged",
        ownership="unchanged",
        requested_action="knowledge",
        knowledge_question=[],
        unresolved=[],
    )

    async def native(_, __):
        return result(invalid)

    state = asyncio.run(
        build_graph().ainvoke(
            {"memory": Memory(), "turn": turn, "message_id": "m", "interpreter": native}
        )
    )
    assert state["response"]["status"] == "CLARIFY"
    assert state["memory"].facts == {}


def _decisions(turn, interpretation):
    async def native(_, __):
        return result(interpretation)

    events: list[dict] = []
    token = TRACE_EVENTS.set(events)
    try:
        state = asyncio.run(
            build_graph().ainvoke(
                {
                    "memory": Memory(),
                    "turn": turn,
                    "message_id": "m",
                    "interpreter": native,
                }
            )
        )
    finally:
        TRACE_EVENTS.reset(token)
    return state, [e for e in events if e["name"] == "clarify_decision"]


def test_rejected_interpretation_records_which_check_failed():
    turn = ingest(
        "My existing mortgage is 2000000 and property is worth 2500000, what is my LTV?",
        turn_ref="t",
    )
    conditional = proposal(turn, action="calculate").model_copy(
        update={"conditional": True}
    )
    state, decisions = _decisions(turn, conditional)
    assert state["response"]["status"] == "CLARIFY"
    assert decisions == [
        {
            "name": "clarify_decision",
            "result": "stopped",
            "layer": "interpretation",
            "reason": "interpretation_unavailable",
            "cause": "NONPERSONAL_CALCULATION",
        }
    ]


def test_clarification_cause_must_be_a_fixed_code():
    with pytest.raises(ValueError):
        clarify_decision(
            "interpretation",
            "interpretation_unavailable",
            [],
            cause="my mortgage is 2m",
        )
