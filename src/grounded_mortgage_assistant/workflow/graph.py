"""Finite candidate graph. It stages only; the API layer is the sole committer."""

from __future__ import annotations

import re
from collections.abc import Awaitable, Callable
from dataclasses import replace
from datetime import UTC, datetime
from typing import Any, TypedDict
from uuid import uuid4

from langgraph.graph import END, START, StateGraph

from grounded_mortgage_assistant.contracts import MemoryReceipt
from grounded_mortgage_assistant.guardrails.privacy import SanitizedTurn
from grounded_mortgage_assistant.llm.conversation import InterpretationResult
from grounded_mortgage_assistant.llm.interpretation_candidates import (
    knowledge_query_is_eligible,
)
from grounded_mortgage_assistant.rag.answer import answer_public_question
from grounded_mortgage_assistant.rag.public_evidence import PublicEvidence
from grounded_mortgage_assistant.rag.public_questions import PublicQuestionScope
from grounded_mortgage_assistant.state.memory import (
    FactStatus,
    Memory,
    choose_action,
    choose_pending_question,
    is_explicit_ltv_request,
    stage_turn,
    stage_uninterpreted_turn,
)
from grounded_mortgage_assistant.state.proofs import resolve_numeric_fact
from grounded_mortgage_assistant.tools.calculations import calculate_ltv
from grounded_mortgage_assistant.tools.draft import (
    MemoryDraftError,
    prepare_memory_draft,
)
from grounded_mortgage_assistant.tools.schemas import LtvCalculation
from grounded_mortgage_assistant.tracing import clarify_decision, trace_event

FIXED_CLARIFICATION = "I could not validate that request. Please provide the missing or corrected details and request one action at a time."
V1_DRAFT_SCOPE_CLARIFICATION = "I can prepare a draft only for refinancing your own existing mortgage without increasing borrowing."
Interpreter = Callable[
    [Memory, SanitizedTurn], Awaitable[InterpretationResult] | InterpretationResult
]
Retriever = Callable[[str], Awaitable[list[dict[str, Any]]] | list[dict[str, Any]]]
Selector = Callable[[str, list[dict[str, Any]]], Awaitable[Any] | Any]


class GraphState(TypedDict, total=False):
    memory: Memory
    turn: SanitizedTurn
    message_id: str
    interpreter: Interpreter
    retriever: Retriever
    selector: Selector
    evidence: PublicEvidence
    public_questions: PublicQuestionScope
    interpreted: InterpretationResult
    staged: Any
    response: dict[str, Any]


async def _maybe(value: Any) -> Any:
    return await value if hasattr(value, "__await__") else value


def memory_receipt(memory: Memory, staged: Any) -> dict[str, Any]:
    return MemoryReceipt(
        revision=memory.revision,
        changes=[
            {
                "field": item.field,
                "fact_id": item.fact_id,
                "field_revision": item.field_revision,
                "status": item.status,
            }
            for item in staged.changes
        ],
        invalidated=list(dict.fromkeys(staged.invalidated)),
    ).model_dump()


def _rejection_cause(error: Exception) -> str:
    """Name the failed check with a fixed code; error text is never recorded."""
    category = getattr(error, "category", None)
    code = category if isinstance(category, str) else (error.args[:1] or [""])[0]
    if isinstance(code, str) and re.fullmatch(r"[A-Z][A-Z0-9_]{2,63}", code):
        return code
    return (
        "UNCLASSIFIED_" + re.sub(r"[^A-Z0-9_]", "", type(error).__name__.upper())[:40]
    )


def _restrictive(
    state: GraphState, code: str, layer: str, *, cause: str | None = None
) -> dict[str, Any]:
    staged = stage_uninterpreted_turn(
        state["memory"],
        state["turn"],
        client_message_id=state["message_id"],
        rejection_code="INTERPRETATION_FAILED",
    )
    clarify_decision(
        layer,
        {
            "INTERPRETATION_FAILED": "interpretation_unavailable",
            "CALCULATION_FAILED": "calculation_unavailable",
            "KNOWLEDGE_QUERY_UNAVAILABLE": "evidence_unavailable",
            "EVIDENCE_UNAVAILABLE": "evidence_unavailable",
        }[code],
        cause=cause,
    )
    return {
        "memory": staged.state,
        "response": {
            "status": "CLARIFY",
            "answer": FIXED_CLARIFICATION,
            "memory_receipt": memory_receipt(staged.state, staged),
            "error": {"code": code, "retry": "new_message"},
        },
    }


def complete_memory_draft(memory: Memory, staged: Any) -> tuple[Memory, dict[str, Any]]:
    """Stage one explicit v1 draft response; the API layer remains the committer."""
    receipt = memory_receipt(memory, staged)
    if (
        memory.goal != "refinance"
        or memory.scope != "own_existing"
        or memory.borrowing != "same"
        or not all(
            memory.scope_proofs.get(name) for name in ("goal", "scope", "borrowing")
        )
    ):
        clarify_decision("draft", "scope_unavailable")
        return memory, {
            "status": "CLARIFY",
            "answer": V1_DRAFT_SCOPE_CLARIFICATION,
            "memory_receipt": receipt,
        }
    pending = choose_pending_question(memory, "review")
    if pending is not None:
        labels = {
            "existing_loan_amount": "your existing mortgage amount",
            "property_value": "your property value",
            "property_type": "your property type",
        }
        memory = replace(memory, pending_question=pending)
        clarify_decision("missing_fact", "pending_question")
        return memory, {
            "status": "CLARIFY",
            "answer": f"Please provide {labels[pending.field]}.",
            "memory_receipt": memory_receipt(memory, staged),
        }
    try:
        draft = prepare_memory_draft(
            memory, now=datetime.now(UTC), draft_id=f"draft_{uuid4().hex}"
        )
    except MemoryDraftError:
        clarify_decision("draft", "draft_unavailable")
        return memory, {
            "status": "CLARIFY",
            "answer": "I could not prepare a mortgage draft from the retained details.",
            "memory_receipt": receipt,
            "error": {"code": MemoryDraftError.code, "retry": "new_message"},
        }
    memory = replace(memory, draft_requested=False)
    return memory, {
        "status": "ANSWERED",
        "answer": "Your mortgage draft is ready.",
        "mortgage_draft_result": "ready",
        "mortgage_draft": draft.draft.model_dump(mode="json"),
        "memory_receipt": memory_receipt(memory, staged),
    }


async def interpret(state: GraphState) -> dict[str, Any]:
    try:
        restrictive = stage_uninterpreted_turn(
            state["memory"],
            state["turn"],
            client_message_id=state["message_id"],
            rejection_code="INTERPRETATION_FAILED",
        )
        result = await _maybe(state["interpreter"](state["memory"], state["turn"]))
        if not isinstance(result, InterpretationResult):
            raise TypeError("interpreter result")
        proposal = result.interpretation
        if (
            proposal.requested_action in {"acknowledge", "clarify"}
            and proposal.subject == "self"
            and not proposal.conditional
            and is_explicit_ltv_request(state["turn"])
        ):
            # Local inference is not fully deterministic even at temperature 0. An
            # explicit LTV request is a calculation by the server's own grammar;
            # amounts and proofs are still validated before anything is computed.
            proposal = proposal.model_copy(update={"requested_action": "calculate"})
            trace_event(
                {
                    "name": "explicit_ltv_request",
                    "result": "calculate",
                    "model_action": result.interpretation.requested_action,
                }
            )
        elif (
            proposal.requested_action == "calculate"
            and not state["turn"].protected_values
            and not {"LTV", "CALCULATE"} & set(state["turn"].model_projection)
        ):
            # The mirror rule: a turn with no amounts and no calculation request
            # cannot start a calculation. A model may label a stated intent such
            # as "I want to refinance ..." as calculate after an earlier LTV.
            proposal = proposal.model_copy(update={"requested_action": "acknowledge"})
            trace_event(
                {
                    "name": "unrequested_calculation",
                    "result": "acknowledge",
                    "model_action": "calculate",
                }
            )
        if proposal.requested_action == "knowledge" and not knowledge_query_is_eligible(
            proposal
        ):
            raise ValueError("EMPTY_KNOWLEDGE_QUERY")
        if proposal.requested_action == "calculate" and (
            proposal.subject != "self" or proposal.conditional
        ):
            raise ValueError("NONPERSONAL_CALCULATION")
        # Source-proved negation/ambiguity wins over a well-formed selection.
        # It is deliberately derived from the original state before inference.
        if restrictive.changes and {"NOT", "DO_NOT", "MAYBE", "OR"}.intersection(
            state["turn"].model_projection
        ):
            clarify_decision("source_validation", "ambiguous_or_negated")
            return {
                "memory": restrictive.state,
                "staged": restrictive,
                "response": {
                    "status": "CLARIFY",
                    "answer": FIXED_CLARIFICATION,
                    "memory_receipt": memory_receipt(restrictive.state, restrictive),
                },
            }
        staged = stage_turn(
            state["memory"],
            state["turn"],
            proposal,
            client_message_id=state["message_id"],
        )
    except Exception as error:  # noqa: BLE001 - all model/provenance errors fail closed.
        return _restrictive(
            state,
            "INTERPRETATION_FAILED",
            "interpretation",
            cause=_rejection_cause(error),
        )
    return {"interpreted": result, "staged": staged}


def _calculation(memory: Memory) -> dict[str, Any]:
    loan_fact, loan, loan_proof = resolve_numeric_fact(memory, "existing_loan_amount")
    value_fact, value, value_proof = resolve_numeric_fact(memory, "property_value")
    client_by_turn = {
        turn_ref: message_id for message_id, turn_ref in memory.client_turns.items()
    }
    raw = calculate_ltv(loan, value)
    return LtvCalculation.model_validate(
        {
            **raw,
            "calculation_id": f"calc_r{memory.revision}_{loan_fact.fact_id}_{value_fact.fact_id}",
            "input_references": [
                {
                    "role": "loan_amount",
                    "message_id": client_by_turn[loan_proof.turn_ref],
                    "quote": loan_proof.original,
                    "span": {"start": loan_proof.start, "end": loan_proof.end},
                },
                {
                    "role": "property_value",
                    "message_id": client_by_turn[value_proof.turn_ref],
                    "quote": value_proof.original,
                    "span": {"start": value_proof.start, "end": value_proof.end},
                },
            ],
        }
    ).model_dump()


async def act(state: GraphState) -> dict[str, Any]:
    if state.get("response"):
        return {}
    staged, memory = state["staged"], state["staged"].state
    if staged.proposal_rejected:
        clarify_decision("memory_validation", "proposal_rejected")
        return {
            "memory": memory,
            "response": {
                "status": "CLARIFY",
                "answer": FIXED_CLARIFICATION,
                "memory_receipt": memory_receipt(memory, staged),
            },
        }
    if memory.draft_requested:
        memory, response = complete_memory_draft(memory, staged)
        return {"memory": memory, "response": response}
    action = choose_action(memory, tuple(item.field for item in staged.changes))
    if action == "calculate":
        try:
            calculation = _calculation(memory)
        except Exception:  # noqa: BLE001 - no proof/value detail leaves this boundary.
            return _restrictive(state, "CALCULATION_FAILED", "calculation")
        answer = f"Your loan-to-value ratio is {calculation['percent']}%. This is the loan amount divided by the property value, not a lending decision."
        memory = replace(memory, active_operation="calculate")
        return {
            "memory": memory,
            "response": {
                "status": "ANSWERED",
                "answer": answer,
                "calculations": [calculation],
                "memory_receipt": memory_receipt(memory, staged),
            },
        }
    if action == "retrieve":
        if (
            not state.get("retriever")
            or not state.get("selector")
            or not state.get("evidence")
        ):
            return _restrictive(state, "KNOWLEDGE_QUERY_UNAVAILABLE", "evidence")
        # Retrieval/selector failures cannot acknowledge staged positives.
        answer = await answer_public_question(
            state["turn"].original,
            retriever=state["retriever"],
            selector=state["selector"],
            evidence=state["evidence"],
            scope=state["public_questions"],
        )
        if answer.status == "unsupported":
            return _restrictive(state, "KNOWLEDGE_QUERY_UNAVAILABLE", "evidence")
        if answer.status == "unavailable":
            return _restrictive(state, "EVIDENCE_UNAVAILABLE", "evidence")
        return {
            "memory": memory,
            "public_knowledge": True,
            "response": {
                "status": "ANSWERED",
                **answer.rendered,
                "memory_receipt": memory_receipt(memory, staged),
            },
        }
    pending = choose_pending_question(memory, action)
    if pending:
        labels = {
            "existing_loan_amount": "your existing mortgage amount",
            "property_value": "your property value",
            "property_type": "your property type",
        }
        memory = replace(memory, pending_question=pending)
        clarify_decision("missing_fact", "pending_question")
        return {
            "memory": memory,
            "response": {
                "status": "CLARIFY",
                "answer": f"Please provide {labels[pending.field]}.",
                "memory_receipt": memory_receipt(memory, staged),
            },
        }
    if action == "clarify" and memory.last_operation == "calculate":
        # A calculation request without both amounts and without a personal basis
        # for a bound follow-up: ask for the amounts in the same chat.
        missing = [
            label
            for name, label in (
                ("existing_loan_amount", "your existing mortgage amount"),
                ("property_value", "your property value"),
            )
            if getattr(memory.facts.get(name), "status", None) != FactStatus.KNOWN
        ]
        clarify_decision("missing_fact", "calculation_inputs")
        return {
            "memory": memory,
            "response": {
                "status": "CLARIFY",
                "answer": f"To calculate your loan-to-value, please provide {' and '.join(missing)}.",
                "memory_receipt": memory_receipt(memory, staged),
            },
        }
    return {
        "memory": memory,
        "response": {
            "status": "ANSWERED",
            "answer": "I have recorded the supported details.",
            "memory_receipt": memory_receipt(memory, staged),
        },
    }


def build_graph():
    graph = StateGraph(GraphState)
    graph.add_node("interpret", interpret)
    graph.add_node("act", act)
    graph.add_edge(START, "interpret")
    graph.add_edge("interpret", "act")
    graph.add_edge("act", END)
    return graph.compile()
