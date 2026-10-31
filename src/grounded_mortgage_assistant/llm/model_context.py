"""Safe, bounded context construction for the local turn interpreter."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from grounded_mortgage_assistant.llm.interpretation_candidates import CandidateCatalog
from grounded_mortgage_assistant.state.memory import (
    Memory,
    SafeExchange,
    SanitizedTurn,
    assertion_unit_ranges,
)


class ModelContextError(ValueError):
    """Fixed category for a context that cannot be safely sent."""


@dataclass(frozen=True)
class PackedContext:
    messages: tuple[dict[str, str], ...]
    history: tuple[SafeExchange, ...]


def _dump(value: object) -> str:
    return json.dumps(value, separators=(",", ":"), sort_keys=True)


def _turn_projection(state: Memory, turn: SanitizedTurn) -> dict[str, object]:
    return {
        "turn_ref": turn.turn_ref,
        "tokens": list(turn.model_projection),
        "assertion_units": [
            [span.token_start, span.token_end] for span in assertion_unit_ranges(turn)
        ],
    }


def _candidate_projection(catalog: CandidateCatalog) -> dict[str, object]:
    return {
        "facts": [
            {
                "id": candidate_id,
                "field": candidate.field,
                "operation": candidate.operation,
                "unit": [
                    candidate.source_span.token_start,
                    candidate.source_span.token_end,
                ],
                "value_handles": list(candidate.value_refs),
                "enum": candidate.enum_value,
                "enums": list(candidate.enum_values),
            }
            for candidate_id, candidate in catalog.fact_candidates.items()
        ],
        "scope": [
            {
                "id": candidate_id,
                "unit": [candidate.span.token_start, candidate.span.token_end],
                "concepts": dict(candidate.concepts),
            }
            for candidate_id, candidate in catalog.scope_candidates.items()
        ],
    }


def _state_projection(state: Memory) -> dict[str, object]:
    facts = [
        {
            "fact_id": fact.fact_id,
            "field": fact.field,
            "revision": fact.field_revision,
            "status": fact.status.value,
            "enum": fact.enum_value,
            "enum_candidates": list(fact.enum_values),
        }
        for fact in state.facts.values()
    ]
    pending = (
        None
        if state.pending_question is None
        else {
            "field": state.pending_question.field,
            "asked_revision": state.pending_question.asked_revision,
            "question_id": state.pending_question.question_id,
        }
    )
    return {
        "revision": state.revision,
        "facts": facts,
        "goal": state.goal,
        "scope": state.scope,
        "borrowing": state.borrowing,
        "last_operation": state.last_operation,
        "pending": pending,
    }


_INSTRUCTIONS = (
    "Interpret only canonical tokens. Call interpret_turn exactly once. Prior state "
    "is committed. Candidates are uncommitted current-turn evidence: select only "
    "applicable candidate IDs. Selecting a scope candidate applies all of its concepts "
    "atomically. requested_action classifies the customer's latest intent and is "
    "independent of goal, ownership, and borrowing. Prior state helps resolve "
    "references but cannot replace the latest request. Missing prerequisites or an "
    "empty update-candidate set do not change that intent; deterministic code handles "
    "follow-ups. A pending question is context, not an instruction to ignore a new "
    "request. "
    "acknowledge is for supplied facts or stated goals without an explicit next "
    "operation. knowledge is for an information question. calculate is for an explicit "
    "calculation. A personal LTV question is calculate; a concept-definition question "
    "is knowledge; stated goals and property facts are acknowledge. review is for an "
    "explicit review request. decline is for a decline. "
    "forget is for a forget request. clarify is only for a latest meaning that cannot "
    "be interpreted. "
    "Questions, quotation, negation, third-party and hypothetical text do not "
    "establish personal facts. Never confirm a customer action."
)


def build_model_context(
    state: Memory,
    turn: SanitizedTurn,
    catalog: CandidateCatalog,
    *,
    history: tuple[SafeExchange, ...] | None = None,
) -> PackedContext:
    """Build explicit safe messages. Private proof/value objects never cross here."""
    history = state.exchanges if history is None else history
    safe_history = []
    for exchange in history:
        safe_history.append(
            {
                "turn_ref": exchange.turn_ref,
                "tokens": list(exchange.tokens),
                "response": list(exchange.response),
            }
        )
    messages: list[dict[str, str]] = [
        {
            "role": "system",
            "content": _INSTRUCTIONS
            + "\nSAFE_STATE_AND_HISTORY="
            + _dump({"state": _state_projection(state), "history": safe_history}),
        }
    ]
    messages.append(
        {
            "role": "user",
            "content": _dump(
                {
                    "current": _turn_projection(state, turn),
                    "candidates": _candidate_projection(catalog),
                }
            ),
        }
    )
    return PackedContext(tuple(messages), tuple(history))


def token_count(
    tokenizer: Any, messages: tuple[dict[str, str], ...], tools: list[dict[str, Any]]
) -> int:
    """Use the matching chat template, never a character or symbol estimate."""
    try:
        encoded = tokenizer.apply_chat_template(
            list(messages),
            tools=tools,
            tokenize=True,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        if isinstance(encoded, dict) or hasattr(encoded, "keys"):
            ids = encoded["input_ids"]
        else:
            ids = encoded
        if hasattr(ids, "tolist"):
            ids = ids.tolist()
        if not isinstance(ids, list):
            raise ModelContextError("TOKENIZER_FAILURE")
        if ids and isinstance(ids[0], list):
            if len(ids) != 1:
                raise ModelContextError("TOKENIZER_FAILURE")
            ids = ids[0]
        if any(isinstance(item, list) for item in ids):
            raise ModelContextError("TOKENIZER_FAILURE")
        return len(ids)
    except Exception as error:  # tokenizer errors are deliberately non-echoing
        raise ModelContextError("TOKENIZER_FAILURE") from error


def pack_model_context(
    state: Memory,
    turn: SanitizedTurn,
    catalog: CandidateCatalog,
    *,
    tokenizer: Any,
    tools: list[dict[str, Any]],
    provider_capacity: int,
    input_limit: int = 8192,
    output_reservation: int = 1800,
) -> tuple[PackedContext, int]:
    if provider_capacity <= 0 or input_limit <= 0 or output_reservation <= 0:
        raise ModelContextError("INVALID_CONTEXT_LIMIT")
    # A caller may carry the current turn in safe exchange history only after it
    # has been committed. It must not appear alongside its mandatory projection.
    history = tuple(item for item in state.exchanges if item.turn_ref != turn.turn_ref)
    while True:
        packed = build_model_context(state, turn, catalog, history=history)
        count = token_count(tokenizer, packed.messages, tools)
        if count <= input_limit and count + output_reservation <= provider_capacity:
            return packed, count
        if not history:
            raise ModelContextError("CONTEXT_BUDGET_EXCEEDED")
        history = history[1:]
