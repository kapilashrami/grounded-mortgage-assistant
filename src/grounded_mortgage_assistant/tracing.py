"""Safe per-request execution events shown in the engineering console."""

from __future__ import annotations

from contextvars import ContextVar
from typing import Any

TRACE_EVENTS: ContextVar[list[dict[str, Any]] | None] = ContextVar(
    "trace_events", default=None
)


def trace_event(event: dict[str, Any]) -> None:
    events = TRACE_EVENTS.get()
    if events is not None:
        events.append(event)


_CLARIFY_DECISIONS = frozenset(
    {
        ("evidence", "evidence_unavailable"),
        ("public_projection", "ambiguous"),
        ("public_projection", "unsupported"),
        ("missing_fact", "calculation_inputs"),
        ("missing_fact", "pending_question"),
        ("calculation", "calculation_unavailable"),
        ("draft", "draft_unavailable"),
        ("draft", "scope_unavailable"),
        ("interpretation", "interpretation_unavailable"),
        ("source_validation", "ambiguous_or_negated"),
        ("memory_validation", "proposal_rejected"),
    }
)


def clarify_decision(
    layer: str, reason: str, events: list[dict[str, Any]] | None = None
) -> None:
    """Record a fixed, safe explanation for an authoritative clarification."""
    if (layer, reason) not in _CLARIFY_DECISIONS:
        raise ValueError("unsupported clarification decision")
    event = {
        "name": "clarify_decision",
        "result": "stopped",
        "layer": layer,
        "reason": reason,
    }
    if events is not None:
        events.append(event)
    else:
        trace_event(event)
