"""Retrieval-first answers for public questions.

The reranker score of the best record decides whether the corpus covers the
question. The selector only chooses among the packed records; it never decides
coverage.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any, Literal

from grounded_mortgage_assistant.rag.public_evidence import PublicEvidence
from grounded_mortgage_assistant.rag.public_questions import PublicQuestionScope

# Chosen by scripts/calibrate_relevance_gate.py over the audited English control cases
# (private); the summary is in docs/results/relevance-gate.json.
# At 0.23: 142 of 146 supported cases kept (97.3%), 7 of 9 unsupported rejected.
RELEVANCE_THRESHOLD = 0.23


@dataclass(frozen=True)
class PublicAnswer:
    status: Literal["answered", "unsupported", "unavailable"]
    records: list[dict[str, Any]]
    top_record_id: str | None
    top_score: float | None
    rendered: dict[str, Any] | None


async def _maybe(value: Any) -> Any:
    return await value if hasattr(value, "__await__") else value


def _score(record: dict[str, Any]) -> float | None:
    score = record.get("score")
    if isinstance(score, bool) or not isinstance(score, int | float):
        return None
    return float(score)


async def answer_public_question(
    question: str,
    *,
    retriever: Callable[[str], Awaitable[list[dict[str, Any]]] | list[dict[str, Any]]],
    selector: Callable[[str, list[dict[str, Any]]], Awaitable[Any] | Any],
    evidence: PublicEvidence,
    scope: PublicQuestionScope,
) -> PublicAnswer:
    """Answer from the corpus, or report that it does not cover the question."""
    try:
        records = await _maybe(retriever(question))
        records = scope.scope_partner_records(question, records)
        if not records:
            return PublicAnswer("unsupported", [], None, None, None)
        top_record_id = records[0]["id"]
        top_score = _score(records[0])
        if top_score is None or top_score < RELEVANCE_THRESHOLD:
            return PublicAnswer("unsupported", records, top_record_id, top_score, None)
        # The selector may pick any packed record, so offer only renderable ones.
        records = [row for row in records if evidence.is_citable(row["id"])]
        if not records:
            return PublicAnswer("unsupported", [], top_record_id, top_score, None)
        selected = await _maybe(selector(question, records))
        if selected.needs_clarification or not selected.evidence_ids:
            return PublicAnswer("unavailable", records, top_record_id, top_score, None)
        rendered = evidence.render(selected.evidence_ids, records)
        return PublicAnswer("answered", records, top_record_id, top_score, rendered)
    except Exception:  # noqa: BLE001 - public retrieval must fail closed.
        return PublicAnswer("unavailable", [], None, None, None)
