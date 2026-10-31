"""Request-local, server-owned candidates for native turn selection."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator
from pydantic_core import PydanticCustomError

from grounded_mortgage_assistant.guardrails.privacy import SanitizedTurn
from grounded_mortgage_assistant.state.memory import (
    SAFE_KNOWLEDGE_TOKENS,
    FactOperation,
    FinancialAnalysis,
    Memory,
    SourceSpan,
    Subject,
    TurnInterpretation,
    analyze_financial_turn,
    assertion_unit_ranges,
    candidate_fact_operations,
    parse_scope_unit,
    scope_property_type,
)

RequestedAction = Literal[
    "knowledge", "calculate", "acknowledge", "review", "decline", "forget", "clarify"
]


class TurnSelection(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    subject: Subject
    conditional: bool
    requested_action: RequestedAction = Field(
        description=(
            "Classify the customer's latest intent, not an assistant workflow step: "
            "acknowledge supplied facts or stated goals without an explicit operation; "
            "knowledge for an explicit information request; calculate for an explicit "
            "calculation; review for an explicit review request; decline for a decline; "
            "forget for a forget request; clarify only when the latest meaning cannot "
            "be interpreted. Missing prerequisites and pending questions do not replace "
            "the latest request; deterministic code handles follow-ups."
        )
    )
    fact_candidate_ids: list[str] = Field(
        max_length=3,
        description=(
            "CURRENT TURN newly asserted or changed fact candidates only. Never select "
            "read-only stored fact IDs. Use [] when there are no candidates."
        ),
    )
    scope_candidate_ids: list[str] = Field(
        max_length=4,
        description=(
            "CURRENT TURN scope candidates only. Use [] when there are no candidates. "
            "Calculations consume validated server memory without selecting past facts."
        ),
    )

    @model_validator(mode="after")
    def coherent(self):
        if len(set(self.fact_candidate_ids)) != len(self.fact_candidate_ids):
            raise PydanticCustomError(
                "duplicate_fact_candidate_ids", "duplicate fact candidate IDs"
            )
        if len(set(self.scope_candidate_ids)) != len(self.scope_candidate_ids):
            raise PydanticCustomError(
                "duplicate_scope_candidate_ids", "duplicate scope candidate IDs"
            )
        return self


class CandidateSelectionError(ValueError):
    def __init__(
        self, category: Literal["INVALID_CANDIDATE_SELECTION", "CONFLICTING_CANDIDATES"]
    ):
        self.category = category
        super().__init__(category)


@dataclass(frozen=True)
class ScopeCandidate:
    span: SourceSpan
    concepts: Mapping[str, str]


@dataclass(frozen=True)
class CandidateCatalog:
    turn_ref: str
    fact_candidates: Mapping[str, FactOperation]
    scope_candidates: Mapping[str, ScopeCandidate]
    knowledge_projection: tuple[str, ...]


def build_candidate_catalog(state: Memory, turn: SanitizedTurn) -> CandidateCatalog:
    analysis: FinancialAnalysis = analyze_financial_turn(state, turn)
    facts = sorted(
        candidate_fact_operations(state, turn, analysis),
        key=lambda item: (item.source_span.token_start, item.field),
    )
    scopes: list[ScopeCandidate] = []
    for span in assertion_unit_ranges(turn):
        concepts = parse_scope_unit(turn, span)
        if concepts:
            scopes.append(ScopeCandidate(span, MappingProxyType(dict(concepts))))
        property_type = scope_property_type(turn, span)
        if property_type is not None:
            facts.append(
                FactOperation(
                    field="property_type",
                    operation="set",
                    source_span=span,
                    enum_value=property_type,
                )
            )
    fact_candidates = {
        f"fc_{turn.turn_ref}_{index}": candidate
        for index, candidate in enumerate(
            sorted(facts, key=lambda item: (item.source_span.token_start, item.field)),
            start=1,
        )
    }
    scope_candidates = {
        f"sc_{turn.turn_ref}_{index}": candidate
        for index, candidate in enumerate(scopes, start=1)
    }
    return CandidateCatalog(
        turn.turn_ref,
        MappingProxyType(fact_candidates),
        MappingProxyType(scope_candidates),
        tuple(
            dict.fromkeys(
                symbol
                for symbol in turn.model_projection
                if symbol in SAFE_KNOWLEDGE_TOKENS
            )
        ),
    )


def knowledge_query_is_eligible(interpretation: TurnInterpretation) -> bool:
    """Only a nonempty server projection may enter a future retrieval selector."""
    return bool(
        interpretation.requested_action == "knowledge"
        and interpretation.knowledge_question
    )


def expand_selection(
    catalog: CandidateCatalog, selection: TurnSelection
) -> TurnInterpretation:
    fact_operations: list[FactOperation] = []
    for candidate_id in selection.fact_candidate_ids:
        candidate = catalog.fact_candidates.get(candidate_id)
        if candidate is None:
            raise CandidateSelectionError("INVALID_CANDIDATE_SELECTION")
        if any(item.field == candidate.field for item in fact_operations):
            raise CandidateSelectionError("CONFLICTING_CANDIDATES")
        fact_operations.append(candidate)
    fact_operations.sort(key=lambda item: (item.source_span.token_start, item.field))
    changes = {"goal": "unchanged", "scope": "unchanged", "borrowing": "unchanged"}
    selected_scopes: list[ScopeCandidate] = []
    for candidate_id in selection.scope_candidate_ids:
        candidate = catalog.scope_candidates.get(candidate_id)
        if candidate is None:
            raise CandidateSelectionError("INVALID_CANDIDATE_SELECTION")
        for name, value in candidate.concepts.items():
            prior = changes[name]
            if prior not in {"unchanged", value}:
                raise CandidateSelectionError("CONFLICTING_CANDIDATES")
            changes[name] = value
        selected_scopes.append(candidate)
    spans = [
        candidate.span
        for candidate in sorted(selected_scopes, key=lambda item: item.span.token_start)
    ]
    try:
        return TurnInterpretation.model_validate(
            {
                "subject": selection.subject,
                "conditional": selection.conditional,
                "operations": fact_operations,
                "goal": changes["goal"],
                "ownership": changes["scope"],
                "borrowing": changes["borrowing"],
                "requested_action": selection.requested_action,
                "knowledge_question": (
                    list(catalog.knowledge_projection)
                    if selection.requested_action == "knowledge"
                    else None
                ),
                "scope_source_spans": spans,
                "unresolved": [],
            },
            strict=True,
        )
    except ValueError as error:
        raise CandidateSelectionError("INVALID_CANDIDATE_SELECTION") from error
