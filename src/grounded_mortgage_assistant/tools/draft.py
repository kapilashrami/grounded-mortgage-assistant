"""Prepare an explicit v1 draft from current, retained memory proofs only."""

from __future__ import annotations

from datetime import datetime, timedelta
from decimal import Decimal

from grounded_mortgage_assistant.guardrails.privacy import OriginalProof
from grounded_mortgage_assistant.state.memory import (
    Fact,
    FactStatus,
    Memory,
    analyze_financial_turn,
    assertion_unit_ranges,
    parse_scope_unit,
    scope_property_type,
)
from grounded_mortgage_assistant.state.proofs import (
    resolve_numeric_fact,
    resolve_original_turn,
)
from grounded_mortgage_assistant.tools.schemas import (
    MortgageDraft,
    MortgageDraftState,
    UserTurn,
)

_FIELDS = ("existing_loan_amount", "property_value", "property_type")


class MemoryDraftError(ValueError):
    """A fixed category for an unsupported explicit memory-to-v1 draft bridge."""

    code = "MEMORY_DRAFT_UNAVAILABLE"

    def __init__(self) -> None:
        super().__init__(self.code)


def prepare_memory_draft(
    memory: Memory, *, now: datetime, draft_id: str
) -> MortgageDraftState:
    """Build a fresh ready v1 draft without scanning unrelated conversation history."""
    try:
        if (
            memory.goal != "refinance"
            or memory.scope != "own_existing"
            or memory.borrowing != "same"
        ):
            raise ValueError("unsupported scope")

        goal_proof = _scope_proof(memory, "goal", "refinance")
        scope_proof = _scope_proof(memory, "scope", "own_existing")
        borrowing_proof = _scope_proof(memory, "borrowing", "same")
        if goal_proof != scope_proof:
            raise ValueError("ambiguous scope source")

        expected: dict[str, tuple[Fact, Decimal | str, OriginalProof]] = {}
        for field in _FIELDS[:2]:
            expected[field] = resolve_numeric_fact(memory, field)
        property_fact = memory.facts["property_type"]
        if (
            property_fact.status is not FactStatus.KNOWN
            or property_fact.enum_value is None
            or not isinstance(property_fact.original_proof, OriginalProof)
        ):
            raise ValueError("unsupported property fact")
        _property_proof(memory, property_fact.original_proof, property_fact.enum_value)
        expected["property_type"] = (
            property_fact,
            property_fact.enum_value,
            property_fact.original_proof,
        )

        expected_values = _expected_values(expected)
        proof_by_name = {
            **{field: value[2] for field, value in expected.items()},
            "personal_refinance": goal_proof,
            "no_increase": borrowing_proof,
        }
        sources: dict[str, UserTurn] = {}
        provenance: dict[str, dict[str, int | str]] = {}
        for name, proof in proof_by_name.items():
            source = resolve_original_turn(memory, proof)
            if source.text.count(proof.original) != 1:
                raise ValueError("ambiguous proof quote")
            sources[source.message_id] = source
            provenance[name] = {
                "message_id": source.message_id,
                "quote": proof.original,
                "start": proof.start,
                "end": proof.end,
            }
        draft = MortgageDraft.model_validate(
            {
                "draft_id": draft_id,
                "revision": 1,
                "base_revision": 0,
                "created_at": now.isoformat(),
                "expires_at": (now + timedelta(minutes=30)).isoformat(),
                "intent": "refinance_without_increase",
                "fields": expected_values,
                "field_provenance": {
                    field: provenance[field] for field in expected_values
                },
                "intent_provenance": {
                    "personal_refinance": provenance["personal_refinance"],
                    "no_increase": provenance["no_increase"],
                },
                "missing_fields": [],
                "issues": [],
                "ready": True,
            }
        )
        return MortgageDraftState(draft=draft, source_turns=list(sources.values()))
    except (KeyError, TypeError, ValueError) as error:
        if isinstance(error, MemoryDraftError):
            raise
        raise MemoryDraftError() from error


def _scope_proof(memory: Memory, name: str, expected: str) -> OriginalProof:
    proofs = memory.scope_proofs.get(name)
    if proofs is None or len(proofs) != 1 or not isinstance(proofs[0], OriginalProof):
        raise ValueError("ambiguous scope proof")
    proof = proofs[0]
    turn = memory.turns.get(proof.turn_ref)
    if turn is None:
        raise ValueError("scope proof ownership")
    for span in assertion_unit_ranges(turn):
        start, end = turn.token_span(span.token_start, span.token_end)
        if (start, end) == (proof.start, proof.end) and parse_scope_unit(
            turn, span
        ).get(name) == expected:
            resolve_original_turn(memory, proof)
            return proof
    raise ValueError("scope proof mismatch")


def _property_proof(memory: Memory, proof: OriginalProof, expected: str | bool) -> None:
    if not isinstance(expected, str):
        raise TypeError("unsupported property fact")
    turn = memory.turns.get(proof.turn_ref)
    if turn is None:
        raise ValueError("property proof ownership")
    for span in assertion_unit_ranges(turn):
        start, end = turn.token_span(span.token_start, span.token_end)
        if (start, end) != (proof.start, proof.end):
            continue
        if scope_property_type(turn, span) == expected or any(
            clause.field == "property_type"
            and clause.kind == "affirmed"
            and clause.enum_values == (expected,)
            for clause in analyze_financial_turn(memory, turn).clauses
        ):
            resolve_original_turn(memory, proof)
            return
    raise ValueError("property proof mismatch")


def _expected_values(
    expected: dict[str, tuple[Fact, Decimal | str, OriginalProof]],
) -> dict[str, int | str]:
    values: dict[str, int | str] = {}
    for field, (_, value, _) in expected.items():
        if field == "property_type":
            values[field] = value
            continue
        if not isinstance(value, Decimal) or value != value.to_integral_value():
            raise ValueError("fractional value")
        values[field] = int(value)
    return values
