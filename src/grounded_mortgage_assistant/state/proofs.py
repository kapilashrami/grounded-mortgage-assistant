"""Shared resolution of retained memory proofs to their exact original sources."""

from __future__ import annotations

from decimal import Decimal

from grounded_mortgage_assistant.guardrails.privacy import OriginalProof
from grounded_mortgage_assistant.state.memory import (
    Fact,
    FactStatus,
    Memory,
    analyze_financial_turn,
)
from grounded_mortgage_assistant.tools.calculations import parse_sek
from grounded_mortgage_assistant.tools.schemas import UserTurn


def resolve_numeric_fact(
    memory: Memory, field: str
) -> tuple[Fact, Decimal, OriginalProof]:
    """Resolve a current numeric fact only when its private proof still owns it."""
    fact = memory.facts[field]
    if (
        fact.status is not FactStatus.KNOWN
        or not fact.value_ref
        or not isinstance(fact.original_proof, OriginalProof)
    ):
        raise ValueError("unsupported calculation fact")
    protected = memory.protected_values[fact.value_ref]
    turn = memory.turns[protected.owner_turn_ref]
    proof = fact.original_proof
    if (
        proof.turn_ref != protected.owner_turn_ref
        or proof != protected.original_proof
        or turn.original[proof.start : proof.end] != proof.original
    ):
        raise ValueError("proof ownership")
    if not protected.currency_explicit:
        analysis = analyze_financial_turn(memory, turn)
        if not any(
            clause.field == field
            and fact.value_ref in clause.value_refs
            and clause.kind == "affirmed"
            for clause in analysis.clauses
        ):
            raise ValueError("implicit currency binding")
    if (
        parse_sek(proof.original, "en", allow_unitless=not protected.currency_explicit)
        != protected.value
    ):
        raise ValueError("proof decimal mismatch")
    return fact, protected.value, proof


def resolve_original_turn(memory: Memory, proof: OriginalProof) -> UserTurn:
    """Return the one retained client original that owns an exact private proof."""
    turn = memory.turns.get(proof.turn_ref)
    if turn is None or turn.original[proof.start : proof.end] != proof.original:
        raise ValueError("proof ownership")
    client_ids = [
        message_id
        for message_id, turn_ref in memory.client_turns.items()
        if turn_ref == proof.turn_ref and message_id not in memory.forgotten_client_ids
    ]
    if len(client_ids) != 1:
        raise ValueError("ambiguous proof source")
    return UserTurn(
        message_id=client_ids[0], text=turn.original, notation_language="en"
    )
