import pytest

from grounded_mortgage_assistant.guardrails.privacy import ingest
from grounded_mortgage_assistant.state.memory import (
    ContextLimits,
    Fact,
    FactStatus,
    Memory,
    MemoryError,
    SafeExchange,
    build_context,
)


def test_retrieval_evidence_appears_once_and_history_keeps_complete_exchange_pairs():
    state = Memory()
    context = build_context(
        state, ContextLimits(input_tokens=100, output_reservation=10), (("E1", "E2"),)
    )
    assert context["evidence"] == ("E1", "E2")
    assert str(context).count("E1") == 1
    assert (
        context["token_estimate_kind"]
        == "conservative_symbol_estimate_not_tokenizer_count"
    )


def test_invalid_budget_is_controlled():
    with pytest.raises(MemoryError, match="CONTEXT_BUDGET_EXCEEDED"):
        build_context(Memory(), ContextLimits(input_tokens=1, output_reservation=2))


def test_mandatory_latest_turn_is_never_pruned_and_can_overflow():
    current = ingest("Thanks.")
    context = build_context(
        Memory(),
        ContextLimits(input_tokens=100, output_reservation=10),
        current_turn=current,
    )
    assert current.model_projection[0] in context["state"]
    with pytest.raises(MemoryError, match="CONTEXT_BUDGET_EXCEEDED"):
        build_context(
            Memory(),
            ContextLimits(input_tokens=2, output_reservation=1),
            current_turn=current,
        )


def test_input_ceiling_does_not_subtract_output_without_provider_capacity():
    counter = lambda tokens: 8192 if tokens else 0
    assert (
        build_context(
            Memory(), ContextLimits(input_tokens=8192, token_counter=counter)
        )["token_estimate"]
        == 8192
    )
    with pytest.raises(MemoryError, match="CONTEXT_BUDGET_EXCEEDED"):
        build_context(
            Memory(),
            ContextLimits(input_tokens=8192, mandatory_tokens=1, token_counter=counter),
        )
    for reservation in (1, 8000):
        assert (
            build_context(
                Memory(),
                ContextLimits(
                    input_tokens=10,
                    output_reservation=reservation,
                    token_counter=lambda tokens: 10 if tokens else 0,
                ),
            )["token_estimate"]
            == 10
        )


def test_provider_capacity_reserves_output_from_provider_budget_only():
    counter = lambda tokens: 7200 if tokens else 0
    assert (
        build_context(
            Memory(),
            ContextLimits(
                input_tokens=8192,
                output_reservation=1800,
                model_context_tokens=9000,
                token_counter=counter,
            ),
        )["token_estimate"]
        == 7200
    )
    with pytest.raises(MemoryError, match="CONTEXT_BUDGET_EXCEEDED"):
        build_context(
            Memory(),
            ContextLimits(
                input_tokens=8192,
                output_reservation=1800,
                model_context_tokens=9000,
                mandatory_tokens=1,
                token_counter=counter,
            ),
        )


def test_context_contains_safe_fact_identity_and_current_turn_once():
    current = ingest("Thanks.")
    state = Memory(
        facts={
            "existing_loan_amount": Fact(
                "f_opaque", "existing_loan_amount", 3, FactStatus.KNOWN
            )
        },
        exchanges=(
            SafeExchange(current.turn_ref, current.model_projection),
            SafeExchange("older", ("THANKS",)),
        ),
    )
    context = build_context(
        state,
        ContextLimits(input_tokens=100, output_reservation=1),
        current_turn=current,
    )
    assert "FACT(f_opaque,existing_loan_amount,r3,known)" in context["state"]
    assert context["state"].count("THANKS") == 1
    assert context["history"] == (("THANKS",),)
    assert "1,560,000" not in str(context)
