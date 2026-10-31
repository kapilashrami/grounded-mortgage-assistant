from dataclasses import replace
from datetime import datetime, timedelta

import pytest

from grounded_mortgage_assistant.guardrails.privacy import OriginalProof, ingest
from grounded_mortgage_assistant.state.memory import (
    FactOperation,
    Memory,
    SourceSpan,
    TurnInterpretation,
    parse_scope_unit,
    stage_turn,
)
from grounded_mortgage_assistant.tools.draft import (
    MemoryDraftError,
    prepare_memory_draft,
)

NOW = datetime.fromisoformat("2026-10-30T16:00:00+02:00")


def proposal(
    turn,
    *,
    operations=(),
    goal="unchanged",
    ownership="unchanged",
    borrowing="unchanged",
):
    scope_source_spans = []
    if any(value != "unchanged" for value in (goal, ownership, borrowing)):
        end = next(
            (
                index
                for index, token in enumerate(turn.tokens)
                if token.symbol
                in {"SENTENCE_END", "QUESTION", "EXCLAMATION", "SEMICOLON"}
            ),
            len(turn.tokens),
        )
        scope_source_spans = [{"token_start": 0, "token_end": end}]
    return TurnInterpretation(
        subject="third_party" if "THIRD_PARTY" in turn.model_projection else "self",
        conditional="IF" in turn.model_projection,
        operations=list(operations),
        goal=goal,
        borrowing=borrowing,
        ownership=ownership,
        requested_action="acknowledge",
        scope_source_spans=scope_source_spans,
        unresolved=[],
    )


def money_operation(turn, field, index):
    return FactOperation(
        field=field,
        operation="set",
        value_refs=[turn.protected_values[index].value_ref],
        source_span={"token_start": 0, "token_end": len(turn.tokens) - 1},
    )


def property_operation(turn):
    return FactOperation(
        field="property_type",
        operation="set",
        enum_value="house",
        source_span={"token_start": 0, "token_end": len(turn.tokens) - 1},
    )


def complete_memory(
    scope_text="I want to refinance my existing mortgage without increasing my loan amount.",
):
    scope = ingest(scope_text, turn_ref="scope")
    state = stage_turn(
        Memory(),
        scope,
        proposal(scope, goal="refinance", ownership="own_existing", borrowing="same"),
        client_message_id="scope-message",
    ).state
    loan = ingest("My mortgage is SEK 1,820,000.", turn_ref="loan")
    state = stage_turn(
        state,
        loan,
        proposal(loan, operations=[money_operation(loan, "existing_loan_amount", 0)]),
        client_message_id="loan-message",
    ).state
    value = ingest("My property value is SEK 2,600,000.", turn_ref="value")
    state = stage_turn(
        state,
        value,
        proposal(value, operations=[money_operation(value, "property_value", 0)]),
        client_message_id="value-message",
    ).state
    property_turn = ingest("My property is house.", turn_ref="property")
    return stage_turn(
        state,
        property_turn,
        proposal(property_turn, operations=[property_operation(property_turn)]),
        client_message_id="property-message",
    ).state


def test_memory_draft_uses_v1_supported_current_affirmation_and_exact_sources():
    memory = complete_memory(
        "I want to refinance my existing mortgage without increasing my loan amount."
    )

    draft = prepare_memory_draft(memory, now=NOW, draft_id="explicit-draft")

    assert draft.draft.ready
    assert draft.draft.fields == {
        "existing_loan_amount": 1820000,
        "property_value": 2600000,
        "property_type": "house",
    }
    assert draft.draft.draft_id == "explicit-draft"
    assert (draft.draft.revision, draft.draft.base_revision) == (1, 0)
    assert draft.draft.created_at == NOW.isoformat()
    assert draft.draft.expires_at == (NOW + timedelta(minutes=30)).isoformat()
    for proof in [
        *draft.draft.field_provenance.values(),
        *draft.draft.intent_provenance.values(),
    ]:
        source = next(
            turn for turn in draft.source_turns if turn.message_id == proof.message_id
        )
        assert source.text[proof.start : proof.end] == proof.quote


@pytest.mark.parametrize("possessive", ["my", "our"])
def test_memory_scope_parser_accepts_canonical_loan_amount_suffixes(possessive):
    turn = ingest(
        f"I want to refinance my existing mortgage without increasing {possessive} loan amount."
    )
    end = next(
        index
        for index, token in enumerate(turn.tokens)
        if token.symbol == "SENTENCE_END"
    )

    assert parse_scope_unit(turn, SourceSpan(token_start=0, token_end=end)) == {
        "goal": "refinance",
        "scope": "own_existing",
        "borrowing": "same",
    }


def test_extra_borrowing_suffix_proves_no_increase_for_the_draft():
    # The canonical scope grammar treats "without extra borrowing" as a
    # no-increase suffix; the draft must accept the same proof it validates.
    text = "I want to refinance my existing mortgage without extra borrowing."
    memory = complete_memory(text)

    draft = prepare_memory_draft(memory, now=NOW, draft_id="explicit-draft")

    assert draft.draft.ready
    assert draft.draft.intent == "refinance_without_increase"
    assert draft.draft.intent_provenance["no_increase"].quote == text.rstrip(".")


@pytest.mark.parametrize(
    "scope_text",
    [
        "I want to increase my mortgage amount.",
        "I do not want to refinance my existing mortgage without increasing my loan amount.",
        "I want to refinance their mortgage without increasing my loan amount.",
        "If I want to refinance my existing mortgage without increasing my loan amount.",
    ],
)
def test_nonpositive_scope_variants_cannot_prepare_memory_draft(scope_text):
    scope = ingest(scope_text, turn_ref="scope")
    state = stage_turn(
        Memory(),
        scope,
        proposal(scope),
        client_message_id="scope-message",
    ).state

    with pytest.raises(MemoryDraftError, match="MEMORY_DRAFT_UNAVAILABLE"):
        prepare_memory_draft(state, now=NOW, draft_id="explicit-draft")


def test_current_compatible_scope_replaces_old_proof_and_revises_memory():
    old = ingest(
        "I want to refinance my existing mortgage without extra borrowing.",
        turn_ref="old-scope",
    )
    state = stage_turn(
        Memory(),
        old,
        proposal(old, goal="refinance", ownership="own_existing", borrowing="same"),
        client_message_id="old-scope-message",
    ).state
    new = ingest(
        "I want to refinance my existing mortgage without increasing my loan amount.",
        turn_ref="new-scope",
    )
    updated = stage_turn(
        state,
        new,
        proposal(new, goal="refinance", ownership="own_existing", borrowing="same"),
        client_message_id="new-scope-message",
    ).state

    assert updated.revision == state.revision + 1
    assert {proof.turn_ref for proof in updated.scope_proofs["goal"]} == {"new-scope"}


@pytest.mark.parametrize("mutation", ["fractional", "range", "foreign", "duplicate"])
def test_memory_draft_rejects_invalid_or_ambiguous_authoritative_sources(mutation):
    memory = complete_memory()
    if mutation == "fractional":
        fact = memory.facts["existing_loan_amount"]
        bad = ingest("My mortgage is SEK 1.5.", turn_ref="loan")
        protected = bad.protected_values[0]
        memory = replace(
            memory,
            protected_values={
                **{
                    key: value
                    for key, value in memory.protected_values.items()
                    if key != fact.value_ref
                },
                protected.value_ref: protected,
            },
            turns={**memory.turns, "loan": bad},
            facts={
                **memory.facts,
                "existing_loan_amount": replace(
                    fact,
                    value_ref=protected.value_ref,
                    original_proof=protected.original_proof,
                ),
            },
        )
    elif mutation == "range":
        fact = memory.facts["property_value"]
        bad = ingest("My property value is SEK 100.", turn_ref="value")
        protected = bad.protected_values[0]
        memory = replace(
            memory,
            protected_values={
                **{
                    key: value
                    for key, value in memory.protected_values.items()
                    if key != fact.value_ref
                },
                protected.value_ref: protected,
            },
            turns={**memory.turns, "value": bad},
            facts={
                **memory.facts,
                "property_value": replace(
                    fact,
                    value_ref=protected.value_ref,
                    original_proof=protected.original_proof,
                ),
            },
        )
    elif mutation == "foreign":
        fact = memory.facts["property_type"]
        memory = replace(
            memory,
            facts={
                **memory.facts,
                "property_type": replace(
                    fact,
                    original_proof=OriginalProof("foreign", 0, 4, "house"),
                ),
            },
        )
    else:
        proof = memory.facts["existing_loan_amount"].original_proof
        turn = memory.turns[proof.turn_ref]
        memory = replace(
            memory,
            turns={
                **memory.turns,
                proof.turn_ref: replace(
                    turn, original=f"{turn.original} {proof.original}"
                ),
            },
        )

    with pytest.raises(MemoryDraftError, match="MEMORY_DRAFT_UNAVAILABLE"):
        prepare_memory_draft(memory, now=NOW, draft_id="explicit-draft")
