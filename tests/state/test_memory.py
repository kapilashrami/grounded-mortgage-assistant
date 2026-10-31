from dataclasses import replace

import pytest

from grounded_mortgage_assistant.guardrails.privacy import (
    CanonicalToken,
    OriginalProof,
    ingest,
)
from grounded_mortgage_assistant.state.memory import (
    ContextLimits,
    FactOperation,
    FactStatus,
    Memory,
    MemoryError,
    PendingQuestion,
    ReplayRecord,
    SafeExchange,
    ScopeObservation,
    SourceSpan,
    TurnInterpretation,
    analyze_financial_turn,
    assertion_unit_ranges,
    build_context,
    choose_action,
    commit_restrictions,
    forget,
    is_explicit_ltv_request,
    parse_scope_unit,
    record_replay,
    replay_response,
    scope_property_type,
    stage_turn,
    stage_uninterpreted_turn,
)


def interpretation(
    turn,
    operations,
    *,
    action="acknowledge",
    subject="self",
    conditional=False,
    goal="unchanged",
    borrowing="unchanged",
    ownership="unchanged",
):
    return TurnInterpretation(
        subject=subject,
        conditional=conditional,
        operations=operations,
        goal=goal,
        borrowing=borrowing,
        ownership=ownership,
        requested_action=action,
        scope_source_spans=(
            [
                {
                    "token_start": 0,
                    "token_end": next(
                        (
                            i
                            for i, token in enumerate(turn.tokens)
                            if token.symbol
                            in {"SENTENCE_END", "QUESTION", "EXCLAMATION", "SEMICOLON"}
                        ),
                        len(turn.tokens),
                    ),
                }
            ]
            if {goal, borrowing, ownership} != {"unchanged"}
            else []
        ),
    )


def money_operation(turn, field, value_index, start, end, operation="set"):
    refs = [turn.protected_values[value_index].value_ref]
    token_index = next(
        index for index, token in enumerate(turn.tokens) if refs[0] in token.symbol
    )
    unit_start = max(
        (
            index + 1
            for index, token in enumerate(turn.tokens[:token_index])
            if token.symbol in {"SENTENCE_END", "QUESTION", "EXCLAMATION", "SEMICOLON"}
        ),
        default=0,
    )
    unit_end = next(
        (
            index
            for index, token in enumerate(turn.tokens[token_index:], token_index)
            if token.symbol in {"SENTENCE_END", "QUESTION", "EXCLAMATION", "SEMICOLON"}
        ),
        len(turn.tokens),
    )
    return FactOperation(
        field=field,
        operation=operation,
        value_refs=refs,
        source_span={"token_start": unit_start, "token_end": unit_end},
    )


def test_correction_replaces_only_target_field_and_invalidates_offer_state():
    first = ingest(
        "My mortgage is SEK 1,820,000 and my property value is SEK 2,600,000."
    )
    state = stage_turn(
        Memory(offer={"id": "old"}),
        first,
        interpretation(
            first,
            [
                money_operation(first, "existing_loan_amount", 0, 0, 5),
                money_operation(first, "property_value", 1, 6, 11),
            ],
        ),
    ).state
    correction = ingest("My mortgage is SEK 1,560,000.")
    result = stage_turn(
        state,
        correction,
        interpretation(
            correction, [money_operation(correction, "existing_loan_amount", 0, 0, 5)]
        ),
    )

    assert result.state.facts["existing_loan_amount"].field_revision == 2
    assert result.state.facts["property_value"].field_revision == 1
    assert result.state.offer is None
    assert set(result.invalidated) == {"calculation", "offer", "review"}


def test_historical_needed_token_does_not_prove_current_refinance_intent():
    turn = ingest("I needed refinance.")
    assert parse_scope_unit(turn, SourceSpan(token_start=0, token_end=3)) == {}


def test_forged_cross_turn_and_conditional_handles_do_not_mutate():
    first = ingest("My mortgage is SEK 1,820,000.")
    other = ingest("My property value is SEK 2,600,000.")
    forged = FactOperation(
        field="existing_loan_amount",
        operation="set",
        value_refs=[other.protected_values[0].value_ref],
        source_span={"token_start": 0, "token_end": 4},
    )
    rejected = stage_turn(Memory(), first, interpretation(first, [forged]))
    assert rejected.proposal_rejected and rejected.rejection_code == "INVALID_PROPOSAL"
    assert rejected.state.facts["existing_loan_amount"].status == FactStatus.AMBIGUOUS
    conditional_turn = ingest("If my mortgage is SEK 1,820,000.")
    valid = money_operation(conditional_turn, "existing_loan_amount", 0, 0, 5)
    with pytest.raises(MemoryError, match="UNPROVEN_PERSONAL_SCOPE"):
        stage_turn(
            Memory(),
            conditional_turn,
            interpretation(conditional_turn, [valid], conditional=True),
        )


def test_ambiguous_amount_blocks_calculation_and_withdraw_removes_fact():
    turn = ingest("My mortgage is SEK 1,560,000 or SEK 1,820,000.")
    op = FactOperation(
        field="existing_loan_amount",
        operation="ambiguous",
        value_refs=[value.value_ref for value in turn.protected_values],
        source_span={"token_start": 0, "token_end": 6},
    )
    state = stage_turn(
        Memory(), turn, interpretation(turn, [op], action="calculate")
    ).state
    assert state.facts["existing_loan_amount"].status == FactStatus.AMBIGUOUS
    assert choose_action(state) == "clarify"
    withdrawn = FactOperation(
        field="existing_loan_amount",
        operation="withdraw",
        source_span={"token_start": 0, "token_end": 4},
    )
    state = stage_turn(
        state,
        ingest("I withdraw my mortgage."),
        interpretation(ingest("I withdraw my mortgage."), [withdrawn]),
    ).state
    assert state.facts["existing_loan_amount"].status == FactStatus.WITHDRAWN


def test_forget_purges_proof_and_tombstones_replay_turn():
    turn = ingest("My mortgage is SEK 1,560,000.")
    state = stage_turn(
        Memory(),
        turn,
        interpretation(turn, [money_operation(turn, "existing_loan_amount", 0, 0, 5)]),
    ).state
    forgotten, ids = forget(state, {turn.turn_ref})
    assert ids == (turn.turn_ref,)
    assert not forgotten.protected_values and not forgotten.facts
    with pytest.raises(MemoryError, match="FORGOTTEN_TURN"):
        stage_turn(forgotten, turn, interpretation(turn, []))


def test_context_is_bounded_safe_and_retains_early_live_fact_after_history_prune():
    turn = ingest("My mortgage is SEK 1,560,000.")
    state = stage_turn(
        Memory(),
        turn,
        interpretation(turn, [money_operation(turn, "existing_loan_amount", 0, 0, 5)]),
    ).state
    for _ in range(25):
        state = stage_turn(
            state, ingest("Thanks."), interpretation(ingest("Thanks."), [])
        ).state
    context = build_context(
        state, ContextLimits(input_tokens=200, output_reservation=10), (("EVIDENCE",),)
    )
    assert len(state.exchanges) == 24
    fact = state.facts["existing_loan_amount"]
    assert (
        f"FACT({fact.fact_id},existing_loan_amount,r{fact.field_revision},{fact.status})"
        in context["state"]
    )
    assert "1,560,000" not in str(context)
    assert context["evidence"] == ("EVIDENCE",)


def test_context_fails_when_mandatory_evidence_cannot_fit():
    with pytest.raises(MemoryError, match="CONTEXT_BUDGET_EXCEEDED"):
        build_context(
            Memory(),
            ContextLimits(input_tokens=10, output_reservation=9, mandatory_tokens=2),
            (("A",),),
        )


def test_replay_body_eviction_preserves_identity_fingerprint_and_order():
    state = Memory()
    for index in range(17):
        state = record_replay(state, f"m{index}", str(index).encode(), ("OK",))
    assert sum(record.response is not None for record in state.replay.values()) == 16
    with pytest.raises(MemoryError, match="REPLAY_EXPIRED"):
        replay_response(state, "m0", b"0")
    with pytest.raises(MemoryError, match="REPLAY_CONFLICT"):
        replay_response(state, "m0", b"changed")
    assert replay_response(state, "m16", b"16") == ("OK",)
    assert record_replay(state, "m0", b"0", ("NEW",)) == state


def test_server_requires_full_units_roles_and_complete_compound_property():
    turn = ingest(
        "My mortgage is SEK 1,560,000 but my property value is SEK 2,600,000."
    )
    narrow = FactOperation(
        field="existing_loan_amount",
        operation="set",
        value_refs=[turn.protected_values[0].value_ref],
        source_span={"token_start": 0, "token_end": 3},
    )
    narrow_proposal = interpretation(turn, [narrow]).model_copy(
        update={"scope_source_spans": []}
    )
    rejected = stage_turn(Memory(), turn, narrow_proposal)
    assert rejected.rejection_code == "INVALID_PROPOSAL"
    assert rejected.state.facts["existing_loan_amount"].status == FactStatus.AMBIGUOUS
    property_turn = ingest("My property is terraced house.")
    op = FactOperation(
        field="property_type",
        operation="set",
        enum_value="terraced_house",
        source_span={"token_start": 0, "token_end": 5},
    )
    assert (
        stage_turn(Memory(), property_turn, interpretation(property_turn, [op]))
        .state.facts["property_type"]
        .enum_value
        == "terraced_house"
    )


def test_scope_proofs_pin_material_changes_and_identical_proposals_do_not_revise():
    turn = ingest("I want to refinance my existing mortgage without extra borrowing.")
    proposal = interpretation(
        turn,
        [],
        goal="refinance",
        borrowing="same",
        ownership="own_existing",
    )
    first = stage_turn(
        Memory(offer={"id": "old"}, accepted_review={"id": "review"}), turn, proposal
    )
    # A scope change makes drafts stale; an LTV depends only on the amounts.
    assert first.state.scope_proofs and set(first.invalidated) == {"offer", "review"}
    repeated = stage_turn(
        first.state, ingest("Thanks."), interpretation(ingest("Thanks."), [])
    )
    assert repeated.state.revision == first.state.revision
    assert repeated.state.scope_proofs == first.state.scope_proofs


def test_accepted_turn_counter_client_tombstone_and_clear_all_cover_nonfacts():
    state = Memory()
    for index in range(100):
        turn = ingest("Thanks.")
        state = stage_turn(
            state, turn, interpretation(turn, []), client_message_id=f"m{index}"
        ).state
    with pytest.raises(MemoryError, match="TURN_CAPACITY_EXCEEDED"):
        turn = ingest("Thanks.")
        stage_turn(state, turn, interpretation(turn, []), client_message_id="overflow")
    cleared, ids = forget(state, clear_all=True)
    assert len(ids) == 100 and not cleared.exchanges and not cleared.replay
    with pytest.raises(MemoryError, match="FORGOTTEN_TURN"):
        stage_turn(
            cleared,
            ingest("Thanks."),
            interpretation(ingest("Thanks."), []),
            client_message_id="m0",
        )


def test_model_unresolved_cannot_create_pending_or_authorize_a_bare_amount():
    thanks = ingest("Thanks.")
    proposed = interpretation(thanks, []).model_copy(
        update={"unresolved": ["property_value"]}
    )
    state = stage_turn(Memory(), thanks, proposed).state
    assert state.pending_question is None
    amount = ingest("SEK 2,600,000.")
    op = FactOperation(
        field="property_value",
        operation="set",
        value_refs=[amount.protected_values[0].value_ref],
        source_span={"token_start": 0, "token_end": 1},
    )
    with pytest.raises(
        MemoryError, match="BARE_VALUE_REQUIRES_CURRENT_PENDING_QUESTION"
    ):
        stage_turn(Memory(), amount, interpretation(amount, [op]))
    basis_turn = ingest(
        "I want to refinance my existing mortgage without extra borrowing."
    )
    basis = stage_turn(
        Memory(),
        basis_turn,
        interpretation(
            basis_turn,
            [],
            goal="refinance",
            borrowing="same",
            ownership="own_existing",
        ),
    ).state
    pending = PendingQuestion(
        "property_value", basis.revision, "q1", basis.scope_proofs["scope"][0]
    )
    accepted = stage_turn(
        Memory(**{**basis.__dict__, "pending_question": pending}),
        amount,
        interpretation(amount, [op]),
    ).state
    assert accepted.facts["property_value"].status == FactStatus.KNOWN


@pytest.mark.parametrize(
    "text, field, expected",
    [
        (
            "Their mortgage is SEK 1,560,000.",
            "existing_loan_amount",
            "NONPERSONAL_ASSERTION",
        ),
        (
            "The mortgage is SEK 1,560,000.",
            "existing_loan_amount",
            "PERSONAL_MARKER_REQUIRED",
        ),
    ],
)
def test_labelled_facts_require_a_positive_personal_marker(text, field, expected):
    turn = ingest(text)
    op = money_operation(turn, field, 0, 0, len(turn.tokens) - 1)
    with pytest.raises(MemoryError, match=expected):
        stage_turn(Memory(), turn, interpretation(turn, [op]))


def test_scope_proofs_are_per_field_and_questions_cannot_establish_eligibility():
    question = ingest(
        "I want to refinance my existing mortgage without extra borrowing?"
    )
    with pytest.raises(MemoryError, match="QUESTION_CANNOT_PROVE_SCOPE"):
        stage_turn(
            Memory(),
            question,
            interpretation(
                question,
                [],
                goal="refinance",
                ownership="own_existing",
                borrowing="same",
            ),
        )
    supported = ingest(
        "We want to refinance our current mortgage without extra borrowing."
    )
    state = stage_turn(
        Memory(),
        supported,
        interpretation(
            supported,
            [],
            goal="refinance",
            ownership="own_existing",
            borrowing="same",
        ),
    ).state
    assert set(state.scope_proofs) == {"goal", "scope", "borrowing"}


def test_blocking_scope_and_withdrawal_are_source_backed_and_invalidate():
    increased = ingest("I want to increase my borrowing.")
    result = stage_turn(
        Memory(offer={"id": "old"}, accepted_review={"id": "review"}),
        increased,
        interpretation(increased, [], borrowing="increase"),
    )
    assert result.state.borrowing == "increase"
    assert result.state.scope_proofs["borrowing"]
    assert set(result.invalidated) == {"offer", "review"}
    withdrawn = ingest("I withdraw my mortgage.")
    op = FactOperation(
        field="existing_loan_amount",
        operation="withdraw",
        source_span={"token_start": 0, "token_end": 4},
    )
    fact = stage_turn(Memory(), withdrawn, interpretation(withdrawn, [op])).state.facts[
        "existing_loan_amount"
    ]
    assert fact.original_proof is not None


def test_explicit_actions_precede_pending_but_knowledge_preserves_it():
    turn = ingest("I want to refinance my existing mortgage without extra borrowing.")
    state = stage_turn(
        Memory(),
        turn,
        interpretation(
            turn, [], goal="refinance", ownership="own_existing", borrowing="same"
        ),
    ).state
    pending = PendingQuestion(
        "property_value", state.revision, "q", state.scope_proofs["scope"][0]
    )
    assert (
        choose_action(
            Memory(
                **{
                    **state.__dict__,
                    "pending_question": pending,
                    "last_operation": "knowledge",
                }
            )
        )
        == "retrieve"
    )
    assert (
        choose_action(
            Memory(
                **{
                    **state.__dict__,
                    "pending_question": pending,
                    "last_operation": "forget",
                }
            )
        )
        == "forget"
    )


def test_replay_identity_cap_and_utf8_body_limit_are_independent_of_turn_count():
    state = Memory()
    for index in range(100):
        state = record_replay(state, f"m{index}", str(index).encode(), ("ok",))
    with pytest.raises(MemoryError, match="IDENTITY_CAPACITY_EXCEEDED"):
        record_replay(state, "m100", b"100", ("ok",))
    multibyte = "å" * (1024 * 1024)
    state = record_replay(Memory(), "unicode", b"u", (multibyte,))
    with pytest.raises(MemoryError, match="REPLAY_EXPIRED"):
        replay_response(state, "unicode", b"u")


def test_selective_forget_returns_client_redactions_and_preserves_unaffected_replay():
    first = ingest("My mortgage is SEK 1,560,000.")
    state = stage_turn(
        Memory(),
        first,
        interpretation(
            first, [money_operation(first, "existing_loan_amount", 0, 0, 5)]
        ),
        client_message_id="first-client",
    ).state
    second = ingest("Thanks.")
    state = stage_turn(
        state, second, interpretation(second, []), client_message_id="second-client"
    ).state
    state = record_replay(state, "first-client", b"first", ("first",))
    state = record_replay(state, "second-client", b"second", ("second",))
    forgotten, redacted = forget(state, {first.turn_ref})
    assert redacted == ("first-client",)
    assert "first-client" not in forgotten.replay
    with pytest.raises(MemoryError, match="FORGOTTEN_TURN"):
        replay_response(forgotten, "first-client", b"first")
    with pytest.raises(MemoryError, match="REPLAY_EXPIRED"):
        replay_response(forgotten, "second-client", b"second")
    with pytest.raises(MemoryError, match="REPLAY_CONFLICT"):
        replay_response(forgotten, "second-client", b"changed")


def test_forget_resets_only_scope_enums_whose_source_was_affected():
    goal_turn = ingest("I want to refinance my existing mortgage.")
    state = stage_turn(
        Memory(), goal_turn, interpretation(goal_turn, [], goal="refinance")
    ).state
    ownership_turn = ingest("I have current mortgage.")
    state = stage_turn(
        state,
        ownership_turn,
        interpretation(ownership_turn, [], ownership="own_existing"),
    ).state
    borrowing_turn = ingest("I don't want to increase my borrowing.")
    state = stage_turn(
        state, borrowing_turn, interpretation(borrowing_turn, [], borrowing="same")
    ).state
    forgotten, _ = forget(state, {ownership_turn.turn_ref})
    assert forgotten.goal == "refinance"
    assert forgotten.scope == "unknown"
    assert forgotten.borrowing == "same"
    shared_turn = ingest(
        "I want to refinance my current mortgage without extra borrowing."
    )
    shared = stage_turn(
        Memory(),
        shared_turn,
        interpretation(
            shared_turn,
            [],
            goal="refinance",
            ownership="own_existing",
            borrowing="same",
        ),
    ).state
    cleared, _ = forget(shared, {shared_turn.turn_ref})
    assert (cleared.goal, cleared.scope, cleared.borrowing) == (
        "unknown",
        "unknown",
        "unknown",
    )


def test_scope_parser_rejects_information_cross_unit_and_omitted_conflict():
    information = ingest("I have information about refinance.")
    rejected = stage_turn(
        Memory(), information, interpretation(information, [], goal="refinance")
    )
    assert rejected.proposal_rejected and rejected.state.goal == "unknown"
    conflict = ingest(
        "I want to refinance my existing mortgage without extra borrowing. I want to increase my borrowing."
    )
    first_end = next(
        i for i, token in enumerate(conflict.tokens) if token.symbol == "SENTENCE_END"
    )
    proposal = interpretation(
        conflict, [], goal="refinance", borrowing="same", ownership="own_existing"
    ).model_copy(
        update={"scope_source_spans": [SourceSpan(token_start=0, token_end=first_end)]}
    )
    assert stage_turn(Memory(), conflict, proposal).state.borrowing == "increase"


def test_third_party_blocker_and_property_type_pending_or_ambiguity():
    third_party = ingest("I want to refinance their mortgage.")
    blocked = stage_turn(
        Memory(offer={"id": "old"}),
        third_party,
        interpretation(third_party, [], ownership="other"),
    )
    assert blocked.state.scope == "other" and blocked.state.offer is None
    basis_turn = ingest("I want to refinance my existing mortgage.")
    basis = stage_turn(
        Memory(),
        basis_turn,
        interpretation(basis_turn, [], goal="refinance", ownership="own_existing"),
    ).state
    pending = PendingQuestion(
        "property_type", basis.revision, "type", basis.scope_proofs["scope"][0]
    )
    bare = ingest("Terraced house.")
    set_type = FactOperation(
        field="property_type",
        operation="set",
        enum_value="terraced_house",
        source_span={"token_start": 0, "token_end": 2},
    )
    assert (
        stage_turn(
            Memory(**{**basis.__dict__, "pending_question": pending}),
            bare,
            interpretation(bare, [set_type]),
        )
        .state.facts["property_type"]
        .status
        == FactStatus.KNOWN
    )
    alternatives = ingest("My property is terraced house or apartment.")
    ambiguous = FactOperation(
        field="property_type",
        operation="ambiguous",
        enum_values=["terraced_house", "apartment"],
        source_span={"token_start": 0, "token_end": 7},
    )
    assert (
        stage_turn(Memory(), alternatives, interpretation(alternatives, [ambiguous]))
        .state.facts["property_type"]
        .status
        == FactStatus.AMBIGUOUS
    )


@pytest.mark.parametrize(
    ("text", "changes"),
    [
        (
            "I want to refinance my mortgage.",
            {"goal": "refinance", "ownership": "own_existing"},
        ),
        (
            "I need to refinance my mortgage.",
            {"goal": "refinance", "ownership": "own_existing"},
        ),
        (
            "I would like to refinance my mortgage.",
            {"goal": "refinance", "ownership": "own_existing"},
        ),
        (
            "I am refinance my mortgage.",
            {"goal": "refinance", "ownership": "own_existing"},
        ),
        (
            "I want a better rate on my mortgage.",
            {"goal": "refinance", "ownership": "own_existing"},
        ),
        (
            "I am looking for a better rate for my mortgage.",
            {"goal": "refinance", "ownership": "own_existing"},
        ),
        ("I want to keep my mortgage amount the same.", {"borrowing": "same"}),
        ("I want to keep the same mortgage amount.", {"borrowing": "same"}),
        ("I don't want to increase my borrowing.", {"borrowing": "same"}),
        ("I will not increase my borrowing.", {"borrowing": "same"}),
        ("I don't want to borrowing more.", {"borrowing": "same"}),
        ("I want to increase my borrowing.", {"borrowing": "increase"}),
        ("I want to increase my mortgage amount.", {"borrowing": "increase"}),
        ("I want to borrowing more.", {"borrowing": "increase"}),
        ("I want to refinance their mortgage.", {"ownership": "other"}),
        ("I don't want to refinance my mortgage.", {"goal": "none"}),
        ("I want not to refinance my mortgage.", {"goal": "none"}),
        ("I need information about mortgage.", {"goal": "information"}),
        ("I have existing mortgage.", {"ownership": "own_existing"}),
        (
            "Actually, please I want to refinance my mortgage without increase my borrowing.",
            {"goal": "refinance", "ownership": "own_existing", "borrowing": "same"},
        ),
    ],
)
def test_every_documented_scope_production_is_a_complete_stage_proof(text, changes):
    turn = ingest(text)
    result = stage_turn(Memory(), turn, interpretation(turn, [], **changes)).state
    assert all(
        getattr(
            result,
            {"goal": "goal", "ownership": "scope", "borrowing": "borrowing"}[name],
        )
        == value
        for name, value in changes.items()
    )


@pytest.mark.parametrize(
    "text",
    [
        "I want to refinance my mortgage?",
        "If I want to refinance my mortgage.",
        'I want to refinance "my mortgage".',
    ],
)
def test_scope_grammar_controls_cannot_prove_positive_refinance(text):
    turn = ingest(text)
    with pytest.raises(MemoryError):
        stage_turn(
            Memory(),
            turn,
            interpretation(turn, [], goal="refinance", ownership="own_existing"),
        )


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        (
            "I want to refinance my mortgage.",
            {"goal": "refinance", "scope": "own_existing"},
        ),
        (
            "Actually, please I want to refinance my mortgage.",
            {"goal": "refinance", "scope": "own_existing"},
        ),
        (
            "I want to refinance my mortgage without extra borrowing.",
            {"goal": "refinance", "scope": "own_existing", "borrowing": "same"},
        ),
        (
            "Actually, please I want to refinance my mortgage without extra borrowing.",
            {"goal": "refinance", "scope": "own_existing", "borrowing": "same"},
        ),
        ("I have current mortgage without extra borrowing.", {"scope": "own_existing"}),
    ],
)
def test_scope_prefix_suffix_parser_has_exact_predicates(text, expected):
    turn = ingest(text)
    end = next(
        i
        for i, token in enumerate(turn.tokens)
        if token.symbol in {"SENTENCE_END", "QUESTION"}
    )
    assert parse_scope_unit(turn, SourceSpan(token_start=0, token_end=end)) == expected


def test_fact_questions_uncertainty_and_type_proof_equality_are_gated():
    question = ingest("My mortgage is SEK 1,560,000?")
    op = money_operation(question, "existing_loan_amount", 0, 0, 5)
    with pytest.raises(MemoryError, match="QUESTION_CANNOT_SET_FACT"):
        stage_turn(Memory(), question, interpretation(question, [op]))
    uncertain = ingest("My mortgage is maybe SEK 1,560,000.")
    op = money_operation(uncertain, "existing_loan_amount", 0, 0, 6)
    rejected = stage_turn(Memory(), uncertain, interpretation(uncertain, [op]))
    assert rejected.proposal_rejected
    assert rejected.state.facts["existing_loan_amount"].status == FactStatus.AMBIGUOUS
    first = ingest("My property is terraced house or apartment.")
    alternatives = FactOperation(
        field="property_type",
        operation="ambiguous",
        enum_values=["apartment", "terraced_house"],
        source_span={"token_start": 0, "token_end": 7},
    )
    state = stage_turn(Memory(), first, interpretation(first, [alternatives])).state
    assert state.facts["property_type"].enum_values == ("terraced_house", "apartment")
    changed = ingest("My property is house or apartment.")
    alternatives = FactOperation(
        field="property_type",
        operation="ambiguous",
        enum_values=["house", "apartment"],
        source_span={"token_start": 0, "token_end": 6},
    )
    result = stage_turn(state, changed, interpretation(changed, [alternatives]))
    assert result.state.facts["property_type"].field_revision == 2


@pytest.mark.parametrize(
    ("text", "attribute", "expected"),
    [
        ("I want to increase my borrowing.", "borrowing", "increase"),
        ("I don't want to refinance my mortgage.", "goal", "none"),
        ("I want to refinance their mortgage.", "scope", "other"),
    ],
)
def test_source_restrictions_revoke_ready_offer_without_model_claim(
    text, attribute, expected
):
    result = stage_turn(
        Memory(offer={"id": "ready"}, accepted_review={"id": "review"}),
        ingest(text),
        interpretation(ingest(text), []),
    )
    assert getattr(result.state, attribute) == expected
    assert result.state.offer is None and result.state.accepted_review is None


def test_restriction_survives_invalid_proposal_without_positive_fact_commit():
    turn = ingest("If my mortgage is SEK 1,560,000; I want to increase my borrowing.")
    op = money_operation(turn, "existing_loan_amount", 0, 0, 5)
    result = stage_turn(Memory(), turn, interpretation(turn, [op]))
    assert result.proposal_rejected and result.rejection_code == "INVALID_PROPOSAL"
    assert not result.state.facts and result.state.borrowing == "increase"


@pytest.mark.parametrize(
    "text", ["My mortgage is not SEK 1,560,000.", "My mortgage is maybe SEK 1,560,000."]
)
def test_source_fact_uncertainty_overrides_model_unchanged_and_preserves_property(text):
    first = ingest(
        "My mortgage is SEK 1,560,000 and my property value is SEK 2,600,000."
    )
    state = stage_turn(
        Memory(),
        first,
        interpretation(
            first,
            [
                money_operation(first, "existing_loan_amount", 0, 0, 5),
                money_operation(first, "property_value", 1, 6, 11),
            ],
        ),
    ).state
    turn = ingest(text)
    result = stage_turn(state, turn, interpretation(turn, []))
    assert result.state.facts["existing_loan_amount"].status == FactStatus.AMBIGUOUS
    assert result.state.facts["property_value"].status == FactStatus.KNOWN
    assert not result.state.protected_values.get(first.protected_values[0].value_ref)
    assert (
        choose_action(
            Memory(**{**result.state.__dict__, "last_operation": "calculate"})
        )
        == "clarify"
    )


def test_negation_restricts_only_its_independently_labelled_clause():
    initial = ingest(
        "My mortgage is SEK 1,560,000 and my property value is SEK 2,600,000."
    )
    state = stage_turn(
        Memory(offer={"id": "ready"}),
        initial,
        interpretation(
            initial,
            [
                money_operation(initial, "existing_loan_amount", 0, 0, 5),
                money_operation(initial, "property_value", 1, 6, 11),
            ],
        ),
    ).state
    turn = ingest(
        "My mortgage is not SEK 1,560,000 and my property value is SEK 2,600,000."
    )

    result = stage_turn(state, turn, interpretation(turn, []))

    assert {name: fact.status for name, fact in result.state.facts.items()} == {
        "existing_loan_amount": FactStatus.AMBIGUOUS,
        "property_value": FactStatus.KNOWN,
    }
    assert result.state.offer is None
    assert choose_action(result.state) != "calculate"


def test_uncertainty_in_a_mixed_unit_conservatively_restricts_all_implicated_fields():
    initial = ingest(
        "My mortgage is SEK 1,560,000 and my property value is SEK 2,600,000."
    )
    state = stage_turn(
        Memory(),
        initial,
        interpretation(
            initial,
            [
                money_operation(initial, "existing_loan_amount", 0, 0, 5),
                money_operation(initial, "property_value", 1, 6, 11),
            ],
        ),
    ).state
    turn = ingest(
        "My mortgage is not SEK 1,560,000 and my property value is maybe SEK 2,600,000."
    )

    result = stage_turn(state, turn, interpretation(turn, []))

    assert all(
        fact.status == FactStatus.AMBIGUOUS for fact in result.state.facts.values()
    )


@pytest.mark.parametrize("text", ["House a.", "A house a."])
def test_pending_property_type_accepts_only_a_complete_bare_type_reply(text):
    basis_turn = ingest("I want to refinance my existing mortgage.")
    basis = stage_turn(
        Memory(),
        basis_turn,
        interpretation(basis_turn, [], goal="refinance", ownership="own_existing"),
    ).state
    pending = PendingQuestion(
        "property_type", basis.revision, "type", basis.scope_proofs["scope"][0]
    )
    turn = ingest(text)
    operation = FactOperation(
        field="property_type",
        operation="set",
        enum_value="house",
        source_span={"token_start": 0, "token_end": len(turn.tokens) - 1},
    )
    with pytest.raises(MemoryError, match="PERSONAL_MARKER_REQUIRED"):
        stage_turn(
            replace(basis, pending_question=pending),
            turn,
            interpretation(turn, [operation]),
        )

    valid = ingest("House.")
    accepted = stage_turn(
        replace(basis, pending_question=pending),
        valid,
        interpretation(
            valid,
            [
                FactOperation(
                    field="property_type",
                    operation="set",
                    enum_value="house",
                    source_span={"token_start": 0, "token_end": 1},
                )
            ],
        ),
    )
    assert accepted.state.facts["property_type"].enum_value == "house"


def test_pending_property_type_rejects_a_dangling_alternative_separator():
    basis_turn = ingest("I want to refinance my existing mortgage.")
    basis = stage_turn(
        Memory(),
        basis_turn,
        interpretation(basis_turn, [], goal="refinance", ownership="own_existing"),
    ).state
    pending = PendingQuestion(
        "property_type", basis.revision, "type", basis.scope_proofs["scope"][0]
    )
    valid = ingest("House or apartment.")
    alternatives = FactOperation(
        field="property_type",
        operation="ambiguous",
        enum_values=["house", "apartment"],
        source_span={"token_start": 0, "token_end": len(valid.tokens) - 1},
    )
    accepted = stage_turn(
        replace(basis, pending_question=pending),
        valid,
        interpretation(valid, [alternatives]),
    )
    assert accepted.state.facts["property_type"].status == FactStatus.AMBIGUOUS

    dangling = ingest("House or apartment or.")
    with pytest.raises(MemoryError, match="PERSONAL_MARKER_REQUIRED"):
        stage_turn(
            replace(basis, pending_question=pending),
            dangling,
            interpretation(
                dangling,
                [
                    FactOperation(
                        field="property_type",
                        operation="ambiguous",
                        enum_values=["house", "apartment"],
                        source_span={
                            "token_start": 0,
                            "token_end": len(dangling.tokens) - 1,
                        },
                    )
                ],
            ),
        )


@pytest.mark.parametrize(
    ("text", "blocks"),
    [
        ("What is refinance.", False),
        ("They refinance their mortgage.", False),
        ("I want to refinance their existing mortgage.", True),
        ("I want to refinance my friend mortgage.", True),
    ],
)
def test_only_personal_third_party_refinance_claim_blocks_an_existing_offer(
    text, blocks
):
    basis_turn = ingest("I want to refinance my existing mortgage.")
    state = stage_turn(
        Memory(),
        basis_turn,
        interpretation(basis_turn, [], goal="refinance", ownership="own_existing"),
    ).state
    turn = ingest(text)

    result = stage_turn(
        replace(state, offer={"id": "ready"}),
        turn,
        interpretation(turn, [], subject="third_party" if "They" in text else "self"),
    )

    assert result.state.scope == ("other" if blocks else "own_existing")
    assert (result.state.offer is None) is blocks


@pytest.mark.parametrize(
    ("text", "subject", "conditional"),
    [
        ("If I want to refinance my friend mortgage.", "self", True),
        ('I want to refinance "my friend mortgage".', "quoted", False),
    ],
)
def test_hypothetical_and_quoted_third_party_text_do_not_revoke_an_offer(
    text, subject, conditional
):
    basis_turn = ingest("I want to refinance my existing mortgage.")
    state = stage_turn(
        Memory(),
        basis_turn,
        interpretation(basis_turn, [], goal="refinance", ownership="own_existing"),
    ).state
    turn = ingest(text)

    result = stage_turn(
        replace(state, offer={"id": "ready"}),
        turn,
        interpretation(turn, [], subject=subject, conditional=conditional),
    )

    assert result.state.scope == "own_existing"
    assert result.state.offer == {"id": "ready"}


def test_restriction_changes_only_report_final_ambiguous_facts():
    turn = ingest("My mortgage is maybe SEK 1,560,000.")
    result = stage_turn(
        Memory(),
        turn,
        interpretation(
            turn,
            [money_operation(turn, "existing_loan_amount", 0, 0, 6)],
        ),
    )

    assert result.proposal_rejected
    assert [(fact.field, fact.status) for fact in result.changes] == [
        ("existing_loan_amount", FactStatus.AMBIGUOUS)
    ]


def test_direct_restriction_commit_enforces_guards_and_bounds_atomically():
    turn = ingest("I want to increase my borrowing.", turn_ref="current")
    proof = OriginalProof(turn.turn_ref, 0, len(turn.original), turn.original)
    restrictions = {"borrowing": ScopeObservation("increase", (proof,))}
    committed = commit_restrictions(
        Memory(offer={"id": "ready"}),
        turn,
        restrictions,
        client_message_id="current-client",
        rejected=True,
        rejection_code="INVALID_PROPOSAL",
    )
    assert (
        committed.proposal_rejected and committed.rejection_code == "INVALID_PROPOSAL"
    )
    assert committed.state.borrowing == "increase" and committed.state.offer is None

    forgotten = Memory(forgotten_turns=frozenset({turn.turn_ref}))
    with pytest.raises(MemoryError, match="FORGOTTEN_TURN"):
        commit_restrictions(forgotten, turn, restrictions, client_message_id="new")
    with pytest.raises(MemoryError, match="REPLAY_EXPIRED"):
        commit_restrictions(
            committed.state, turn, restrictions, client_message_id="current-client"
        )
    with pytest.raises(MemoryError, match="TURN_CAPACITY_EXCEEDED"):
        commit_restrictions(
            Memory(accepted_turns=100), turn, restrictions, client_message_id="new"
        )

    identities = Memory(
        client_turns={f"live-{index}": f"turn-{index}" for index in range(34)},
        replay={
            f"replay-{index}": ReplayRecord(str(index), "ACCEPTED", None)
            for index in range(33)
        },
        forgotten_client_ids=frozenset(f"forgotten-{index}" for index in range(33)),
    )
    with pytest.raises(MemoryError, match="IDENTITY_CAPACITY_EXCEEDED"):
        commit_restrictions(identities, turn, restrictions, client_message_id="new")

    protected_turns = tuple(
        ingest("I want to increase my borrowing.", turn_ref=f"protected-{index}")
        for index in range(32)
    )
    protected_proofs = tuple(
        OriginalProof(item.turn_ref, 0, len(item.original), item.original)
        for item in protected_turns
    )
    protected_state = Memory(
        turns={item.turn_ref: item for item in protected_turns},
        scope_proofs={"goal": protected_proofs},
    )
    with pytest.raises(MemoryError, match="PROTECTED_ORIGINAL_CAPACITY_EXCEEDED"):
        commit_restrictions(
            protected_state, turn, restrictions, client_message_id="new"
        )
    assert protected_state.turns == {item.turn_ref: item for item in protected_turns}

    oversized = replace(turn, turn_ref="oversized", original="å" * (128 * 1024))
    oversized_proof = OriginalProof(
        oversized.turn_ref, 0, len(oversized.original), oversized.original
    )
    oversized_state = Memory(
        turns={oversized.turn_ref: oversized},
        scope_proofs={"goal": (oversized_proof,)},
    )
    with pytest.raises(MemoryError, match="PROTECTED_ORIGINAL_CAPACITY_EXCEEDED"):
        commit_restrictions(
            oversized_state, turn, restrictions, client_message_id="new"
        )

    exchanges = tuple(SafeExchange(f"old-{index}", ("OK",)) for index in range(24))
    retained = commit_restrictions(
        Memory(exchanges=exchanges), turn, restrictions, client_message_id="new"
    ).state.exchanges
    assert len(retained) == 24
    assert retained[0].turn_ref == "old-1" and retained[-1].turn_ref == turn.turn_ref

    utf8_retained = commit_restrictions(
        Memory(exchanges=(SafeExchange("large", ("å" * (128 * 1024),)),)),
        turn,
        restrictions,
        client_message_id="new",
    ).state.exchanges
    assert utf8_retained == (SafeExchange(turn.turn_ref, turn.model_projection),)

    oversized_history_turn = replace(
        turn,
        tokens=(CanonicalToken("å" * (128 * 1024), 0, 0, 1),),
    )
    history_state = Memory()
    with pytest.raises(MemoryError, match="SAFE_HISTORY_CAPACITY_EXCEEDED"):
        commit_restrictions(
            history_state,
            oversized_history_turn,
            restrictions,
            client_message_id="new",
        )
    assert history_state.exchanges == ()


def test_omitted_positive_correction_withholds_the_stale_loan_for_calculation():
    initial = ingest(
        "My mortgage is SEK 1,560,000 and my property value is SEK 2,600,000."
    )
    state = stage_turn(
        Memory(),
        initial,
        interpretation(
            initial,
            [
                money_operation(initial, "existing_loan_amount", 0, 0, 5),
                money_operation(initial, "property_value", 1, 6, 11),
            ],
        ),
    ).state
    correction = ingest("My mortgage is SEK 1,820,000.")

    rejected = stage_turn(
        state, correction, interpretation(correction, [], action="calculate")
    )

    assert rejected.proposal_rejected
    assert rejected.rejection_code == "INCOMPLETE_FACT_PROPOSAL"
    assert rejected.state.facts["existing_loan_amount"].status == FactStatus.AMBIGUOUS
    assert (
        choose_action(replace(rejected.state, last_operation="calculate")) == "clarify"
    )


def test_complete_positive_correction_replaces_the_loan_and_calculates():
    initial = ingest(
        "My mortgage is SEK 1,560,000 and my property value is SEK 2,600,000."
    )
    state = stage_turn(
        Memory(),
        initial,
        interpretation(
            initial,
            [
                money_operation(initial, "existing_loan_amount", 0, 0, 5),
                money_operation(initial, "property_value", 1, 6, 11),
            ],
        ),
    ).state
    correction = ingest("My mortgage is SEK 1,820,000.")

    accepted = stage_turn(
        state,
        correction,
        interpretation(
            correction,
            [money_operation(correction, "existing_loan_amount", 0, 0, 5)],
            action="calculate",
        ),
    )

    assert not accepted.proposal_rejected
    assert accepted.state.facts["existing_loan_amount"].status == FactStatus.KNOWN
    assert choose_action(accepted.state) == "calculate"


def test_incomplete_or_invalid_positive_batches_withhold_every_asserted_field():
    initial = ingest(
        "My mortgage is SEK 1,560,000 and my property value is SEK 2,600,000."
    )
    state = stage_turn(
        Memory(),
        initial,
        interpretation(
            initial,
            [
                money_operation(initial, "existing_loan_amount", 0, 0, 5),
                money_operation(initial, "property_value", 1, 6, 11),
            ],
        ),
    ).state
    correction = ingest(
        "My mortgage is SEK 1,820,000 and my property value is SEK 2,900,000."
    )
    loan = money_operation(correction, "existing_loan_amount", 0, 0, 5)

    omitted = stage_turn(state, correction, interpretation(correction, [loan]))
    assert omitted.rejection_code == "INCOMPLETE_FACT_PROPOSAL"
    assert all(
        fact.status == FactStatus.AMBIGUOUS for fact in omitted.state.facts.values()
    )

    separate = ingest(
        "My mortgage is SEK 1,820,000. My property value is SEK 2,900,000."
    )
    separate_loan = money_operation(separate, "existing_loan_amount", 0, 0, 0)
    across_units = stage_turn(
        state, separate, interpretation(separate, [separate_loan])
    )
    assert across_units.rejection_code == "INCOMPLETE_FACT_PROPOSAL"
    assert all(
        fact.status == FactStatus.AMBIGUOUS
        for fact in across_units.state.facts.values()
    )

    invalid_property = FactOperation(
        field="property_value",
        operation="set",
        value_refs=[correction.protected_values[0].value_ref],
        source_span={"token_start": 0, "token_end": 11},
    )
    invalid = stage_turn(
        state, correction, interpretation(correction, [loan, invalid_property])
    )
    assert invalid.rejection_code == "INVALID_PROPOSAL"
    assert all(
        fact.status == FactStatus.AMBIGUOUS for fact in invalid.state.facts.values()
    )


@pytest.mark.parametrize(
    ("text", "subject", "conditional"),
    [
        ("My mortgage is SEK 1,820,000?", "self", False),
        ("If my mortgage is SEK 1,820,000.", "self", True),
        ('"My mortgage is SEK 1,820,000."', "quoted", False),
        ("Their mortgage is SEK 1,820,000.", "third_party", False),
        ("I need information about mortgage.", "self", False),
    ],
)
def test_non_declarative_or_nonpersonal_mentions_preserve_current_authority(
    text, subject, conditional
):
    initial = ingest("My mortgage is SEK 1,560,000.")
    state = stage_turn(
        Memory(),
        initial,
        interpretation(
            initial, [money_operation(initial, "existing_loan_amount", 0, 0, 5)]
        ),
    ).state
    turn = ingest(text)

    result = stage_turn(
        state,
        turn,
        interpretation(turn, [], subject=subject, conditional=conditional),
    )

    assert result.state.facts["existing_loan_amount"].status == FactStatus.KNOWN


def test_forgetting_an_omitted_correction_does_not_restore_the_old_value():
    initial = ingest("My mortgage is SEK 1,560,000.")
    state = stage_turn(
        Memory(),
        initial,
        interpretation(
            initial, [money_operation(initial, "existing_loan_amount", 0, 0, 5)]
        ),
    ).state
    correction = ingest("My mortgage is SEK 1,820,000.")
    withheld = stage_turn(state, correction, interpretation(correction, [])).state

    forgotten, _ = forget(withheld, {correction.turn_ref})

    assert "existing_loan_amount" not in forgotten.facts


def test_mixed_clause_analysis_keeps_local_polarity_and_commits_property():
    initial = ingest(
        "My mortgage is SEK 1,560,000 and my property value is SEK 2,600,000."
    )
    state = stage_turn(
        Memory(),
        initial,
        interpretation(
            initial,
            [
                money_operation(initial, "existing_loan_amount", 0, 0, 5),
                money_operation(initial, "property_value", 1, 6, 11),
            ],
        ),
    ).state
    turn = ingest(
        "My mortgage is not SEK 1,560,000 and my property value is SEK 2,900,000."
    )
    result = stage_turn(
        state,
        turn,
        interpretation(turn, [money_operation(turn, "property_value", 1, 0, 0)]),
    )
    assert result.state.facts["existing_loan_amount"].status == FactStatus.AMBIGUOUS
    assert result.state.facts["property_value"].status == FactStatus.KNOWN


def test_shared_analysis_conflicts_and_public_uninterpreted_turn_fail_closed():
    first = ingest("My mortgage is SEK 1,560,000.")
    state = stage_turn(
        Memory(),
        first,
        interpretation(
            first, [money_operation(first, "existing_loan_amount", 0, 0, 5)]
        ),
    ).state
    conflict = ingest("My mortgage is SEK 1,560,000 and SEK 1,820,000.")
    conflicted = stage_uninterpreted_turn(
        state,
        conflict,
        client_message_id="conflict",
        rejection_code="INTERPRETATION_FAILED",
    )
    assert conflicted.state.facts["existing_loan_amount"].status == FactStatus.AMBIGUOUS
    forgotten, _ = forget(conflicted.state, {conflict.turn_ref})
    assert "existing_loan_amount" not in forgotten.facts

    correction = ingest("My mortgage is SEK 1,820,000.")
    withheld = stage_uninterpreted_turn(
        state,
        correction,
        client_message_id="correction",
        rejection_code="INTERPRETATION_FAILED",
    )
    assert withheld.state.facts["existing_loan_amount"].status == FactStatus.AMBIGUOUS
    reaffirmed = stage_uninterpreted_turn(
        state,
        first,
        client_message_id="reaffirmed",
        rejection_code="INTERPRETATION_FAILED",
    )
    assert reaffirmed.state.facts["existing_loan_amount"].status == FactStatus.KNOWN


def test_owned_ltv_parameter_production_is_covered_as_financial_input():
    turn = ingest(
        "Calculate LTV for my mortgage of SEK 1,560,000 and my property value of SEK 2,600,000."
    )
    result = stage_turn(
        Memory(),
        turn,
        interpretation(
            turn,
            [
                money_operation(turn, "existing_loan_amount", 0, 0, 0),
                money_operation(turn, "property_value", 1, 0, 0),
            ],
            action="calculate",
        ),
    )
    assert choose_action(result.state) == "calculate"


def test_shared_financial_grammar_handles_conflicts_repeats_and_information_mentions():
    first = ingest("My mortgage is SEK 1,560,000.")
    state = stage_turn(
        Memory(),
        first,
        interpretation(
            first, [money_operation(first, "existing_loan_amount", 0, 0, 5)]
        ),
    ).state
    shorthand = ingest("My mortgage is SEK 1,560,000 and SEK 1,820,000.")
    conflict = stage_turn(state, shorthand, interpretation(shorthand, []))
    assert conflict.state.facts["existing_loan_amount"].status == FactStatus.AMBIGUOUS

    repeated = ingest("My mortgage is SEK 1,560,000 and my mortgage is SEK 1,560,000.")
    repeated_result = stage_turn(
        state,
        repeated,
        interpretation(
            repeated,
            [money_operation(repeated, "existing_loan_amount", 0, 0, 0)],
        ),
    )
    assert (
        repeated_result.state.facts["existing_loan_amount"].status == FactStatus.KNOWN
    )

    information = ingest("I need information about mortgage SEK 1,820,000.")
    unchanged = stage_turn(state, information, interpretation(information, []))
    assert unchanged.state.facts["existing_loan_amount"].status == FactStatus.KNOWN


@pytest.mark.parametrize(
    "text",
    [
        "My mortgage is SEK 1,560,000 or.",
        "My mortgage is SEK 1,560,000 and maybe SEK 1,820,000.",
    ],
)
def test_incomplete_financial_tails_cannot_commit_or_preserve_a_known_amount(text):
    initial = ingest("My mortgage is SEK 1,560,000.")
    state = stage_turn(
        Memory(),
        initial,
        interpretation(
            initial, [money_operation(initial, "existing_loan_amount", 0, 0, 5)]
        ),
    ).state
    turn = ingest(text)
    operation = money_operation(turn, "existing_loan_amount", 0, 0, 0)
    result = stage_turn(state, turn, interpretation(turn, [operation]))
    assert result.proposal_rejected
    assert result.state.facts["existing_loan_amount"].status == FactStatus.AMBIGUOUS


def test_property_is_worth_is_a_property_value_production():
    turn = ingest("My property is worth SEK 2,600,000.")
    result = stage_turn(
        Memory(),
        turn,
        interpretation(turn, [money_operation(turn, "property_value", 0, 0, 0)]),
    )
    assert result.state.facts["property_value"].status == FactStatus.KNOWN


def test_uninterpreted_unrelated_question_only_bookkeeps_the_turn():
    basis_turn = ingest("I want to refinance my existing mortgage.")
    basis = stage_turn(
        Memory(),
        basis_turn,
        interpretation(basis_turn, [], goal="refinance", ownership="own_existing"),
    ).state
    pending = PendingQuestion(
        "property_value", basis.revision, "pending", basis.scope_proofs["scope"][0]
    )
    state = replace(
        basis,
        offer={"id": "ready"},
        accepted_review={"id": "review"},
        pending_question=pending,
    )
    question = ingest("What is LTV?")
    result = stage_uninterpreted_turn(
        state,
        question,
        client_message_id="question",
        rejection_code="INTERPRETATION_FAILED",
    )
    assert result.state.offer == {"id": "ready"}
    assert result.state.accepted_review == {"id": "review"}
    assert result.state.pending_question == pending
    assert result.state.accepted_turns == state.accepted_turns + 1


@pytest.mark.parametrize(
    ("text", "field", "kind", "candidate_count"),
    [
        ("My mortgage is SEK 1,560,000.", "existing_loan_amount", "affirmed", 1),
        ("My loan: SEK 1,560,000.", "existing_loan_amount", "affirmed", 1),
        ("My property value is SEK 2,600,000.", "property_value", "affirmed", 1),
        ("My property value: SEK 2,600,000.", "property_value", "affirmed", 1),
        ("My property is worth SEK 2,600,000.", "property_value", "affirmed", 1),
        ("My property is house.", "property_type", "affirmed", 1),
        ("My mortgage is not SEK 1,560,000.", "existing_loan_amount", "negated", 1),
        ("My mortgage is maybe SEK 1,560,000.", "existing_loan_amount", "uncertain", 1),
        (
            "My mortgage is SEK 1,560,000 or SEK 1,820,000.",
            "existing_loan_amount",
            "alternatives",
            2,
        ),
        (
            "My mortgage is SEK 1,560,000 and SEK 1,820,000.",
            "existing_loan_amount",
            "affirmed",
            1,
        ),
        (
            "Calculate LTV for my mortgage of SEK 1,560,000 and my property value of SEK 2,600,000.",
            "existing_loan_amount",
            "affirmed",
            1,
        ),
    ],
)
def test_financial_grammar_productions_have_full_unit_clauses(
    text, field, kind, candidate_count
):
    turn = ingest(text)
    analysis = analyze_financial_turn(Memory(), turn)
    clause = next(item for item in analysis.clauses if item.field == field)
    assert clause.kind == kind
    assert len(clause.value_refs or clause.enum_values) == candidate_count
    assert clause.unit_span.token_start == 0
    assert clause.unit_span.token_end == len(turn.tokens) - 1


@pytest.mark.parametrize(
    ("text", "kind"),
    [
        ("My property is worth SEK 2,600,000.", "affirmed"),
        ("My property is not worth SEK 2,600,000.", "negated"),
        ("My property is maybe worth SEK 2,600,000.", "uncertain"),
        ("My property is maybe not worth SEK 2,600,000.", "uncertain"),
        ("My property is not maybe worth SEK 2,600,000.", "uncertain"),
    ],
)
def test_worth_polarity_productions_are_cursor_consumed(text, kind):
    turn = ingest(text)
    analysis = analyze_financial_turn(Memory(), turn)
    assert analysis.clauses[0].field == "property_value"
    assert analysis.clauses[0].kind == kind


@pytest.mark.parametrize(
    "text",
    [
        "My mortgage is SEK 1,560,000 and.",
        "My mortgage is SEK 1,560,000 and not SEK 1,820,000.",
        "My mortgage is SEK 1,560,000 and maybe SEK 1,820,000.",
    ],
)
def test_malformed_financial_continuations_withhold_in_both_public_paths(text):
    first = ingest("My mortgage is SEK 1,560,000.")
    state = stage_turn(
        Memory(),
        first,
        interpretation(
            first, [money_operation(first, "existing_loan_amount", 0, 0, 5)]
        ),
    ).state
    turn = ingest(text)
    staged = stage_turn(state, turn, interpretation(turn, []))
    uninterpreted = stage_uninterpreted_turn(
        state,
        turn,
        client_message_id="malformed",
        rejection_code="INTERPRETATION_FAILED",
    )
    assert staged.state.facts["existing_loan_amount"].status == FactStatus.AMBIGUOUS
    assert (
        uninterpreted.state.facts["existing_loan_amount"].status == FactStatus.AMBIGUOUS
    )


@pytest.mark.parametrize(
    ("text", "concepts", "property_type"),
    [
        (
            "I want to refinance my existing mortgage without extra borrowing.",
            {"goal": "refinance", "scope": "own_existing", "borrowing": "same"},
            None,
        ),
        (
            "I want to refinance my existing mortgage without extra borrowing for my house.",
            {"goal": "refinance", "scope": "own_existing", "borrowing": "same"},
            "house",
        ),
        (
            "I want to refinance my existing mortgage without increasing our loan amount for my apartment.",
            {"goal": "refinance", "scope": "own_existing", "borrowing": "same"},
            "apartment",
        ),
        (
            "I want to refinance my existing mortgage without borrowing more for my house.",
            {"goal": "refinance", "scope": "own_existing", "borrowing": "same"},
            "house",
        ),
        ("I want to refinance my friend mortgage.", {"scope": "other"}, None),
        (
            "I want to refinance my friend mortgage without extra borrowing for my house.",
            {},
            None,
        ),
        (
            "I want to refinance my existing mortgage without extra borrowing for my friend house.",
            {},
            None,
        ),
    ],
)
def test_personal_refinance_accepts_every_no_increase_clause_and_rejects_third_parties(
    text, concepts, property_type
):
    turn = ingest(text)
    (span,) = assertion_unit_ranges(turn)
    assert parse_scope_unit(turn, span) == concepts
    assert scope_property_type(turn, span) == property_type


@pytest.mark.parametrize(
    "text",
    [
        "My existing mortgage is 2000000 and property is worth 2500000, what is my loan to value?",
        "What is my loan to value, my existing mortgage is 2000000 and property is worth 2500000",
        "My existing mortgage is 2000000 and property is worth 2500000. What is my loan to value?",
        "My existing mortgage is SEK 2,000,000 and my property is worth SEK 2,500,000, can you calculate my loan to value?",
    ],
)
def test_question_clause_is_its_own_unit_in_either_order(text):
    turn = ingest(text)
    analysis = analyze_financial_turn(Memory(), turn)
    values = {
        clause.field: clause.value_refs
        for clause in analysis.clauses
        if clause.kind == "affirmed"
    }
    assert set(values) == {"existing_loan_amount", "property_value"}
    assert len(values["existing_loan_amount"]) == 1
    assert len(values["property_value"]) == 1
    assert is_explicit_ltv_request(turn)


def test_bare_ltv_question_is_an_explicit_request_without_clauses():
    turn = ingest("What is my loan to value?")
    assert is_explicit_ltv_request(turn)
    assert analyze_financial_turn(Memory(), turn).clauses == ()


def test_amounts_without_a_question_are_not_an_explicit_ltv_request():
    turn = ingest("My existing mortgage is 2000000 and property is worth 2500000")
    assert not is_explicit_ltv_request(turn)


def test_comma_before_a_statement_does_not_split_the_unit():
    # A comma that does not open a question keeps the statement one unit. The
    # ", and" grammar was already unsupported before question clauses split, so
    # the clause result is pinned to the pre-change baseline ("uncertain").
    turn = ingest(
        "My mortgage is SEK 2,000,000, and my property is worth SEK 2,500,000."
    )
    assert len(assertion_unit_ranges(turn)) == 1
    analysis = analyze_financial_turn(Memory(), turn)
    assert [(clause.field, clause.kind) for clause in analysis.clauses] == [
        ("existing_loan_amount", "uncertain")
    ]
