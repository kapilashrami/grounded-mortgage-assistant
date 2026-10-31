import pytest

from grounded_mortgage_assistant.guardrails.privacy import OriginalProof, ingest
from grounded_mortgage_assistant.llm.interpretation_candidates import (
    CandidateSelectionError,
    TurnSelection,
    build_candidate_catalog,
    expand_selection,
    knowledge_query_is_eligible,
)
from grounded_mortgage_assistant.state.memory import Memory, PendingQuestion, stage_turn


def selection(**changes):
    base = {
        "subject": "self",
        "conditional": False,
        "requested_action": "acknowledge",
        "fact_candidate_ids": [],
        "scope_candidate_ids": [],
    }
    base.update(changes)
    return TurnSelection(**base)


def test_catalog_ids_are_deterministic_and_expand_scope_atomically():
    turn = ingest(
        "I want to refinance my existing mortgage without extra borrowing.",
        turn_ref="server_turn",
    )
    catalog = build_candidate_catalog(Memory(), turn)
    assert list(catalog.scope_candidates) == ["sc_server_turn_1"]
    proposal = expand_selection(
        catalog, selection(scope_candidate_ids=["sc_server_turn_1"])
    )
    assert (proposal.goal, proposal.ownership, proposal.borrowing) == (
        "refinance",
        "own_existing",
        "same",
    )


def test_foreign_duplicate_and_conflicting_candidate_selections_fail_closed():
    turn = ingest("My mortgage is SEK 1560000.", turn_ref="server_turn")
    catalog = build_candidate_catalog(Memory(), turn)
    candidate_id = next(iter(catalog.fact_candidates))
    with pytest.raises(CandidateSelectionError, match="INVALID_CANDIDATE_SELECTION"):
        expand_selection(catalog, selection(fact_candidate_ids=["fc_other_1"]))
    with pytest.raises(ValueError, match="duplicate fact candidate IDs"):
        selection(fact_candidate_ids=[candidate_id, candidate_id])


def test_catalog_selection_stages_pending_amount_and_compound_property_type():
    basis = ingest("I want to refinance my existing mortgage.", turn_ref="basis")
    scope_catalog = build_candidate_catalog(Memory(), basis)
    state = stage_turn(
        Memory(),
        basis,
        expand_selection(
            scope_catalog,
            selection(scope_candidate_ids=list(scope_catalog.scope_candidates)),
        ),
    ).state
    pending = PendingQuestion(
        "existing_loan_amount",
        state.revision,
        "pending",
        state.scope_proofs["scope"][0],
    )
    amount = ingest("SEK 1560000.", turn_ref="amount")
    catalog = build_candidate_catalog(
        Memory(**{**state.__dict__, "pending_question": pending}), amount
    )
    proposal = expand_selection(
        catalog, selection(fact_candidate_ids=list(catalog.fact_candidates))
    )
    staged = stage_turn(
        Memory(**{**state.__dict__, "pending_question": pending}), amount, proposal
    )
    assert (
        staged.state.facts["existing_loan_amount"].value_ref
        == amount.protected_values[0].value_ref
    )

    type_pending = PendingQuestion(
        "property_type", state.revision, "type", state.scope_proofs["scope"][0]
    )
    property_type = ingest("Terraced house or apartment.", turn_ref="type")
    catalog = build_candidate_catalog(
        Memory(**{**state.__dict__, "pending_question": type_pending}), property_type
    )
    proposal = expand_selection(
        catalog, selection(fact_candidate_ids=list(catalog.fact_candidates))
    )
    assert proposal.operations[0].enum_values == ["terraced_house", "apartment"]


def test_empty_selection_and_stale_pending_produce_no_positive_candidates():
    turn = ingest("My mortgage is not SEK 1560000.", turn_ref="negative")
    assert not build_candidate_catalog(Memory(), turn).fact_candidates
    stale = PendingQuestion(
        "property_value", 1, "stale", OriginalProof("old", 0, 1, "x")
    )
    bare = ingest("SEK 2600000.", turn_ref="bare")
    catalog = build_candidate_catalog(Memory(pending_question=stale), bare)
    assert not catalog.fact_candidates
    proposal = expand_selection(catalog, selection())
    assert not proposal.operations


def test_knowledge_projection_is_current_canonical_ordered_and_server_owned():
    prior = ingest("I want to refinance my mortgage.", turn_ref="prior")
    state = stage_turn(
        Memory(),
        prior,
        expand_selection(
            build_candidate_catalog(Memory(), prior),
            selection(scope_candidate_ids=["sc_prior_1"]),
        ),
    ).state
    turn = ingest("What information about mortgage information?", turn_ref="current")
    catalog = build_candidate_catalog(state, turn)
    assert catalog.knowledge_projection == ("INFORMATION", "MORTGAGE")
    assert "REFINANCE" not in catalog.knowledge_projection
    proposal = expand_selection(catalog, selection(requested_action="knowledge"))
    assert proposal.knowledge_question == ["INFORMATION", "MORTGAGE"]
    assert knowledge_query_is_eligible(proposal)
    empty = expand_selection(
        build_candidate_catalog(Memory(), ingest("Thanks.", turn_ref="empty")),
        selection(requested_action="knowledge"),
    )
    assert empty.knowledge_question == []
    assert not knowledge_query_is_eligible(empty)


def test_knowledge_tokens_are_not_a_selection_field():
    with pytest.raises(ValueError, match="Extra inputs are not permitted"):
        selection(knowledge_tokens=[])
