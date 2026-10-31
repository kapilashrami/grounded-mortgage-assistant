from decimal import Decimal

import pytest

from grounded_mortgage_assistant.guardrails.privacy import (
    IngressError,
    display_projection,
    ingest,
)


def test_ingress_replaces_complete_sek_expression_with_owned_opaque_handle():
    turn = ingest("My mortgage is SEK 1,560,000.")

    assert turn.protected_values[0].value == Decimal(1560000)
    assert "1,560,000" not in " ".join(turn.model_projection)
    assert turn.protected_values[0].value_ref in " ".join(turn.model_projection)
    assert turn.tokens[-1].symbol == "SENTENCE_END"


def test_private_canary_is_absent_from_every_safe_projection():
    canary = "1,560,000"
    turn = ingest(f"My mortgage is SEK {canary}.")

    assert canary not in str(turn.model_projection)
    assert canary not in str([(token.symbol, token.index) for token in turn.tokens])


def test_need_inflections_have_distinct_canonical_tokens():
    assert ingest("I need help.").model_projection[1] == "NEED"
    assert ingest("I needs help.").model_projection[1] == "NEEDS"
    assert ingest("I needed help.").model_projection[1] == "NEEDED"
    assert ingest("I needing help.").model_projection[1] == "NEEDING"


def test_comma_delimited_amounts_produce_two_exact_private_handles():
    turn = ingest(
        "My mortgage is SEK 1,820,000, and my property value is SEK 2,600,000."
    )
    assert len(turn.protected_values) == 2
    values = [
        token.symbol for token in turn.tokens if token.symbol.startswith("VALUE(")
    ]
    assert values == [
        f"VALUE({turn.protected_values[0].value_ref})",
        f"VALUE({turn.protected_values[1].value_ref})",
    ]


@pytest.mark.parametrize(
    "text, code",
    [
        ("Call me at jane@example.com", "SENSITIVE_TEXT"),
        ("My codeword is zebra", "UNKNOWN_TEXT"),
    ],
)
def test_sensitive_or_unknown_text_fails_closed_without_echo(text, code):
    with pytest.raises(IngressError) as error:
        ingest(text)
    assert error.value.code == code
    assert text not in str(error.value)


def test_quote_and_conditional_boundaries_are_canonicalized():
    turn = ingest('If my friend says "my mortgage is SEK 1,560,000".')

    assert turn.model_projection == (
        "IF",
        "MY",
        "THIRD_PARTY",
        "SAYS",
        "QUOTE_START",
        "MY",
        "MORTGAGE",
        "IS",
        f"VALUE({turn.protected_values[0].value_ref})",
        "QUOTE_END",
        "SENTENCE_END",
    )


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        (
            "I want to refinance my existing mortgage.",
            "I want to refinance my existing mortgage.",
        ),
        ("Draft an application for me", "Draft an application for me"),
        (
            "Can you help me with a extra amount?",
            "Can you help me with an extra amount?",
        ),
        ("what is a LTV", "What is an LTV"),
        ("i don't need it. thanks", "I don't need it. Thanks"),
        (
            'My partner says "we want a better rate"',
            'My ‹person› says "we want a better rate"',
        ),
        ("follow up on the value", "Follow up on the value"),
        ("a followup", "A follow-up"),
    ],
)
def test_display_projection_reads_as_english(text, expected):
    assert display_projection(ingest(text).model_projection) == expected


def test_display_projection_masks_amounts():
    turn = ingest("My mortgage is SEK 1,560,000.")

    assert display_projection(turn.model_projection) == "My mortgage is ‹amount›."


def test_display_projection_rejects_symbols_outside_the_vocabulary():
    with pytest.raises(IngressError):
        display_projection(("SELF", "WANT", "Kartick"))
