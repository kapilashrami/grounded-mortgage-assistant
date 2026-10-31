from decimal import Decimal

import pytest

from grounded_mortgage_assistant.tools.calculations import (
    CalculationInputError,
    calculate_ltv,
    money_expressions,
    parse_sek,
)


@pytest.mark.parametrize(
    ("loan", "value", "expected"),
    [
        ("2,4 mkr", "3 mkr", "80.00"),
        ("1 kr", "333,33 kr", "0.30"),
        ("1 kr", "3,23 kr", "30.96"),
        ("0 kr", "3 mkr", "0.00"),
        ("3,3 mkr", "3 mkr", "110.00"),
    ],
)
def test_ltv_is_decimal_and_rounds_half_up(loan, value, expected):
    assert calculate_ltv(parse_sek(loan), parse_sek(value))["percent"] == expected


@pytest.mark.parametrize(
    "value", ["1.2 kr", "1 23 kr", "1 usd", "1 kr + 2 kr", "-1 kr"]
)
def test_invalid_money_is_rejected(value):
    if value == "-1 kr":
        with pytest.raises(CalculationInputError):
            calculate_ltv(parse_sek(value), Decimal(3))
    else:
        with pytest.raises(CalculationInputError):
            parse_sek(value)


def test_ltv_canonicalizes_negative_zero_and_rejects_unrepresentable_amounts():
    assert calculate_ltv(Decimal("-0"), Decimal(3)) == {
        "tool": "loan_to_value",
        "formula": "loan_amount/property_value*100",
        "version": "ltv-v1",
        "loan_amount_sek": "0",
        "property_value_sek": "3",
        "percent": "0.00",
        "rounding": "ROUND_HALF_UP, 2 decimal places",
        "assumptions": "Loan amount divided by property value only.",
    }
    with pytest.raises(CalculationInputError):
        calculate_ltv(Decimal("1E100"), Decimal(3))


def test_ltv_direct_decimal_boundary_and_rounding_contract():
    assert calculate_ltv(Decimal(1), Decimal(32))["percent"] == "3.13"
    assert calculate_ltv(Decimal(0), Decimal(3))["percent"] == "0.00"
    maximum = Decimal("999999999999999.99")
    assert (
        calculate_ltv(Decimal(1), maximum)["property_value_sek"] == "999999999999999.99"
    )


@pytest.mark.parametrize(
    ("expression", "language", "expected"),
    [
        ("SEK 2,400,000", "en", Decimal(2400000)),
        ("SEK\t2,400,000", "en", Decimal(2400000)),
        ("2,400,000 SEK", "en", Decimal(2400000)),
        ("2,4 mkr", "sv", Decimal(2400000)),
    ],
)
def test_parse_sek_accepts_complete_prefix_and_suffix_currency_expressions(
    expression, language, expected
):
    assert parse_sek(expression, language) == expected


def test_money_expression_boundaries_keep_prefix_currency_and_exclude_english_delimiters():
    text = (
        "Calculate the loan-to-value ratio for a mortgage of SEK 2,400,000 and a property "
        "value of SEK 3,000,000."
    )
    assert money_expressions(text) == [
        ("SEK 2,400,000", 52, 65),
        ("SEK 3,000,000", 90, 103),
    ]


@pytest.mark.parametrize("space", ["  ", "\u00a0", "\t"])
def test_money_expression_boundaries_include_complete_prefix_currency_whitespace(space):
    text = f"SEK{space}2,400,000 and SEK{space}3,000,000"
    first = f"SEK{space}2,400,000"
    second = f"SEK{space}3,000,000"
    assert money_expressions(text) == [
        (first, 0, len(first)),
        (second, len(first) + 5, len(text)),
    ]


@pytest.mark.parametrize(
    ("text", "amount", "end"),
    [
        ("SEK 2,400,000, next", "SEK 2,400,000", 13),
        ("SEK 2,400,000. next", "SEK 2,400,000", 13),
        ("SEK 2,400,000; next", "SEK 2,400,000", 13),
        ("SEK 2,400,000, eagle", "SEK 2,400,000", 13),
    ],
)
def test_prefix_currency_sentence_delimiter_uses_exact_valid_slice(text, amount, end):
    assert money_expressions(text) == [(amount, 0, end)]


@pytest.mark.parametrize(
    "text",
    [
        "SEK 2,400,000,, next",
        "SEK 2,400,000, e3",
        "SEK 2,400,000, e+3",
        "SEK 2,400,000, 3",
        "SEK 2,400,000, +3",
        "SEK 2,400,000, /3",
    ],
)
def test_prefix_currency_delimiter_does_not_strip_numeric_or_malformed_continuation(
    text,
):
    amount = money_expressions(text)[0][0]
    with pytest.raises(CalculationInputError):
        parse_sek(amount, "en")


def test_prefix_currency_decimal_continuation_is_not_split_as_sentence_punctuation():
    assert money_expressions("SEK 2,400,000.5") == [("SEK 2,400,000.5", 0, 15)]


def test_money_expressions_retains_complete_sentence_final_span():
    assert money_expressions("SEK 2,400,000. What now?")[0] == ("SEK 2,400,000", 0, 13)
