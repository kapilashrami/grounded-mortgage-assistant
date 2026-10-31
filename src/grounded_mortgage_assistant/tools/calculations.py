"""Deterministic, evidence-free calculations used by the chat service."""

import re
from decimal import ROUND_HALF_UP, Decimal, InvalidOperation, localcontext

CALCULATION_VERSION = "ltv-v1"
FORMULA = "loan_amount/property_value*100"
_SPACE = r"[ \u00a0\u202f]"
_MONEY = re.compile(
    rf"(?:(?P<prefix>sek)\s+)?(?P<number>(?:\+{_SPACE}*|[\-−])?(?:\d{{1,3}}(?:{_SPACE}\d{{3}})+(?:,\d+)?|\d{{1,3}}(?:,\d{{3}})+(?:\.\d+)?|\d+(?:[,.]\d+)?))\s*(?P<unit>mkr|mnkr|tkr|kkr|kr|kronor|sek|miljon(?:er)?\s+kronor|tusen\s+kronor|(?:million|thousand)\s+(?:sek|kronor|kr))?",
    re.IGNORECASE,
)
_NUMBER_LIKE = re.compile(
    r"(?<![\w.,+\-−/’'])(?:[+\-−][ \u00a0\u202f]*)*(?:[0-9][0-9 \u00a0\u202f.,eE+\-−/’'_]*|NaN\b|Infinity\b|Inf\b|∞)",
    re.IGNORECASE,
)
_UNIT_WORD = re.compile(r"[^\W\d_]\w*", re.UNICODE)
_SEPARATORS = {"och", "men", "eller", "samt", "till", "är", "and", "but", "or"}


class CalculationInputError(ValueError):
    pass


def parse_sek(
    expression: str, language: str = "sv", *, allow_unitless: bool = False
) -> Decimal:
    """Parse one complete Swedish money expression into canonical SEK."""
    if language == "sv" and "." in expression:
        raise CalculationInputError("ogiltigt belopp")
    match = _MONEY.fullmatch(expression.strip())
    if not match:
        raise CalculationInputError("ogiltigt belopp")
    number = match.group("number")
    unit = match.group("unit") or match.group("prefix")
    if unit is None and not allow_unitless:
        raise CalculationInputError("ogiltigt belopp")
    if unit is None:
        unit = "sek"
    unit = unit.casefold()
    if language == "en":
        return _parse_english_sek(number, unit)
    if number.count(",") > 1:
        raise CalculationInputError("ogiltigt belopp")
    number = number.replace("−", "-")
    sign = (
        "-"
        if number.lstrip().startswith("-")
        else "+"
        if number.lstrip().startswith("+")
        else ""
    )
    integer, *fraction = number.lstrip("+-   ").split(",")
    if re.search(_SPACE, integer):
        groups = re.split(_SPACE, integer)
        if not all(len(group) == 3 for group in groups[1:]):
            raise CalculationInputError("ogiltig tusentalsgruppering")
        integer = "".join(groups)
    try:
        digits = (integer + (fraction[0] if fraction else "")).lstrip("0") or "0"
        if len(digits.rstrip("0") or "0") > 17:
            raise CalculationInputError("beloppet har för många siffror")
        with localcontext() as context:
            context.prec = 50
            value = Decimal(sign + integer + ("." + fraction[0] if fraction else ""))
    except InvalidOperation as error:
        raise CalculationInputError("ogiltigt belopp") from error
    scale = (
        Decimal(1000000)
        if unit.startswith(("mkr", "mnkr", "miljon", "million"))
        else Decimal(1000)
        if unit.startswith(("tkr", "kkr", "tusen", "thousand"))
        else Decimal(1)
    )
    with localcontext() as context:
        context.prec = 50
        result = value * scale
        if not result.is_finite() or result.copy_abs() > Decimal("999999999999999.99"):
            raise CalculationInputError("beloppet är för stort")
        cents = result.quantize(Decimal("0.01"))
    if not result.is_finite() or result != cents:
        raise CalculationInputError("beloppet har för många decimaler")
    return result


def canonical_sek(value: Decimal) -> str:
    if not value.is_finite():
        raise CalculationInputError("ogiltigt belopp")
    with localcontext() as context:
        context.prec = 50
        value = value.quantize(Decimal("0.01"))
    if value.is_zero():
        value = Decimal(0)
    return format(value, "f").rstrip("0").rstrip(".") or "0"


def _parse_english_sek(number: str, unit: str) -> Decimal:
    number = re.sub(_SPACE, "", number.replace("−", "-"))
    if re.fullmatch(r"[+-]?\d{1,3},\d{3}", number):
        raise CalculationInputError("ambiguous English amount")
    if "," in number:
        if not re.fullmatch(r"[+-]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?", number):
            raise CalculationInputError("invalid thousands grouping")
        number = number.replace(",", "")
    if not re.fullmatch(r"[+-]?\d+(?:\.\d+)?", number):
        raise CalculationInputError("invalid amount")
    return parse_sek(number.replace(".", ",") + " " + unit, "sv")


def calculate_ltv(loan_amount: Decimal, property_value: Decimal) -> dict[str, str]:
    if not loan_amount.is_finite() or not property_value.is_finite():
        raise CalculationInputError("ogiltigt belopp")
    maximum = Decimal("999999999999999.99")
    if loan_amount.copy_abs() > maximum or property_value.copy_abs() > maximum:
        raise CalculationInputError("beloppet är för stort")
    with localcontext() as context:
        context.prec = 50
        try:
            loan_cents = loan_amount.quantize(Decimal("0.01"))
            value_cents = property_value.quantize(Decimal("0.01"))
        except InvalidOperation as error:
            raise CalculationInputError("ogiltigt belopp") from error
        if loan_amount != loan_cents or property_value != value_cents:
            raise CalculationInputError("beloppet har för många decimaler")
    if loan_amount < 0 or property_value <= 0:
        raise CalculationInputError("beloppet måste vara positivt")
    with localcontext() as context:
        context.prec = 50
        percent = (loan_amount / property_value * 100).quantize(
            Decimal("0.01"), rounding=ROUND_HALF_UP
        )
    if percent.is_zero():
        percent = Decimal(0)
    return {
        "tool": "loan_to_value",
        "formula": FORMULA,
        "version": CALCULATION_VERSION,
        "loan_amount_sek": canonical_sek(loan_amount),
        "property_value_sek": canonical_sek(property_value),
        "percent": format(percent, ".2f"),
        "rounding": "ROUND_HALF_UP, 2 decimal places",
        "assumptions": "Loan amount divided by property value only.",
    }


def _prefix_currency_before_delimiter(
    text: str, start: int, numeric_end: int
) -> tuple[str, int] | None:
    """Split one sentence delimiter only after a complete valid prefix amount."""
    if (
        not re.match(r"(?i)sek\s", text[start:])
        and text[start : start + 1] not in "$€£"
    ):
        return None
    candidates: list[tuple[str, int]] = []
    for index in range(start + 1, numeric_end):
        if text[index] not in ",.":
            continue
        if not text[index - 1].isdigit():
            continue
        following = index + 1
        if following < len(text) and not text[following].isspace():
            continue
        while following < len(text) and text[following].isspace():
            following += 1
        if following < len(text):
            tail = text[following:]
            if tail[0].isdigit() or tail[0] in ",.+-/’'":
                continue
            if re.match(r"[eE][+-]?\d", tail):
                continue
        candidate = text[start:index]
        try:
            parse_sek(candidate, "en")
        except CalculationInputError:
            continue
        candidates.append((candidate, index))
    return candidates[0] if len(candidates) == 1 else None


def money_expressions(text: str) -> list[tuple[str, int, int]]:
    values = []
    cursor = 0
    while item := _NUMBER_LIKE.search(text, cursor):
        start, cursor = item.start(), item.end()
        prefix = re.search(r"(?i)(?<!\w)sek\s+$", text[:start])
        if prefix:
            start = prefix.start()
        if start and text[start - 1] in "$€£":
            start -= 1
        raw_numeric_end = cursor
        numeric_end = cursor
        while numeric_end > start and text[numeric_end - 1] in " \u00a0\u202f":
            numeric_end -= 1
        split = _prefix_currency_before_delimiter(text, start, raw_numeric_end)
        if split is not None:
            candidate, end = split
            values.append((candidate, start, end))
            cursor = item.end()
            continue
        cursor = item.end()
        while cursor < len(text) and text[cursor] in " \u00a0\u202f":
            cursor += 1
        prefix_or_suffix_currency = cursor < len(text) and text[cursor] in "$€£"
        if prefix_or_suffix_currency:
            cursor += 1
        word = _UNIT_WORD.match(text, cursor)
        # A sentence or clause delimiter closes the amount; the next word starts
        # a new clause and must not be read as a unit ("2500000. What ...").
        delimited = text[item.start() : item.end()].rstrip(" \u00a0\u202f")[-1:] in {
            ".",
            ",",
        }
        if word and word.group().casefold() not in _SEPARATORS and not delimited:
            cursor = word.end()
            if word.group().casefold() in {
                "miljon",
                "miljoner",
                "tusen",
                "million",
                "thousand",
            }:
                while cursor < len(text) and text[cursor] in " \u00a0\u202f":
                    cursor += 1
                second = _UNIT_WORD.match(text, cursor)
                if second and second.group().casefold() not in _SEPARATORS:
                    cursor = second.end()
        else:
            cursor = cursor if prefix_or_suffix_currency else numeric_end
        if cursor < len(text) and text[cursor] in "$€£":
            cursor += 1
        if cursor > start and text[cursor - 1] in ".,":
            previous = text[cursor - 2] if cursor > start + 1 else ""
            following = text[cursor] if cursor < len(text) else ""
            if previous.isdigit() and (not following or not following.isdigit()):
                cursor -= 1
        values.append((text[start:cursor], start, cursor))
        cursor = max(cursor, item.end())
    return values
