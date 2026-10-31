"""Fail-closed local ingress for the restricted-English synthetic prototype.

This is deliberately not a general PII detector.  Text which cannot be
represented by the reviewed vocabulary is rejected before any model boundary.
"""

from __future__ import annotations

import re
import secrets
from dataclasses import dataclass
from decimal import Decimal
from itertools import pairwise

from grounded_mortgage_assistant.tools.calculations import (
    CalculationInputError,
    money_expressions,
    parse_sek,
)


class IngressError(ValueError):
    """A fixed, non-echoing ingress failure category."""

    def __init__(self, code: str):
        self.code = code
        super().__init__(code)


@dataclass(frozen=True)
class OriginalProof:
    turn_ref: str
    start: int
    end: int
    original: str


@dataclass(frozen=True)
class ProtectedValue:
    value_ref: str
    owner_turn_ref: str
    value: Decimal
    original_proof: OriginalProof
    currency_explicit: bool = True


@dataclass(frozen=True)
class CanonicalToken:
    symbol: str
    index: int
    original_start: int
    original_end: int


@dataclass(frozen=True)
class SanitizedTurn:
    turn_ref: str
    tokens: tuple[CanonicalToken, ...]
    protected_values: tuple[ProtectedValue, ...]
    # This private map is retained so staged facts can prove source spans.  It
    # is never included by ``model_projection``.
    original: str

    @property
    def model_projection(self) -> tuple[str, ...]:
        return tuple(token.symbol for token in self.tokens)

    def token_span(self, start: int, end: int) -> tuple[int, int]:
        if not (0 <= start < end <= len(self.tokens)):
            raise IngressError("INVALID_CANONICAL_SPAN")
        return self.tokens[start].original_start, self.tokens[end - 1].original_end


def detector_text_with_approved_amounts(
    turn: SanitizedTurn, approved_value_refs: tuple[str, ...]
) -> str:
    """Replace only verified financial value proofs in detector-only text."""
    if not approved_value_refs:
        raise IngressError("INVALID_AMOUNT_PROJECTION")
    values = {value.value_ref: value for value in turn.protected_values}
    spans: list[tuple[int, int]] = []
    for value_ref in approved_value_refs:
        value = values.get(value_ref)
        if value is None:
            raise IngressError("INVALID_AMOUNT_PROJECTION")
        proof = value.original_proof
        if (
            value.owner_turn_ref != turn.turn_ref
            or proof.turn_ref != turn.turn_ref
            or proof.start < 0
            or proof.end <= proof.start
            or proof.end > len(turn.original)
            or proof.original != turn.original[proof.start : proof.end]
        ):
            raise IngressError("INVALID_AMOUNT_PROJECTION")
        spans.append((proof.start, proof.end))
    spans.sort()
    if any(end > next_start for (_, end), (next_start, _) in pairwise(spans)):
        raise IngressError("INVALID_AMOUNT_PROJECTION")
    parts: list[str] = []
    cursor = 0
    for start, end in spans:
        parts.extend((turn.original[cursor:start], "amount"))
        cursor = end
    parts.append(turn.original[cursor:])
    return "".join(parts)


_SENSITIVE = (
    re.compile(r"[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}"),
    re.compile(r"(?:\+?\d[\s-]?){8,}\d"),
    re.compile(
        r"\b(?:personnummer|address|addressen|telefon|phone|email)\b", re.IGNORECASE
    ),
)
_PUNCTUATION = {
    ".": "SENTENCE_END",
    "?": "QUESTION",
    "!": "EXCLAMATION",
    ",": "COMMA",
    ";": "SEMICOLON",
    ":": "COLON",
}
_WORDS = {
    "i": "SELF",
    "my": "MY",
    "me": "SELF",
    "we": "SELF_PLURAL",
    "our": "OUR",
    "the": "THE",
    "a": "A",
    "an": "A",
    "this": "THIS",
    "that": "THAT",
    "is": "IS",
    "are": "ARE",
    "was": "WAS",
    "be": "BE",
    "have": "HAVE",
    "has": "HAS",
    "want": "WANT",
    # Need inflections are distinct historical/modal canonical tokens. Intent
    # parsers deliberately accept only the reviewed present-tense productions.
    "need": "NEED",
    "needs": "NEEDS",
    "needed": "NEEDED",
    "needing": "NEEDING",
    "please": "PLEASE",
    "can": "CAN",
    "could": "COULD",
    "would": "WOULD",
    "like": "LIKE",
    "keep": "KEEP",
    "more": "MORE",
    "am": "AM",
    "will": "WILL",
    "looking": "LOOKING",
    "unchanged": "UNCHANGED",
    "you": "YOU",
    "help": "HELP",
    "with": "WITH",
    "for": "FOR",
    "to": "TO",
    "of": "OF",
    "on": "ON",
    "in": "IN",
    "and": "AND",
    "or": "OR",
    "but": "BUT",
    "not": "NOT",
    "no": "NO",
    "don't": "DO_NOT",
    "do": "DO",
    "does": "DOES",
    "if": "IF",
    "when": "WHEN",
    "unless": "UNLESS",
    "maybe": "MAYBE",
    "perhaps": "MAYBE",
    "their": "THEIR",
    "they": "THIRD_PARTY",
    "he": "THIRD_PARTY",
    "she": "THIRD_PARTY",
    "someone": "THIRD_PARTY",
    "friend": "THIRD_PARTY",
    "partner": "THIRD_PARTY",
    "says": "SAYS",
    "said": "SAID",
    "mortgage": "MORTGAGE",
    "mortgages": "MORTGAGE",
    "loan": "LOAN",
    "loans": "LOAN",
    "existing": "EXISTING",
    "current": "CURRENT",
    "property": "PROPERTY",
    "home": "PROPERTY",
    "house": "HOUSE",
    "apartment": "APARTMENT",
    "flat": "APARTMENT",
    "terraced": "TERRACED",
    "holiday": "HOLIDAY",
    "value": "VALUE_WORD",
    "values": "VALUE_WORD",
    "worth": "WORTH",
    "amount": "AMOUNT",
    "refinance": "REFINANCE",
    "refinancing": "REFINANCE",
    "rate": "RATE",
    "rates": "RATE",
    "better": "BETTER",
    "same": "SAME",
    "extra": "EXTRA",
    "without": "WITHOUT",
    "increase": "INCREASE",
    "increased": "INCREASE",
    "increasing": "INCREASE",
    "borrowing": "BORROWING",
    "borrow": "BORROWING",
    "borrowed": "BORROWING",
    "calculate": "CALCULATE",
    "ltv": "LTV",
    "loan-to-value": "LTV",
    "draft": "DRAFT",
    "application": "APPLICATION",
    "information": "INFORMATION",
    "forget": "FORGET",
    "remove": "FORGET",
    "delete": "FORGET",
    "withdraw": "WITHDRAW",
    "change": "CHANGE",
    "correct": "CHANGE",
    "correction": "CHANGE",
    "correcting": "CHANGE",
    "update": "CHANGE",
    "review": "REVIEW",
    "decline": "DECLINE",
    "yes": "YES",
    "thanks": "THANKS",
    "actually": "ACTUALLY",
    "now": "NOW",
    "thank": "THANK",
    "what": "WHAT",
    "how": "HOW",
    "it": "IT",
    "all": "ALL",
    "details": "DETAILS",
    "from": "FROM",
    "about": "ABOUT",
    "follow": "FOLLOW_UP",
    "followup": "FOLLOW_UP",
    "up": "UP",
}

# Match supported multiword and hyphenated aliases before ordinary words. This
# keeps canonical lookup and the exact original span intact.
_ALIASES = {"loan to value": "LTV"}
_COMPOUND_WORDS = tuple(
    sorted(
        (*_ALIASES, *(word for word in _WORDS if "-" in word)), key=len, reverse=True
    )
)
_WORD = re.compile(
    rf"{'|'.join(re.escape(word) for word in _COMPOUND_WORDS)}|[A-Za-z]+(?:'[A-Za-z]+)?|\d|[^\s]",
    re.IGNORECASE,
)


def _new_ref(prefix: str) -> str:
    return f"{prefix}_{secrets.token_urlsafe(18)}"


def ingest(
    text: str, *, language: str = "en", turn_ref: str | None = None
) -> SanitizedTurn:
    """Convert a complete supported turn to server-owned symbols and handles."""
    if language != "en" or not text or len(text) > 4000:
        raise IngressError("UNSUPPORTED_TEXT")
    if any(pattern.search(text) for pattern in _SENSITIVE):
        raise IngressError("SENSITIVE_TEXT")
    turn_ref = turn_ref or _new_ref("t")
    amounts: dict[tuple[int, int], ProtectedValue] = {}
    for expression, start, end in money_expressions(text):
        try:
            value = parse_sek(expression, "en", allow_unitless=True)
        except CalculationInputError as error:
            raise IngressError("INVALID_MONEY") from error
        proof = OriginalProof(turn_ref, start, end, expression)
        explicit_currency = any(
            marker in expression.casefold()
            for marker in (
                "sek",
                "kr",
                "kronor",
                "mkr",
                "mnkr",
                "tkr",
                "kkr",
                "$",
                "€",
                "£",
            )
        )
        amounts[(start, end)] = ProtectedValue(
            _new_ref("v"), turn_ref, value, proof, explicit_currency
        )

    tokens: list[CanonicalToken] = []
    cursor = 0
    in_quote = False
    for match in _WORD.finditer(text):
        start, end, item = match.start(), match.end(), match.group()
        if start < cursor:
            continue
        matching_amount = next(
            ((span, value) for span, value in amounts.items() if span[0] == start), None
        )
        if matching_amount:
            (amount_start, amount_end), protected = matching_amount
            tokens.append(
                CanonicalToken(
                    f"VALUE({protected.value_ref})",
                    len(tokens),
                    amount_start,
                    amount_end,
                )
            )
            cursor = amount_end
            continue
        if item in {'"', "'"}:
            tokens.append(
                CanonicalToken(
                    "QUOTE_END" if in_quote else "QUOTE_START", len(tokens), start, end
                )
            )
            in_quote = not in_quote
        elif item in _PUNCTUATION:
            tokens.append(CanonicalToken(_PUNCTUATION[item], len(tokens), start, end))
        elif item.casefold() in _ALIASES or item.casefold() in _WORDS:
            symbol = _ALIASES.get(item.casefold()) or _WORDS.get(item.casefold())
            tokens.append(CanonicalToken(symbol, len(tokens), start, end))
        elif item.isalpha() or "'" in item:
            raise IngressError("UNKNOWN_TEXT")
        else:
            raise IngressError("UNSUPPORTED_TEXT")
        cursor = end
    if in_quote:
        raise IngressError("UNSUPPORTED_TEXT")
    if not tokens:
        raise IngressError("UNSUPPORTED_TEXT")
    return SanitizedTurn(turn_ref, tuple(tokens), tuple(amounts.values()), text)
