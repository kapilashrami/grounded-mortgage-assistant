"""Pure, private working memory for the restricted synthetic mortgage demo."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field, replace
from datetime import datetime
from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from grounded_mortgage_assistant.guardrails.privacy import (
    CanonicalToken,
    OriginalProof,
    ProtectedValue,
    SanitizedTurn,
)


class MemoryError(ValueError):
    pass


class FactStatus(StrEnum):
    KNOWN = "known"
    AMBIGUOUS = "ambiguous"
    WITHDRAWN = "withdrawn"


# The frozen interpreter contract and the only facts memory stores.
FieldName = Literal["existing_loan_amount", "property_value", "property_type"]
Subject = Literal["self", "third_party", "hypothetical", "quoted", "unknown"]
SAFE_KNOWLEDGE_TOKENS = frozenset(
    {
        "REFINANCE",
        "MORTGAGE",
        "LOAN",
        "RATE",
        "PROPERTY",
        "LTV",
        "INFORMATION",
        "CALCULATE",
        "REVIEW",
    }
)
_BOUNDARIES = frozenset({"SENTENCE_END", "QUESTION", "EXCLAMATION", "SEMICOLON"})
# Canonical openings of an explicit personal LTV request ("What is my loan to
# value, ...", "Calculate my LTV.").
_EXPLICIT_LTV_PREFIXES = (
    ("WHAT", "IS", "MY", "LTV"),
    ("CAN", "YOU", "CALCULATE", "MY", "LTV"),
    ("CALCULATE", "MY", "LTV"),
    ("CALCULATE", "LTV"),
    ("PLEASE", "CALCULATE", "MY", "LTV"),
)
# Canonical words that open an interrogative clause after a comma.
_QUESTION_OPENERS = frozenset({"WHAT", "HOW", "CAN", "COULD", "WOULD", "IS"})


def is_explicit_ltv_request(turn: SanitizedTurn) -> bool:
    """True when any clause of the turn opens with an explicit personal LTV request."""
    return any(
        _unit(turn, span)[: len(prefix)] == prefix
        for span in assertion_unit_ranges(turn)
        for prefix in _EXPLICIT_LTV_PREFIXES
    )


_ROLE = {
    "existing_loan_amount": frozenset({"MORTGAGE", "LOAN"}),
    "property_value": frozenset({"PROPERTY"}),
}


class SourceSpan(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)
    token_start: int = Field(ge=0)
    token_end: int = Field(gt=0)

    @model_validator(mode="after")
    def ordered(self):
        if self.token_end <= self.token_start:
            raise ValueError("source span must be ordered")
        return self


class FactOperation(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)
    field: FieldName
    operation: Literal["set", "ambiguous", "withdraw"]
    value_refs: list[str] = Field(default_factory=list, max_length=2)
    enum_value: (
        Literal["house", "terraced_house", "apartment", "holiday_home"] | None
    ) = None
    enum_values: list[
        Literal["house", "terraced_house", "apartment", "holiday_home"]
    ] = Field(default_factory=list, max_length=4)
    source_span: SourceSpan

    @model_validator(mode="after")
    def shape(self):
        numeric = self.field != "property_type"
        if self.operation == "set" and (
            numeric != (len(self.value_refs) == 1)
            or (self.enum_value is not None) != (not numeric)
            or self.enum_values
        ):
            raise ValueError("invalid set shape")
        if (
            self.operation == "ambiguous"
            and numeric
            and (
                len(self.value_refs) != 2
                or self.enum_value is not None
                or self.enum_values
            )
        ):
            raise ValueError("numeric ambiguity needs two values")
        if (
            self.operation == "ambiguous"
            and not numeric
            and (
                self.value_refs
                or self.enum_value is not None
                or not 2 <= len(self.enum_values) <= 4
                or len(set(self.enum_values)) != len(self.enum_values)
            )
        ):
            raise ValueError("property ambiguity needs distinct enum values")
        if self.operation == "withdraw" and (
            self.value_refs or self.enum_value is not None or self.enum_values
        ):
            raise ValueError("withdraw has no value")
        return self


class TurnInterpretation(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)
    subject: Subject
    conditional: bool
    operations: list[FactOperation] = Field(default_factory=list, max_length=3)
    goal: Literal["refinance", "information", "none", "unknown", "unchanged"]
    borrowing: Literal["same", "increase", "unknown", "unchanged"]
    ownership: Literal["own_existing", "other", "unknown", "unchanged"]
    requested_action: Literal[
        "knowledge",
        "calculate",
        "acknowledge",
        "review",
        "decline",
        "forget",
        "clarify",
    ]
    knowledge_question: list[str] | None = Field(default=None, max_length=128)
    scope_source_spans: list[SourceSpan] = Field(default_factory=list, max_length=4)
    unresolved: list[
        Literal[
            "amount", "property_value", "property_type", "scope", "goal", "borrowing"
        ]
    ] = Field(default_factory=list, max_length=6)

    @model_validator(mode="after")
    def coherent(self):
        if len({x.field for x in self.operations}) != len(self.operations):
            raise ValueError("duplicate field operation")
        if self.knowledge_question is not None and self.requested_action != "knowledge":
            raise ValueError("only knowledge can carry a question")
        if self.knowledge_question and any(
            x not in SAFE_KNOWLEDGE_TOKENS for x in self.knowledge_question
        ):
            raise ValueError("knowledge question must use safe canonical vocabulary")
        return self


@dataclass(frozen=True)
class Fact:
    fact_id: str
    field: FieldName
    field_revision: int
    status: FactStatus
    value_ref: str | None = None
    enum_value: str | None = None
    enum_values: tuple[str, ...] = ()
    original_proof: OriginalProof | tuple[OriginalProof, ...] | None = None


@dataclass(frozen=True)
class SafeExchange:
    turn_ref: str
    tokens: tuple[str, ...]
    response: tuple[str, ...] = ()


@dataclass(frozen=True)
class ReplayRecord:
    fingerprint: str
    outcome: str
    response: tuple[str, ...] | None


@dataclass(frozen=True)
class PendingQuestion:
    """A fixed server question which may bind one later bare numeric reply."""

    field: FieldName
    asked_revision: int
    question_id: str
    personal_basis: OriginalProof


@dataclass(frozen=True)
class ScopeParts:
    core: tuple[str, ...]
    no_increase_suffix: bool


@dataclass(frozen=True)
class Memory:
    revision: int = 0
    facts: dict[FieldName, Fact] = field(default_factory=dict)
    protected_values: dict[str, ProtectedValue] = field(default_factory=dict)
    turns: dict[str, SanitizedTurn] = field(default_factory=dict)
    goal: str = "unknown"
    scope: str = "unknown"
    borrowing: str = "unknown"
    pending_question: PendingQuestion | None = None
    last_operation: str | None = None
    # Active calculation intent survives a supported correction; it is distinct
    # from the latest conversational classification.
    active_operation: str | None = None
    offer: dict | None = None
    accepted_review: dict | None = None
    exchanges: tuple[SafeExchange, ...] = ()
    replay: dict[str, ReplayRecord] = field(default_factory=dict)
    forgotten_turns: frozenset[str] = frozenset()
    forgotten_client_ids: frozenset[str] = frozenset()
    declined_offer: bool = False
    accepted_turns: int = 0
    client_turns: dict[str, str] = field(default_factory=dict)
    scope_proofs: dict[
        Literal["goal", "scope", "borrowing"], tuple[OriginalProof, ...]
    ] = field(default_factory=dict)
    draft_requested: bool = False
    last_subject: Subject = "unknown"
    last_conditional: bool = False


_CALCULATION_FIELDS = frozenset({"existing_loan_amount", "property_value"})


def _invalidated(changed: bool, changed_fields: Iterable[str]) -> tuple[str, ...]:
    """An LTV depends only on the two amounts; a draft depends on every fact."""
    if not changed:
        return ()
    calculation = ("calculation",) if _CALCULATION_FIELDS & set(changed_fields) else ()
    return (*calculation, "offer", "review")


@dataclass(frozen=True)
class StageResult:
    state: Memory
    changes: tuple[Fact, ...]
    invalidated: tuple[str, ...]
    proposal_rejected: bool = False
    rejection_code: str | None = None


@dataclass(frozen=True)
class ScopeObservation:
    value: str
    proofs: tuple[OriginalProof, ...]


@dataclass(frozen=True)
class SourceFactRestriction:
    field: FieldName
    proofs: tuple[OriginalProof, ...]
    reason: Literal["negated_value", "uncertain_value", "conflicting_values"]


@dataclass(frozen=True)
class FinancialClause:
    field: FieldName
    unit_span: SourceSpan
    clause_span: SourceSpan
    kind: Literal["affirmed", "negated", "uncertain", "alternatives"]
    value_refs: tuple[str, ...] = ()
    enum_values: tuple[str, ...] = ()

    proof: OriginalProof | None = None

    @property
    def operation(self) -> Literal["set", "ambiguous"]:
        return "ambiguous" if len(self.value_refs or self.enum_values) > 1 else "set"


@dataclass(frozen=True)
class FinancialAnalysis:
    clauses: tuple[FinancialClause, ...]
    conflicts: dict[FieldName, tuple[OriginalProof, ...]]


@dataclass(frozen=True)
class ContextLimits:
    input_tokens: int = 8192
    output_reservation: int = 1800
    model_context_tokens: int | None = None
    mandatory_tokens: int = 0
    token_counter: Callable[[tuple[str, ...]], int] | None = None


def _identity_ids(state: Memory) -> set[str]:
    return set(state.client_turns) | set(state.replay) | set(state.forgotten_client_ids)


def _response_bytes(response: tuple[str, ...]) -> int:
    return len(json.dumps(response, ensure_ascii=False, separators=(",", ":")).encode())


def _exchange_bytes(exchange: SafeExchange) -> int:
    return len(
        json.dumps(
            (exchange.turn_ref, exchange.tokens, exchange.response),
            ensure_ascii=False,
            separators=(",", ":"),
        ).encode()
    )


def _check_event_identity(state: Memory, turn_ref: str, client_message_id: str) -> None:
    """Validate identity for either a text turn or a server-generated answer."""
    if (
        turn_ref in state.forgotten_turns
        or client_message_id in state.forgotten_client_ids
    ):
        raise MemoryError("FORGOTTEN_TURN")
    if client_message_id in state.client_turns:
        raise MemoryError(
            "REPLAY_EXPIRED"
            if state.client_turns[client_message_id] == turn_ref
            else "REPLAY_CONFLICT"
        )
    if state.accepted_turns >= 100:
        raise MemoryError("TURN_CAPACITY_EXCEEDED")
    if len(_identity_ids(state) | {client_message_id}) > 100:
        raise MemoryError("IDENTITY_CAPACITY_EXCEEDED")


def record_control_event(
    state: Memory, *, client_message_id: str, control_ref: str
) -> Memory:
    """Record a server control action without manufacturing a customer turn."""
    _check_event_identity(state, control_ref, client_message_id)
    return replace(
        state,
        accepted_turns=state.accepted_turns + 1,
        client_turns={**state.client_turns, client_message_id: control_ref},
    )


def _check_turn_identity(
    state: Memory, turn: SanitizedTurn, client_message_id: str
) -> None:
    _check_event_identity(state, turn.turn_ref, client_message_id)


def _retained_turns(
    state: Memory,
    turn: SanitizedTurn,
    facts: dict[FieldName, Fact],
    scope_proofs: dict[str, tuple[OriginalProof, ...]],
) -> dict[str, SanitizedTurn]:
    live_refs = {
        proof.turn_ref
        for fact in facts.values()
        for proof in (
            fact.original_proof
            if isinstance(fact.original_proof, tuple)
            else (fact.original_proof,)
        )
        if proof
    }
    live_refs.update(
        proof.turn_ref for proofs in scope_proofs.values() for proof in proofs
    )
    retained = {**state.turns, turn.turn_ref: turn}
    retained = {ref: saved for ref, saved in retained.items() if ref in live_refs}
    if (
        len(retained) > 32
        or sum(len(saved.original.encode()) for saved in retained.values()) > 128 * 1024
    ):
        raise MemoryError("PROTECTED_ORIGINAL_CAPACITY_EXCEEDED")
    return retained


def _bounded_exchanges(
    exchanges: tuple[SafeExchange, ...], incoming: SafeExchange
) -> tuple[SafeExchange, ...]:
    retained = (*exchanges, incoming)
    while (
        len(retained) > 24
        or sum(_exchange_bytes(item) for item in retained) > 128 * 1024
    ):
        if len(retained) == 1:
            raise MemoryError("SAFE_HISTORY_CAPACITY_EXCEEDED")
        retained = retained[1:]
    return retained


def _tokens(turn: SanitizedTurn, span: SourceSpan) -> tuple[str, ...]:
    if span.token_end > len(turn.tokens):
        raise MemoryError("INVALID_SOURCE_SPAN")
    return tuple(x.symbol for x in turn.tokens[span.token_start : span.token_end])


def _unit(turn: SanitizedTurn, span: SourceSpan) -> tuple[str, ...]:
    if span.token_end > len(turn.tokens):
        raise MemoryError("INVALID_SOURCE_SPAN")
    if span not in assertion_unit_ranges(turn):
        raise MemoryError("SOURCE_SPAN_MUST_BE_FULL_ASSERTION_UNIT")
    return _tokens(turn, span)


def _proof(turn: SanitizedTurn, span: SourceSpan) -> OriginalProof:
    start, end = turn.token_span(span.token_start, span.token_end)
    return OriginalProof(turn.turn_ref, start, end, turn.original[start:end])


def _personal_operation(symbols: tuple[str, ...]):
    if {
        "NOT",
        "DO_NOT",
        "QUOTE_START",
        "QUOTE_END",
        "IF",
        "WHEN",
        "UNLESS",
        "THIRD_PARTY",
        "BUT",
        "THEIR",
    }.intersection(symbols):
        raise MemoryError("NONPERSONAL_ASSERTION")
    if not {"SELF", "MY", "OUR"}.intersection(symbols):
        raise MemoryError("PERSONAL_MARKER_REQUIRED")


def _unit_ends_question(turn: SanitizedTurn, span: SourceSpan) -> bool:
    return (
        span.token_end < len(turn.tokens)
        and turn.tokens[span.token_end].symbol == "QUESTION"
    )


def _question_comma(tokens: tuple[CanonicalToken, ...], start: int, index: int) -> bool:
    """A comma that joins a question clause to a statement clause."""
    if tokens[index].symbol != "COMMA":
        return False
    following = tokens[index + 1].symbol if index + 1 < len(tokens) else None
    opening = tokens[start].symbol if start < index else None
    return following in _QUESTION_OPENERS or opening in _QUESTION_OPENERS


def assertion_unit_ranges(turn: SanitizedTurn) -> tuple[SourceSpan, ...]:
    """Public, safe canonical ranges - no original text or value is exposed."""
    ranges: list[SourceSpan] = []
    start = 0
    for index, token in enumerate(turn.tokens):
        if token.symbol in _BOUNDARIES or _question_comma(turn.tokens, start, index):
            if start < index:
                ranges.append(SourceSpan(token_start=start, token_end=index))
            start = index + 1
    if start < len(turn.tokens):
        ranges.append(SourceSpan(token_start=start, token_end=len(turn.tokens)))
    return tuple(ranges)


# Canonical no-increase clauses, accepted as a trailing suffix of any scope
# assertion and inline before a property adjunct of a personal refinance.
_NO_INCREASE_SUFFIXES = (
    ("WITHOUT", "EXTRA", "BORROWING"),
    ("WITHOUT", "INCREASE", "MY", "BORROWING"),
    ("WITHOUT", "INCREASE", "OUR", "BORROWING"),
    ("WITHOUT", "INCREASE", "MY", "MORTGAGE", "AMOUNT"),
    ("WITHOUT", "INCREASE", "OUR", "MORTGAGE", "AMOUNT"),
    ("WITHOUT", "INCREASE", "MY", "LOAN", "AMOUNT"),
    ("WITHOUT", "INCREASE", "OUR", "LOAN", "AMOUNT"),
)
_INLINE_NO_INCREASE = (("WITHOUT", "BORROWING", "MORE"), *_NO_INCREASE_SUFFIXES)


def _strip_prefix_suffix(symbols: tuple[str, ...]) -> ScopeParts:
    start = 0
    if symbols[:2] == ("ACTUALLY", "COMMA"):
        start = 2
    if symbols[start : start + 1] == ("PLEASE",):
        start += 1
    end = len(symbols)
    no_increase_suffix = False
    for suffix in _NO_INCREASE_SUFFIXES:
        if symbols[end - len(suffix) : end] == suffix:
            end -= len(suffix)
            no_increase_suffix = True
            break
    return ScopeParts(symbols[start:end], no_increase_suffix)


def _target(tokens: tuple[str, ...], index: int, possessive: str) -> int | None:
    if index >= len(tokens) or tokens[index] != possessive:
        return None
    index += 1
    if index < len(tokens) and tokens[index] in {"EXISTING", "CURRENT"}:
        index += 1
    return (
        index + 1
        if index < len(tokens) and tokens[index] in {"MORTGAGE", "LOAN"}
        else None
    )


def parse_scope_unit(turn: SanitizedTurn, span: SourceSpan) -> dict[str, str]:
    """Fully consume one canonical assertion and emit its proved predicates."""
    symbols = _unit(turn, span)
    if _unit_ends_question(turn, span) or {
        "QUOTE_START",
        "QUOTE_END",
        "IF",
        "WHEN",
        "UNLESS",
        "MAYBE",
        "OR",
        "BUT",
    }.intersection(symbols):
        return {}
    parts = _strip_prefix_suffix(symbols)
    core = parts.core
    result: dict[str, str] = {}
    desire_end = (
        2
        if len(core) >= 2
        and core[0] in {"SELF", "SELF_PLURAL"}
        and core[1] in {"WANT", "NEED"}
        else 3
        if len(core) >= 3
        and core[0] in {"SELF", "SELF_PLURAL"}
        and core[1:3] == ("WOULD", "LIKE")
        else None
    )
    if desire_end is not None and core[desire_end : desire_end + 2] == (
        "TO",
        "REFINANCE",
    ):
        target_end = _target(core, desire_end + 2, "MY") or _target(
            core, desire_end + 2, "OUR"
        )
        if target_end is not None:
            cursor = target_end
            result.update(goal="refinance", scope="own_existing")
            for phrase in _INLINE_NO_INCREASE:
                if core[cursor : cursor + len(phrase)] == phrase:
                    result["borrowing"] = "same"
                    cursor += len(phrase)
                    break
            if core[cursor : cursor + 2] == ("FOR", "MY"):
                cursor += 2
                parsed_property = _type_clause(core, cursor)
                if (
                    parsed_property is not None
                    and len(parsed_property[0]) == 1
                    and parsed_property[1] == len(core)
                ):
                    cursor = parsed_property[1]
            if cursor == len(core):
                # The stripped suffix ("without extra borrowing", ...) still proves
                # no increase; returning early must not drop it.
                if parts.no_increase_suffix:
                    result["borrowing"] = "same"
                return result
            return {}
    # Standalone ownership: P HAVE [A] EXIST MORT.
    if len(core) in {4, 5} and core[0] in {"SELF", "SELF_PLURAL"} and core[1] == "HAVE":
        own_start = 3 if len(core) == 5 and core[2] == "A" else 2
        if core[own_start:] in (
            ("EXISTING", "MORTGAGE"),
            ("CURRENT", "MORTGAGE"),
            ("EXISTING", "LOAN"),
            ("CURRENT", "LOAN"),
        ):
            result["scope"] = "own_existing"
    # Information is intentionally distinct from refinance intent.
    if desire_end is not None and core[desire_end:] in {
        ("INFORMATION", "ABOUT", "REFINANCE"),
        ("INFORMATION", "ABOUT", "MORTGAGE"),
        ("INFORMATION", "ABOUT", "BORROWING"),
    }:
        result["goal"] = "information"
    for pronoun in ("SELF", "SELF_PLURAL"):
        if (
            core[:3] == (pronoun, "DO_NOT", "WANT")
            or core[:4]
            == (
                pronoun,
                "WILL",
                "NOT",
                "INCREASE",
            )
            or core[:4] == (pronoun, "WANT", "NOT", "TO")
        ):
            rest = core[3:]
            if rest[:1] == ("TO",):
                rest = rest[1:]
            if (
                rest
                and rest[0] == "REFINANCE"
                and (
                    _target(rest, 1, "MY") == len(rest)
                    or _target(rest, 1, "OUR") == len(rest)
                )
            ):
                result["goal"] = "none"
            if rest in {
                ("INCREASE", "MY", "BORROWING"),
                ("INCREASE", "OUR", "BORROWING"),
                ("BORROWING", "MORE"),
            }:
                result["borrowing"] = "same"
        if (
            len(core) >= 4
            and core[0] == pronoun
            and desire_end is not None
            and core[desire_end : desire_end + 1] == ("TO",)
        ):
            verb = core[desire_end + 1 :]
            if verb[:1] == ("REFINANCE",):
                if _target(verb, 1, "MY") == len(verb) or _target(
                    verb, 1, "OUR"
                ) == len(verb):
                    result.update(goal="refinance", scope="own_existing")
                elif _target(verb, 1, "THEIR") == len(verb) or verb[1:] in {
                    ("MY", "THIRD_PARTY", "MORTGAGE"),
                    ("MY", "THIRD_PARTY", "LOAN"),
                    ("OUR", "THIRD_PARTY", "MORTGAGE"),
                    ("OUR", "THIRD_PARTY", "LOAN"),
                }:
                    result["scope"] = "other"
            if verb[:1] == ("INCREASE",) and verb[1:] in (
                ("MY", "BORROWING"),
                ("OUR", "BORROWING"),
                ("MY", "MORTGAGE", "AMOUNT"),
                ("OUR", "MORTGAGE", "AMOUNT"),
            ):
                result["borrowing"] = "increase"
            if verb[:2] == ("BORROWING", "MORE"):
                result["borrowing"] = "increase"
            if verb in (
                ("KEEP", "MY", "MORTGAGE", "AMOUNT", "THE", "SAME"),
                ("KEEP", "OUR", "MORTGAGE", "AMOUNT", "THE", "SAME"),
                ("KEEP", "THE", "SAME", "MORTGAGE", "AMOUNT"),
            ):
                result["borrowing"] = "same"
        if desire_end is not None and core[0] == pronoun:
            rate = core[desire_end:]
            if rate[:1] == ("A",):
                rate = rate[1:]
            if rate[:3] in (("BETTER", "RATE", "ON"), ("BETTER", "RATE", "FOR")) and (
                _target(rate, 3, "MY") == len(rate)
                or _target(rate, 3, "OUR") == len(rate)
            ):
                result.update(goal="refinance", scope="own_existing")
        if core[:4] == (pronoun, "AM", "LOOKING", "FOR"):
            rate = core[4:]
            if rate[:1] == ("A",):
                rate = rate[1:]
            if rate[:3] in (("BETTER", "RATE", "ON"), ("BETTER", "RATE", "FOR")) and (
                _target(rate, 3, "MY") == len(rate)
                or _target(rate, 3, "OUR") == len(rate)
            ):
                result.update(goal="refinance", scope="own_existing")
        if (
            len(core) >= 3
            and core[0] == pronoun
            and core[1] == "AM"
            and core[2] == "REFINANCE"
            and (
                _target(core, 3, "MY") == len(core)
                or _target(core, 3, "OUR") == len(core)
            )
        ):
            result.update(goal="refinance", scope="own_existing")
    if parts.no_increase_suffix and result.get("goal") == "refinance":
        result["borrowing"] = "same"
    return result


def scope_property_type(turn: SanitizedTurn, span: SourceSpan) -> str | None:
    """Return one owned-property adjunct only from a complete refinance assertion."""
    concepts = parse_scope_unit(turn, span)
    symbols = _unit(turn, span)
    if concepts.get("goal") != "refinance" or "FOR" not in symbols:
        return None
    index = symbols.index("FOR")
    if symbols[index + 1 : index + 2] != ("MY",):
        return None
    parsed = _type_clause(symbols, index + 2)
    if parsed is None or len(parsed[0]) != 1 or parsed[1] != len(symbols):
        return None
    return parsed[0][0]


def _candidates(
    symbols: tuple[str, ...], field: FieldName, pending: PendingQuestion | None
) -> list[str]:
    if field == "property_type":
        return []
    anchor = None
    result = []
    for symbol in symbols:
        if symbol in {"MORTGAGE", "LOAN", "PROPERTY"}:
            anchor = symbol
        elif symbol.startswith("VALUE(") and (
            anchor in _ROLE[field]
            or (anchor is None and pending is not None and pending.field == field)
        ):
            result.append(symbol[6:-1])
    return result


def _property_types(symbols: tuple[str, ...]) -> tuple[str, ...]:
    phrases = (
        ("terraced_house", ("TERRACED", "HOUSE")),
        ("holiday_home", ("HOLIDAY", "PROPERTY")),
        ("apartment", ("APARTMENT",)),
        ("house", ("HOUSE",)),
    )
    found: list[str] = []
    index = 0
    while index < len(symbols):
        match = next(
            (
                item
                for item in phrases
                if symbols[index : index + len(item[1])] == item[1]
            ),
            None,
        )
        if match:
            found.append(match[0])
            index += len(match[1])
        else:
            index += 1
    return tuple(found)


def _bare_type_shape(symbols: tuple[str, ...]) -> bool:
    index = count = 0
    phrases = (
        ("TERRACED", "HOUSE"),
        ("HOLIDAY", "PROPERTY"),
        ("APARTMENT",),
        ("HOUSE",),
    )
    while index < len(symbols):
        if symbols[index : index + 1] == ("A",):
            index += 1
        phrase = next(
            (item for item in phrases if symbols[index : index + len(item)] == item),
            None,
        )
        if phrase is None:
            return False
        index += len(phrase)
        count += 1
        if index == len(symbols):
            return True
        if symbols[index] != "OR":
            return False
        index += 1
        if index == len(symbols):
            return False
    return count > 0


def observe_scope_restrictions(
    state: Memory, turn: SanitizedTurn
) -> dict[str, ScopeObservation]:
    """Find source-proved blockers without trusting the model proposal."""
    observed: dict[str, list[tuple[str, OriginalProof]]] = {}
    for span in assertion_unit_ranges(turn):
        symbols = _unit(turn, span)
        if (
            _unit_ends_question(turn, span)
            or symbols[:1]
            in {
                ("WHAT",),
                ("HOW",),
                ("CAN",),
                ("IS",),
            }
            or {
                "QUOTE_START",
                "QUOTE_END",
                "IF",
                "WHEN",
                "UNLESS",
            }.intersection(symbols)
        ):
            continue
        parsed = parse_scope_unit(turn, span)
        if "THIRD_PARTY" in symbols and parsed.get("scope") != "other":
            continue
        proof = _proof(turn, span)
        for name, value in parsed.items():
            if (name, value) in {
                ("goal", "none"),
                ("scope", "other"),
                ("borrowing", "increase"),
            }:
                observed.setdefault(name, []).append((value, proof))
        if not parsed:
            if "REFINANCE" in symbols:
                observed.setdefault("goal", []).append(("unknown", proof))
            if {"THEIR", "THIRD_PARTY"}.intersection(
                symbols
            ) and "REFINANCE" in symbols:
                observed.setdefault("scope", []).append(("unknown", proof))
            if {"INCREASE", "EXTRA", "BORROWING"}.intersection(symbols):
                observed.setdefault("borrowing", []).append(("unknown", proof))
    result: dict[str, ScopeObservation] = {}
    for name, values in observed.items():
        unique = {value for value, _ in values}
        value = next(iter(unique)) if len(unique) == 1 else "unknown"
        result[name] = ScopeObservation(value, tuple(proof for _, proof in values))
    return result


def observe_fact_restrictions(
    turn: SanitizedTurn,
) -> dict[FieldName, SourceFactRestriction]:
    """Observe only explicit personal labelled value assertions that deny certainty."""
    found: dict[FieldName, SourceFactRestriction] = {}
    for span in assertion_unit_ranges(turn):
        symbols = _unit(turn, span)
        if (
            _unit_ends_question(turn, span)
            or symbols[:1] in {("WHAT",), ("HOW",), ("CAN",), ("IS",)}
            or {
                "QUOTE_START",
                "QUOTE_END",
                "IF",
                "WHEN",
                "UNLESS",
                "THIRD_PARTY",
                "THEIR",
            }.intersection(symbols)
        ):
            continue
        if not {"SELF", "MY", "OUR"}.intersection(symbols) or not any(
            symbol.startswith("VALUE(") for symbol in symbols
        ):
            continue
        fields = _labelled_value_fields(symbols)
        if {"MAYBE", "OR"}.intersection(symbols):
            reason = "uncertain_value" if "MAYBE" in symbols else "conflicting_values"
            restricted = fields
        else:
            clauses = _independent_labelled_clauses(symbols)
            restricted = {
                field
                for clause in clauses
                if {"NOT", "DO_NOT"}.intersection(clause)
                for field in _labelled_value_fields(clause)
            }
            reason = "negated_value"
        if reason:
            for name in restricted:
                found[name] = SourceFactRestriction(name, (_proof(turn, span),), reason)
    return found


def _labelled_value_fields(symbols: tuple[str, ...]) -> set[FieldName]:
    if not any(symbol.startswith("VALUE(") for symbol in symbols):
        return set()
    fields: set[FieldName] = set()
    if {"MORTGAGE", "LOAN"}.intersection(symbols):
        fields.add("existing_loan_amount")
    if "PROPERTY" in symbols:
        fields.add("property_value")
    return fields


def _independent_labelled_clauses(
    symbols: tuple[str, ...],
) -> tuple[tuple[str, ...], ...]:
    clauses: list[tuple[str, ...]] = []
    start = 0
    for index, symbol in enumerate(symbols):
        if symbol in {"AND", "COMMA"}:
            if start < index:
                clauses.append(symbols[start:index])
            start = index + 1
    if start < len(symbols):
        clauses.append(symbols[start:])
    return tuple(clause for clause in clauses if _labelled_value_fields(clause))


def _value_clause(
    symbols: tuple[str, ...], start: int
) -> tuple[tuple[str, ...], int] | None:
    if start >= len(symbols) or not symbols[start].startswith("VALUE("):
        return None
    values = [symbols[start][6:-1]]
    index = start + 1
    while (
        index + 1 < len(symbols)
        and symbols[index] == "OR"
        and symbols[index + 1].startswith("VALUE(")
    ):
        values.append(symbols[index + 1][6:-1])
        index += 2
    if symbols[index : index + 1] == ("OR",):
        return None
    return tuple(values), index


def _type_clause(
    symbols: tuple[str, ...], start: int
) -> tuple[tuple[str, ...], int] | None:
    values: list[str] = []
    index = start
    while True:
        if symbols[index : index + 1] == ("A",):
            index += 1
        matched = next(
            (
                (name, tokens)
                for name, tokens in (
                    ("terraced_house", ("TERRACED", "HOUSE")),
                    ("holiday_home", ("HOLIDAY", "PROPERTY")),
                    ("apartment", ("APARTMENT",)),
                    ("house", ("HOUSE",)),
                )
                if symbols[index : index + len(tokens)] == tokens
            ),
            None,
        )
        if matched is None:
            return None
        values.append(matched[0])
        index += len(matched[1])
        if symbols[index : index + 1] != ("OR",):
            return tuple(values), index
        index += 1
        if index == len(symbols):
            return None


def _parse_financial_clause(
    symbols: tuple[str, ...], start: int, inherited: str | None
) -> tuple[FieldName, str, tuple[str, ...], tuple[str, ...], int, str] | None:
    index = start
    maybe = symbols[index : index + 1] == ("MAYBE",)
    if maybe:
        index += 1
    possessive = (
        symbols[index]
        if index < len(symbols) and symbols[index] in {"MY", "OUR"}
        else inherited
    )
    if possessive is None:
        return None
    if index < len(symbols) and symbols[index] in {"MY", "OUR"}:
        index += 1
    if symbols[index : index + 1] in {("EXISTING",), ("CURRENT",)}:
        index += 1
    if symbols[index : index + 1] in {("MORTGAGE",), ("LOAN",)}:
        index += 1
        if symbols[index : index + 1] == ("AMOUNT",):
            index += 1
        field: FieldName = "existing_loan_amount"
    elif symbols[index : index + 2] == ("PROPERTY", "VALUE_WORD"):
        index += 2
        field = "property_value"
    elif symbols[index : index + 1] == ("PROPERTY",):
        index += 1
        field = "property_type"
    else:
        return None
    if symbols[index : index + 1] in {("IS",), ("COLON",), ("OF",)}:
        index += 1
    else:
        return None
    negated = False
    for _ in range(2):
        if symbols[index : index + 1] == ("MAYBE",) and not maybe:
            maybe = True
            index += 1
        elif symbols[index : index + 1] in {("NOT",), ("DO_NOT",)} and not negated:
            negated = True
            index += 1
        else:
            break
    if symbols[index : index + 1] in {("MAYBE",), ("NOT",), ("DO_NOT",)}:
        return None
    if field == "property_type" and symbols[index : index + 1] == ("WORTH",):
        field = "property_value"
    if field == "property_type":
        parsed = _type_clause(symbols, index)
    else:
        if field == "property_value" and symbols[index : index + 1] == ("WORTH",):
            index += 1
        parsed = _value_clause(symbols, index)
    if parsed is None:
        return None
    candidates, end = parsed
    kind = (
        "uncertain"
        if maybe
        else "negated"
        if negated
        else "alternatives"
        if len(candidates) > 1
        else "affirmed"
    )
    return (
        field,
        kind,
        candidates if field != "property_type" else (),
        candidates if field == "property_type" else (),
        end,
        possessive,
    )


def analyze_financial_turn(state: Memory, turn: SanitizedTurn) -> FinancialAnalysis:
    del state
    clauses: list[FinancialClause] = []
    for unit_span in assertion_unit_ranges(turn):
        symbols = _unit(turn, unit_span)
        if (
            _unit_ends_question(turn, unit_span)
            or symbols[:1] in {("WHAT",), ("HOW",), ("CAN",), ("IS",)}
            or {
                "QUOTE_START",
                "QUOTE_END",
                "IF",
                "WHEN",
                "UNLESS",
                "THIRD_PARTY",
                "THEIR",
            }.intersection(symbols)
        ):
            continue
        start = 2 if symbols[:2] == ("ACTUALLY", "COMMA") else 0
        if symbols[start : start + 1] == ("PLEASE",):
            start += 1
        if symbols[start : start + 3] == ("CALCULATE", "LTV", "FOR"):
            start += 3
        if symbols[start : start + 2] in {
            ("SELF", "HAVE"),
            ("SELF_PLURAL", "HAVE"),
        }:
            type_start = start + 2
            parsed_type = _type_clause(symbols, type_start)
            if parsed_type is not None and parsed_type[1] == len(symbols):
                types, end = parsed_type
                clauses.append(
                    FinancialClause(
                        "property_type",
                        unit_span,
                        SourceSpan(
                            token_start=unit_span.token_start + start,
                            token_end=unit_span.token_start + end,
                        ),
                        "alternatives" if len(types) > 1 else "affirmed",
                        (),
                        types,
                        _proof(turn, unit_span),
                    )
                )
            continue
        inherited: str | None = None
        buffered: list[FinancialClause] = []
        malformed_fields: set[FieldName] = set()
        while start < len(symbols):
            parsed = _parse_financial_clause(symbols, start, inherited)
            if parsed is None:
                if buffered:
                    malformed_fields.update(clause.field for clause in buffered)
                tail = symbols[start:]
                if tail[:1] in {("MY",), ("OUR",)}:
                    if {"MORTGAGE", "LOAN"}.intersection(tail):
                        malformed_fields.add("existing_loan_amount")
                    if "PROPERTY" in tail:
                        malformed_fields.add(
                            "property_value"
                            if "VALUE_WORD" in tail or "WORTH" in tail
                            else "property_type"
                        )
                break
            field, kind, refs, enums, end, inherited = parsed
            buffered.append(
                FinancialClause(
                    field,
                    unit_span,
                    SourceSpan(
                        token_start=unit_span.token_start + start,
                        token_end=unit_span.token_start + end,
                    ),
                    kind,
                    refs,
                    enums,
                    _proof(turn, unit_span),
                )
            )
            if end == len(symbols):
                break
            if (
                field != "property_type"
                and symbols[end : end + 1] == ("AND",)
                and (shortened := _value_clause(symbols, end + 1)) is not None
            ):
                values, end = shortened
                clauses.append(
                    FinancialClause(
                        field,
                        unit_span,
                        SourceSpan(
                            token_start=unit_span.token_start + end - len(values),
                            token_end=unit_span.token_start + end,
                        ),
                        "alternatives" if len(values) > 1 else "affirmed",
                        values,
                        (),
                        _proof(turn, unit_span),
                    )
                )
                if end == len(symbols):
                    break
            if symbols[end : end + 1] not in {("AND",), ("COMMA",)}:
                malformed_fields.update(clause.field for clause in buffered)
                break
            start = end + 1
            if start == len(symbols):
                malformed_fields.update(clause.field for clause in buffered)
                break
        if malformed_fields:
            clauses.extend(
                replace(clause, kind="uncertain")
                for clause in buffered
                if clause.field in malformed_fields
            )
            clauses.extend(
                FinancialClause(
                    field,
                    unit_span,
                    unit_span,
                    "uncertain",
                    proof=_proof(turn, unit_span),
                )
                for field in malformed_fields
                if not any(clause.field == field for clause in buffered)
            )
        else:
            clauses.extend(buffered)
    conflicts: dict[FieldName, tuple[OriginalProof, ...]] = {}
    for field in ("existing_loan_amount", "property_value", "property_type"):
        affirmed = [
            clause
            for clause in clauses
            if clause.field == field and clause.kind == "affirmed"
        ]
        protected = {value.value_ref: value.value for value in turn.protected_values}
        values = {
            tuple(protected[value_ref] for value_ref in clause.value_refs)
            if clause.value_refs
            else clause.enum_values
            for clause in affirmed
        }
        if len(values) > 1:
            conflicts[field] = tuple(
                clause.proof for clause in affirmed if clause.proof
            )
    return FinancialAnalysis(tuple(clauses), conflicts)


def _analysis_restrictions(
    state: Memory, turn: SanitizedTurn, analysis: FinancialAnalysis, *, failed: bool
) -> dict[FieldName, SourceFactRestriction]:
    grouped = {
        field: tuple(clause for clause in analysis.clauses if clause.field == field)
        for field in ("existing_loan_amount", "property_value", "property_type")
    }
    result: dict[FieldName, SourceFactRestriction] = {}
    for asserted_field, clauses in grouped.items():
        if not clauses:
            continue
        proofs = tuple(clause.proof for clause in clauses if clause.proof)
        intrinsic = asserted_field in analysis.conflicts or any(
            clause.kind in {"negated", "uncertain"}
            or (failed and clause.kind == "alternatives")
            for clause in clauses
        )
        if intrinsic:
            result[asserted_field] = SourceFactRestriction(
                asserted_field, proofs, "conflicting_values"
            )
        elif failed:
            clause = clauses[0]
            prior = state.facts.get(asserted_field)
            current_values = {
                value.value_ref: value.value for value in turn.protected_values
            }
            previous_values = {
                value_ref: value.value
                for value_ref, value in state.protected_values.items()
            }
            same = (
                prior
                and prior.status == FactStatus.KNOWN
                and (
                    (
                        clause.value_refs
                        and prior.value_ref in previous_values
                        and any(
                            previous_values[prior.value_ref]
                            == current_values.get(value_ref)
                            for value_ref in clause.value_refs
                        )
                    )
                    or (clause.enum_values and prior.enum_value in clause.enum_values)
                )
            )
            if not same:
                result[asserted_field] = SourceFactRestriction(
                    asserted_field, proofs, "uncertain_value"
                )
    return result


def _validate_financial_coverage(
    analysis: FinancialAnalysis, proposal: TurnInterpretation
) -> None:
    for clause in analysis.clauses:
        if clause.kind not in {"affirmed", "alternatives"}:
            continue
        matching = [
            operation
            for operation in proposal.operations
            if operation.field == clause.field
            and operation.source_span == clause.unit_span
        ]
        if not matching or (clause.field in analysis.conflicts):
            raise MemoryError("INCOMPLETE_FACT_PROPOSAL")


def commit_restrictions(
    state: Memory,
    turn: SanitizedTurn,
    restrictions: dict[str, ScopeObservation],
    fact_restrictions: dict[FieldName, SourceFactRestriction] | None = None,
    *,
    client_message_id: str,
    rejected: bool = False,
    rejection_code: str | None = None,
    force: bool = False,
) -> StageResult:
    """Commit only independently observed restrictions and normal turn bookkeeping."""
    fact_restrictions = fact_restrictions or {}
    _check_turn_identity(state, turn, client_message_id)
    if not restrictions and not fact_restrictions and not force:
        return StageResult(state, (), (), rejected, rejection_code)
    scope_proofs = dict(state.scope_proofs)
    goal, scope, borrowing = state.goal, state.scope, state.borrowing
    for name, observation in restrictions.items():
        scope_proofs[name] = observation.proofs
        if name == "goal":
            goal = observation.value
        elif name == "scope":
            scope = observation.value
        else:
            borrowing = observation.value
    facts = dict(state.facts)
    for name, restriction in fact_restrictions.items():
        prior = facts.get(name)
        facts[name] = Fact(
            f"f_{turn.turn_ref}_{name}",
            name,
            (prior.field_revision if prior else 0) + 1,
            FactStatus.AMBIGUOUS,
            original_proof=restriction.proofs,
        )
    turns = _retained_turns(state, turn, facts, scope_proofs)
    exchanges = _bounded_exchanges(
        state.exchanges, SafeExchange(turn.turn_ref, turn.model_projection)
    )
    changed = bool(restrictions or fact_restrictions)
    updated = replace(
        state,
        revision=state.revision + int(changed),
        facts=facts,
        protected_values={
            key: value
            for key, value in state.protected_values.items()
            if any(fact.value_ref == key for fact in facts.values())
        },
        turns=turns,
        goal=goal,
        scope=scope,
        borrowing=borrowing,
        scope_proofs=scope_proofs,
        pending_question=None if changed else state.pending_question,
        offer=None if changed else state.offer,
        accepted_review=None if changed else state.accepted_review,
        exchanges=exchanges,
        accepted_turns=state.accepted_turns + 1,
        client_turns={**state.client_turns, client_message_id: turn.turn_ref},
    )
    return StageResult(
        updated,
        tuple(facts[name] for name in fact_restrictions),
        _invalidated(changed, fact_restrictions),
        rejected,
        rejection_code,
    )


def stage_uninterpreted_turn(
    state: Memory,
    turn: SanitizedTurn,
    *,
    client_message_id: str,
    rejection_code: Literal[
        "INTERPRETATION_FAILED", "INVALID_PROPOSAL", "INCOMPLETE_FACT_PROPOSAL"
    ],
) -> StageResult:
    """Fail closed after unavailable or malformed interpretation."""
    if rejection_code not in {
        "INTERPRETATION_FAILED",
        "INVALID_PROPOSAL",
        "INCOMPLETE_FACT_PROPOSAL",
    }:
        raise MemoryError("INVALID_REJECTION_CODE")
    _check_turn_identity(state, turn, client_message_id)
    analysis = analyze_financial_turn(state, turn)
    return commit_restrictions(
        state,
        turn,
        observe_scope_restrictions(state, turn),
        _analysis_restrictions(state, turn, analysis, failed=True),
        client_message_id=client_message_id,
        rejected=True,
        rejection_code=rejection_code,
        force=True,
    )


def _validate(
    turn: SanitizedTurn,
    proposal: TurnInterpretation,
    pending: PendingQuestion | None,
    analysis: FinancialAnalysis | None = None,
) -> dict[str, tuple[OriginalProof, ...]]:
    handles = {x.value_ref: x for x in turn.protected_values}
    all_symbols = turn.model_projection
    conditional = any(x in {"IF", "WHEN", "UNLESS"} for x in all_symbols)
    if conditional != proposal.conditional:
        raise MemoryError("CONDITIONAL_SCOPE_CONFLICT")
    if (
        "QUOTE_START" in all_symbols or "QUOTE_END" in all_symbols
    ) and proposal.subject != "quoted":
        raise MemoryError("QUOTE_SCOPE_CONFLICT")
    if "THIRD_PARTY" in all_symbols and proposal.subject == "self" and not conditional:
        raise MemoryError("SUBJECT_SCOPE_CONFLICT")
    if proposal.operations and (proposal.subject != "self" or conditional):
        raise MemoryError("UNPROVEN_PERSONAL_SCOPE")
    proofs: dict[str, tuple[OriginalProof, ...]] = {}
    for op in proposal.operations:
        clauses = (
            [
                clause
                for clause in analysis.clauses
                if clause.field == op.field
                and clause.unit_span == op.source_span
                and clause.kind in {"affirmed", "alternatives"}
            ]
            if analysis
            else []
        )
        if clauses:
            clause = clauses[0]
            if op.operation != clause.operation:
                raise MemoryError("INCOMPLETE_FACT_PROPOSAL")
            if clause.value_refs and (
                len(op.value_refs) != len(clause.value_refs)
                or set(op.value_refs) != set(clause.value_refs)
                or any(
                    value not in handles
                    or handles[value].owner_turn_ref != turn.turn_ref
                    for value in op.value_refs
                )
            ):
                raise MemoryError("HANDLE_OUTSIDE_SOURCE_SPAN")
            types = (op.enum_value,) if op.operation == "set" else tuple(op.enum_values)
            if clause.enum_values and (
                len(types) != len(clause.enum_values)
                or set(types) != set(clause.enum_values)
            ):
                raise MemoryError("ENUM_OUTSIDE_SOURCE_SPAN")
            continue
        symbols = _unit(turn, op.source_span)
        if _unit_ends_question(turn, op.source_span):
            raise MemoryError("QUESTION_CANNOT_SET_FACT")
        if op.operation == "set" and {"MAYBE", "OR"}.intersection(symbols):
            raise MemoryError("UNCERTAIN_OR_ALTERNATIVE_SET")
        bare_value_reply = (
            op.operation == "set"
            and op.field != "property_type"
            and not {"MORTGAGE", "LOAN", "PROPERTY"}.intersection(symbols)
        )
        bare_type_reply = op.field == "property_type" and _bare_type_shape(symbols)
        if not bare_value_reply and not bare_type_reply:
            _personal_operation(symbols)
        selected = [x for x in op.value_refs if f"VALUE({x})" in symbols]
        if len(selected) != len(op.value_refs) or len(selected) != len(set(selected)):
            raise MemoryError("HANDLE_OUTSIDE_SOURCE_SPAN")
        if any(
            x not in handles or handles[x].owner_turn_ref != turn.turn_ref
            for x in selected
        ):
            raise MemoryError("FOREIGN_HANDLE")
        if op.field == "property_type":
            types = _property_types(symbols)
            if op.operation == "set" and types != (op.enum_value,):
                raise MemoryError("ENUM_OUTSIDE_SOURCE_SPAN")
            if op.operation == "ambiguous" and (
                "OR" not in symbols or set(types) != set(op.enum_values)
            ):
                raise MemoryError("AMBIGUITY_REQUIRES_ALL_TYPE_CANDIDATES")
            if bare_type_reply and (
                pending is None or pending.field != "property_type"
            ):
                raise MemoryError("BARE_VALUE_REQUIRES_CURRENT_PENDING_QUESTION")
        candidates = _candidates(symbols, op.field, pending)
        if (
            op.operation == "set"
            and op.field != "property_type"
            and bare_value_reply
            and (
                pending is None
                or any(not symbol.startswith("VALUE(") for symbol in symbols)
            )
        ):
            raise MemoryError("BARE_VALUE_REQUIRES_CURRENT_PENDING_QUESTION")
        if (
            op.operation == "set"
            and op.field != "property_type"
            and candidates != op.value_refs
        ):
            raise MemoryError("SET_REQUIRES_ONE_ROLE_CANDIDATE")
        if (
            op.operation == "ambiguous"
            and op.field != "property_type"
            and ("OR" not in symbols or set(candidates) != set(op.value_refs))
        ):
            raise MemoryError("AMBIGUITY_REQUIRES_ALL_ROLE_CANDIDATES")
        if op.operation == "withdraw" and (
            "WITHDRAW" not in symbols
            or not _ROLE.get(op.field, frozenset({"PROPERTY"})).intersection(symbols)
        ):
            raise MemoryError("WITHDRAW_REQUIRES_FIELD_MARKER")
    changes = {
        "goal": proposal.goal,
        "scope": proposal.ownership,
        "borrowing": proposal.borrowing,
    }
    supplied = {span.token_start: span for span in proposal.scope_source_spans}
    if len(supplied) != len(proposal.scope_source_spans):
        raise MemoryError("DUPLICATE_SOURCE_SPAN")
    if any(_unit_ends_question(turn, span) for span in proposal.scope_source_spans):
        raise MemoryError("QUESTION_CANNOT_PROVE_SCOPE")
    parsed_supplied = {
        (span.token_start, span.token_end): parse_scope_unit(turn, span)
        for span in proposal.scope_source_spans
    }
    for parsed in parsed_supplied.values():
        if not parsed:
            raise MemoryError("SCOPE_PROOF_CONCEPT_MISSING")
    all_parsed = {
        (span.token_start, span.token_end): parse_scope_unit(turn, span)
        for span in assertion_unit_ranges(turn)
    }
    for name, value in changes.items():
        if value == "unchanged":
            continue
        if not proposal.scope_source_spans:
            raise MemoryError("SCOPE_REQUIRES_SOURCE_PROOF")
        matching = tuple(
            _proof(turn, span)
            for span in proposal.scope_source_spans
            if parsed_supplied[(span.token_start, span.token_end)].get(name) == value
        )
        if not matching:
            raise MemoryError("SCOPE_PROOF_CONCEPT_MISSING")
        if any(parsed.get(name) not in {None, value} for parsed in all_parsed.values()):
            raise MemoryError("CONFLICTING_SCOPE_ASSERTION")
        proofs[name] = matching
    changed_names = {name for name, value in changes.items() if value != "unchanged"}
    if any(
        not set(parsed).intersection(changed_names)
        for parsed in parsed_supplied.values()
    ):
        raise MemoryError("EXTRANEOUS_SCOPE_SOURCE_SPAN")
    return proofs


def _same(old: Fact | None, new: Fact) -> bool:
    return old is not None and (
        old.status,
        old.value_ref,
        old.enum_value,
        old.enum_values,
        old.original_proof,
    ) == (
        new.status,
        new.value_ref,
        new.enum_value,
        new.enum_values,
        new.original_proof,
    )


def valid_pending_question(state: Memory) -> PendingQuestion | None:
    """Return the pending question only while its original authority still holds."""
    pending = state.pending_question
    if not isinstance(pending, PendingQuestion):
        return None
    if pending.asked_revision != state.revision or (
        not any(
            pending.personal_basis in group for group in state.scope_proofs.values()
        )
        and not any(
            fact.status == FactStatus.KNOWN
            and fact.original_proof == pending.personal_basis
            for fact in state.facts.values()
        )
    ):
        return None
    return pending


def candidate_fact_operations(
    state: Memory, turn: SanitizedTurn, analysis: FinancialAnalysis
) -> tuple[FactOperation, ...]:
    """Server-generate source-valid fact candidates without accepting any proposal."""

    def candidate_operation(**data: object) -> FactOperation:
        try:
            return FactOperation(**data)
        except ValueError as error:
            raise MemoryError("INVALID_CANDIDATE_CATALOG") from error

    pending = valid_pending_question(state)
    blocked = {
        field
        for field in ("existing_loan_amount", "property_value", "property_type")
        if field in analysis.conflicts
        or any(
            clause.field == field and clause.kind in {"negated", "uncertain"}
            for clause in analysis.clauses
        )
    }
    proposed: list[FactOperation] = []
    for clause in analysis.clauses:
        if clause.field in blocked or clause.kind not in {"affirmed", "alternatives"}:
            continue
        data: dict[str, object] = {
            "field": clause.field,
            "operation": clause.operation,
            "source_span": clause.unit_span,
        }
        if clause.field == "property_type":
            if clause.operation == "set":
                data["enum_value"] = clause.enum_values[0]
            else:
                data["enum_values"] = list(clause.enum_values)
        else:
            data["value_refs"] = list(clause.value_refs)
        proposed.append(candidate_operation(**data))
    if pending is not None:
        for span in assertion_unit_ranges(turn):
            symbols = _unit(turn, span)
            if pending.field == "property_type" and _bare_type_shape(symbols):
                types = _property_types(symbols)
                data = {
                    "field": pending.field,
                    "operation": "set" if len(types) == 1 else "ambiguous",
                    "source_span": span,
                }
                if len(types) == 1:
                    data["enum_value"] = types[0]
                else:
                    data["enum_values"] = list(types)
                proposed.append(candidate_operation(**data))
            elif pending.field != "property_type":
                refs = _candidates(symbols, pending.field, pending)
                if len(refs) == 1 and all(
                    symbol.startswith("VALUE(") for symbol in symbols
                ):
                    proposed.append(
                        candidate_operation(
                            field=pending.field,
                            operation="set",
                            value_refs=refs,
                            source_span=span,
                        )
                    )
    for span in assertion_unit_ranges(turn):
        symbols = _unit(turn, span)
        if "WITHDRAW" not in symbols:
            continue
        for candidate_field in (
            "existing_loan_amount",
            "property_value",
            "property_type",
        ):
            if _ROLE.get(candidate_field, frozenset({"PROPERTY"})).intersection(
                symbols
            ):
                proposed.append(
                    candidate_operation(
                        field=candidate_field, operation="withdraw", source_span=span
                    )
                )
    accepted: list[FactOperation] = []
    for operation in proposed:
        proposal = TurnInterpretation(
            subject="self",
            conditional=False,
            operations=[operation],
            goal="unchanged",
            borrowing="unchanged",
            ownership="unchanged",
            requested_action="acknowledge",
            unresolved=[],
        )
        try:
            _validate(turn, proposal, pending, analysis)
        except MemoryError:
            continue
        if operation not in accepted:
            accepted.append(operation)
    return tuple(accepted)


def stage_turn(
    current_state: Memory,
    sanitized_turn: SanitizedTurn,
    interpretation: TurnInterpretation,
    now: datetime | None = None,
    *,
    client_message_id: str | None = None,
) -> StageResult:
    del now
    client_message_id = client_message_id or sanitized_turn.turn_ref
    _check_turn_identity(current_state, sanitized_turn, client_message_id)
    restrictions = observe_scope_restrictions(current_state, sanitized_turn)
    pending = valid_pending_question(current_state)
    analysis = analyze_financial_turn(current_state, sanitized_turn)
    fact_restrictions = _analysis_restrictions(
        current_state, sanitized_turn, analysis, failed=False
    )
    try:
        proofs = _validate(sanitized_turn, interpretation, pending, analysis)
        _validate_financial_coverage(analysis, interpretation)
    except MemoryError as error:
        withheld = {
            **fact_restrictions,
            **_analysis_restrictions(
                current_state, sanitized_turn, analysis, failed=True
            ),
        }
        if restrictions or withheld:
            return commit_restrictions(
                current_state,
                sanitized_turn,
                restrictions,
                withheld,
                client_message_id=client_message_id,
                rejected=True,
                rejection_code=(
                    "INCOMPLETE_FACT_PROPOSAL"
                    if str(error) == "INCOMPLETE_FACT_PROPOSAL"
                    else "INVALID_PROPOSAL"
                ),
            )
        raise
    values = {
        **current_state.protected_values,
        **{x.value_ref: x for x in sanitized_turn.protected_values},
    }
    facts = dict(current_state.facts)
    changes = []
    for op in interpretation.operations:
        prior = facts.get(op.field)
        status = {
            "set": FactStatus.KNOWN,
            "ambiguous": FactStatus.AMBIGUOUS,
            "withdraw": FactStatus.WITHDRAWN,
        }[op.operation]
        original = (
            tuple(values[x].original_proof for x in op.value_refs)
            if op.operation == "ambiguous" and op.field != "property_type"
            else (
                _proof(sanitized_turn, op.source_span)
                if op.field == "property_type" and op.operation == "set"
                else (
                    values[op.value_refs[0]].original_proof
                    if op.operation == "set"
                    else _proof(sanitized_turn, op.source_span)
                )
            )
        )
        candidate = Fact(
            fact_id=f"f_{sanitized_turn.turn_ref}_{op.field}",
            field=op.field,
            field_revision=(prior.field_revision if prior else 0) + 1,
            status=status,
            value_ref=op.value_refs[0]
            if op.operation == "set" and op.field != "property_type"
            else None,
            enum_value=op.enum_value
            if op.field == "property_type" and op.operation == "set"
            else None,
            enum_values=tuple(
                value
                for value in ("house", "terraced_house", "apartment", "holiday_home")
                if value in op.enum_values
            ),
            original_proof=original,
        )
        if not _same(prior, candidate):
            facts[op.field] = candidate
            changes.append(candidate)
    for name, restriction in fact_restrictions.items():
        prior = facts.get(name)
        facts[name] = Fact(
            f"f_{sanitized_turn.turn_ref}_{name}",
            name,
            (prior.field_revision if prior else 0) + 1,
            FactStatus.AMBIGUOUS,
            original_proof=restriction.proofs,
        )
        changes = [change for change in changes if change.field != name]
        changes.append(facts[name])
    goal = (
        current_state.goal
        if interpretation.goal == "unchanged"
        else interpretation.goal
    )
    borrowing = (
        current_state.borrowing
        if interpretation.borrowing == "unchanged"
        else interpretation.borrowing
    )
    scope = (
        current_state.scope
        if interpretation.ownership == "unchanged"
        else interpretation.ownership
    )
    for name, observation in restrictions.items():
        proofs[name] = observation.proofs
        if name == "goal":
            goal = observation.value
        elif name == "scope":
            scope = observation.value
        else:
            borrowing = observation.value
    scope_reaffirmed = any(
        value != "unchanged"
        for value in (
            interpretation.goal,
            interpretation.borrowing,
            interpretation.ownership,
        )
    )
    material_scope = scope_reaffirmed or (goal, borrowing, scope) != (
        current_state.goal,
        current_state.borrowing,
        current_state.scope,
    )
    changed = bool(changes) or material_scope
    next_scope_proofs = dict(current_state.scope_proofs)
    next_scope_proofs.update(proofs)
    turns = _retained_turns(current_state, sanitized_turn, facts, next_scope_proofs)
    handles = {f.value_ref for f in facts.values() if f.value_ref}
    for fact in facts.values():
        if isinstance(fact.original_proof, tuple):
            handles.update(
                key
                for key, value in values.items()
                if value.original_proof in fact.original_proof
            )
    values = {k: v for k, v in values.items() if k in handles}
    exchanges = _bounded_exchanges(
        current_state.exchanges,
        SafeExchange(sanitized_turn.turn_ref, sanitized_turn.model_projection),
    )
    clients = dict(current_state.client_turns)
    clients[client_message_id] = sanitized_turn.turn_ref
    return StageResult(
        replace(
            current_state,
            revision=current_state.revision + int(changed),
            facts=facts,
            protected_values=values,
            turns=turns,
            goal=goal,
            scope=scope,
            borrowing=borrowing,
            # ``unresolved`` is advisory model output only. The response planner
            # commits a PendingQuestion separately after it chooses fixed text.
            pending_question=(
                None
                if (
                    isinstance(current_state.pending_question, PendingQuestion)
                    and any(
                        op.field == current_state.pending_question.field
                        for op in interpretation.operations
                    )
                )
                else current_state.pending_question
            ),
            last_operation=interpretation.requested_action,
            last_subject=interpretation.subject,
            last_conditional=interpretation.conditional,
            active_operation=(
                "calculate"
                if interpretation.requested_action == "calculate"
                else None
                if (
                    interpretation.requested_action == "review"
                    and interpretation.subject == "self"
                    and not interpretation.conditional
                )
                else current_state.active_operation
                if interpretation.requested_action == "acknowledge"
                else None
                if interpretation.requested_action in {"forget", "decline"}
                else current_state.active_operation
            ),
            offer=None if changed else current_state.offer,
            accepted_review=None if changed else current_state.accepted_review,
            exchanges=exchanges,
            accepted_turns=current_state.accepted_turns + 1,
            client_turns=clients,
            scope_proofs=next_scope_proofs,
            draft_requested=(
                True
                if (
                    interpretation.requested_action == "review"
                    and interpretation.subject == "self"
                    and not interpretation.conditional
                )
                else current_state.draft_requested
                if interpretation.requested_action == "acknowledge"
                else False
            ),
        ),
        tuple(changes),
        _invalidated(changed, (item.field for item in changes)),
    )


def choose_action(state: Memory, changed_fields: tuple[str, ...] = ()) -> str:
    if state.last_operation == "forget":
        return "forget"
    if state.last_operation == "decline":
        return "acknowledge"
    if state.last_operation == "knowledge":
        return "retrieve"
    if state.last_operation == "calculate" or (
        state.active_operation == "calculate"
        and state.last_operation == "acknowledge"
        and state.last_subject == "self"
        and not state.last_conditional
        and bool(set(changed_fields) & {"existing_loan_amount", "property_value"})
    ):
        required = ("existing_loan_amount", "property_value")
        return (
            "calculate"
            if all(
                state.facts.get(x, Fact("", x, 0, FactStatus.WITHDRAWN)).status
                == FactStatus.KNOWN
                for x in required
            )
            else "clarify"
        )
    return "clarify" if state.pending_question else "acknowledge"


def choose_pending_question(
    state: Memory, allowed_action: str
) -> PendingQuestion | None:
    """Select one deterministic missing fact after validated personal scope."""
    ownership_proofs = state.scope_proofs.get("scope", ())
    fact_proofs = tuple(
        fact.original_proof
        for fact in state.facts.values()
        if fact.status == FactStatus.KNOWN
        and isinstance(fact.original_proof, OriginalProof)
    )
    basis = ownership_proofs[0] if ownership_proofs else next(iter(fact_proofs), None)
    if basis is None or allowed_action in {
        "retrieve",
        "knowledge",
        "forget",
        "decline",
    }:
        return None
    if allowed_action == "calculate":
        fields = ("existing_loan_amount", "property_value")
    elif (
        allowed_action in {"acknowledge", "review", "clarify"}
        and state.goal == "refinance"
        and state.scope_proofs.get("goal")
    ):
        fields = ("existing_loan_amount", "property_value", "property_type")
    else:
        return None
    for candidate_field in fields:
        fact = state.facts.get(candidate_field)
        if fact is not None and fact.status == FactStatus.AMBIGUOUS:
            return PendingQuestion(
                candidate_field,
                state.revision,
                f"pending_{state.revision}_{candidate_field}",
                basis,
            )
    for missing_field in fields:
        fact = state.facts.get(missing_field)
        if fact is None or fact.status != FactStatus.KNOWN:
            return PendingQuestion(
                field=missing_field,
                asked_revision=state.revision,
                question_id=f"pending_{state.revision}_{missing_field}",
                personal_basis=basis,
            )
    return None


def build_context(
    state: Memory,
    limits: ContextLimits | None = None,
    retrieval_pack: tuple[tuple[str, ...], ...] = (),
    *,
    current_turn: SanitizedTurn | None = None,
) -> dict:
    limits = limits or ContextLimits()
    if (
        limits.input_tokens <= 0
        or limits.output_reservation <= 0
        or limits.mandatory_tokens < 0
        or (
            limits.model_context_tokens is not None and limits.model_context_tokens <= 0
        )
    ):
        raise MemoryError("CONTEXT_BUDGET_EXCEEDED")
    ceiling = limits.input_tokens
    if limits.model_context_tokens is not None:
        ceiling = min(ceiling, limits.model_context_tokens - limits.output_reservation)
    if ceiling <= 0:
        raise MemoryError("CONTEXT_BUDGET_EXCEEDED")
    mandatory = (
        "STATE",
        f"REVISION({state.revision})",
        f"GOAL({state.goal})",
        f"SCOPE({state.scope})",
        f"BORROWING({state.borrowing})",
        f"PENDING({state.pending_question.field if state.pending_question else 'none'})",
        f"OPERATION({state.last_operation or 'none'})",
        *(
            f"FACT({f.fact_id},{n},r{f.field_revision},{f.status})"
            for n, f in sorted(state.facts.items())
        ),
        *(current_turn.model_projection if current_turn else ()),
    )
    counter = limits.token_counter or (lambda xs: sum(len(x.split()) + 1 for x in xs))
    evidence = tuple(x for g in retrieval_pack for x in g)
    used = limits.mandatory_tokens + counter(mandatory) + counter(evidence)
    if used > ceiling:
        raise MemoryError("CONTEXT_BUDGET_EXCEEDED")
    history = []
    for exchange in reversed(state.exchanges):
        if current_turn is not None and exchange.turn_ref == current_turn.turn_ref:
            continue
        pair = exchange.tokens + exchange.response
        cost = counter(pair)
        if used + cost <= ceiling:
            history.append(pair)
            used += cost
    return {
        "token_estimate": used,
        "token_estimate_kind": "supplied_counter"
        if limits.token_counter
        else "conservative_symbol_estimate_not_tokenizer_count",
        "state": mandatory,
        "evidence": evidence,
        "history": tuple(reversed(history)),
    }


def replay_key(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def record_replay(
    state: Memory,
    message_id: str,
    payload: bytes,
    response: tuple[str, ...],
    outcome: str = "ACCEPTED",
) -> Memory:
    if message_id in state.forgotten_client_ids or message_id in state.forgotten_turns:
        raise MemoryError("FORGOTTEN_TURN")
    fingerprint = replay_key(payload)
    existing = state.replay.get(message_id)
    if existing:
        if existing.fingerprint != fingerprint:
            raise MemoryError("REPLAY_CONFLICT")
        return state
    if len(_identity_ids(state) | {message_id}) > 100:
        raise MemoryError("IDENTITY_CAPACITY_EXCEEDED")
    replay = {
        **state.replay,
        message_id: ReplayRecord(fingerprint, outcome, response),
    }
    while True:
        bodies = [key for key, record in replay.items() if record.response is not None]
        body_bytes = sum(_response_bytes(replay[key].response or ()) for key in bodies)
        if len(bodies) <= 16 and body_bytes <= 1024 * 1024:
            break
        oldest = bodies[0]
        record = replay[oldest]
        replay[oldest] = ReplayRecord(record.fingerprint, record.outcome, None)
    return replace(state, replay=replay)


def replay_response(state: Memory, message_id: str, payload: bytes) -> tuple[str, ...]:
    if message_id in state.forgotten_client_ids or message_id in state.forgotten_turns:
        raise MemoryError("FORGOTTEN_TURN")
    record = state.replay.get(message_id)
    if record is None or record.fingerprint != replay_key(payload):
        raise MemoryError("REPLAY_CONFLICT")
    if record.response is None:
        raise MemoryError("REPLAY_EXPIRED")
    return record.response


def forget(
    state: Memory, turn_refs: set[str] | None = None, *, clear_all: bool = False
) -> tuple[Memory, tuple[str, ...]]:
    affected = (
        (
            set(state.turns)
            | {x.turn_ref for x in state.exchanges}
            | set(state.client_turns.values())
        )
        if clear_all
        else set(turn_refs or ())
    )
    if not affected and not clear_all:
        raise MemoryError("AMBIGUOUS_FORGET_SCOPE")
    facts = {n: f for n, f in state.facts.items() if not _fact_uses_turn(f, affected)}
    scope_proofs = {
        name: tuple(proof for proof in proofs if proof.turn_ref not in affected)
        for name, proofs in state.scope_proofs.items()
    }
    goal = state.goal if scope_proofs.get("goal") else "unknown"
    scope = state.scope if scope_proofs.get("scope") else "unknown"
    borrowing = state.borrowing if scope_proofs.get("borrowing") else "unknown"
    if clear_all:
        goal = scope = borrowing = "unknown"
    redacted_client_ids = frozenset(
        k for k, v in state.client_turns.items() if v in affected
    )
    tombstones = state.forgotten_client_ids | redacted_client_ids
    replay = {
        message_id: ReplayRecord(record.fingerprint, record.outcome, None)
        for message_id, record in state.replay.items()
        if message_id not in redacted_client_ids
    }
    return replace(
        state,
        revision=state.revision + 1,
        facts=facts,
        goal=goal,
        scope=scope,
        borrowing=borrowing,
        protected_values={
            k: v
            for k, v in state.protected_values.items()
            if v.owner_turn_ref not in affected
        },
        turns={k: v for k, v in state.turns.items() if k not in affected},
        exchanges=(),
        replay=replay,
        offer=None,
        accepted_review=None,
        pending_question=None,
        last_operation=None,
        active_operation=None,
        forgotten_turns=state.forgotten_turns | frozenset(affected),
        forgotten_client_ids=tombstones,
        client_turns={k: v for k, v in state.client_turns.items() if v not in affected},
        scope_proofs=scope_proofs,
        draft_requested=False,
    ), tuple(sorted(redacted_client_ids))


def _fact_uses_turn(fact: Fact, affected: set[str]) -> bool:
    proofs = (
        fact.original_proof
        if isinstance(fact.original_proof, tuple)
        else (fact.original_proof,)
    )
    return any(x and x.turn_ref in affected for x in proofs)
