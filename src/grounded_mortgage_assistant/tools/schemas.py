"""Wire schemas for LTV calculations and the explicit refinance draft."""

from datetime import datetime, timedelta
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class UserTurn(BaseModel):
    """An immutable original user message that owns an exact proof span."""

    model_config = ConfigDict(extra="forbid")
    message_id: str = Field(min_length=1, max_length=80)
    text: str = Field(min_length=1, max_length=1000)
    notation_language: Literal["sv", "en"] | None = None


class CalculationInputReference(BaseModel):
    model_config = ConfigDict(extra="forbid")
    role: Literal["loan_amount", "property_value"]
    message_id: str = Field(min_length=1)
    quote: str

    class Span(BaseModel):
        model_config = ConfigDict(extra="forbid")
        start: int = Field(ge=0)
        end: int = Field(gt=0)

        @model_validator(mode="after")
        def ordered(self):
            if self.end <= self.start:
                raise ValueError("span must be ordered")
            return self

    span: Span


class LtvCalculation(BaseModel):
    model_config = ConfigDict(extra="forbid")
    calculation_id: str
    tool: Literal["loan_to_value"]
    formula: Literal["loan_amount/property_value*100"]
    version: Literal["ltv-v1"]
    loan_amount_sek: str = Field(pattern=r"^(?:0|[1-9]\d*)(?:\.\d{1,2})?$")
    property_value_sek: str = Field(
        pattern=r"^(?:(?:[1-9]\d*)(?:\.\d{1,2})?|0\.(?:0[1-9]|[1-9]\d?))$"
    )
    percent: str = Field(pattern=r"^\d+\.\d{2}$")
    rounding: Literal["ROUND_HALF_UP, 2 decimal places"]
    assumptions: str
    input_references: list[CalculationInputReference] = Field(
        min_length=2, max_length=2
    )

    @model_validator(mode="after")
    def references_have_roles(self):
        if {item.role for item in self.input_references} != {
            "loan_amount",
            "property_value",
        }:
            raise ValueError("calculation requires both input roles")
        if (
            len(
                {
                    (item.message_id, item.span.start, item.span.end)
                    for item in self.input_references
                }
            )
            != 2
        ):
            raise ValueError("calculation references must be distinct")
        return self


MortgageFieldName = Literal["existing_loan_amount", "property_value", "property_type"]
MortgageIntentName = Literal["personal_refinance", "no_increase"]
MortgageIssue = Literal[
    "missing_supported_fields",
    "missing_intent",
    "invalid_field",
    "ambiguous_field",
    "unsupported_intent",
]


class MortgageProvenance(BaseModel):
    model_config = ConfigDict(extra="forbid")
    message_id: str = Field(min_length=1, max_length=80)
    quote: str = Field(min_length=1, max_length=1000)
    start: int = Field(ge=0, le=1000, strict=True)
    end: int = Field(ge=0, le=1000, strict=True)

    @model_validator(mode="after")
    def span_matches_quote(self):
        if self.end <= self.start or len(self.quote) != self.end - self.start:
            raise ValueError("provenance span must match quote length")
        return self


class MortgageDraft(BaseModel):
    """Server-owned, deliberately small refinance form proposal."""

    model_config = ConfigDict(extra="forbid")
    schema_version: Literal["mortgage-draft-v1"] = "mortgage-draft-v1"
    draft_id: str = Field(min_length=1, max_length=80)
    revision: int = Field(ge=1, strict=True)
    base_revision: int = Field(ge=0, strict=True)
    created_at: str
    expires_at: str
    intent: Literal["refinance_without_increase"] | None
    fields: dict[MortgageFieldName, int | str] = Field(default_factory=dict)
    field_provenance: dict[MortgageFieldName, MortgageProvenance] = Field(
        default_factory=dict
    )
    intent_provenance: dict[MortgageIntentName, MortgageProvenance] = Field(
        default_factory=dict
    )
    missing_fields: list[MortgageFieldName]
    issues: list[MortgageIssue] = Field(default_factory=list, max_length=5)
    ready: bool

    @model_validator(mode="after")
    def validate_supported_fields(self):
        if self.base_revision >= self.revision:
            raise ValueError("base_revision must be less than revision")
        try:
            created = datetime.fromisoformat(self.created_at)
            expires = datetime.fromisoformat(self.expires_at)
        except ValueError as exc:
            raise ValueError("timestamps must be ISO timestamps") from exc
        if created.tzinfo is None or expires.tzinfo is None:
            raise ValueError("timestamps must include a timezone")
        lifetime = expires - created
        if lifetime <= timedelta(0) or lifetime > timedelta(minutes=30):
            raise ValueError("expiry must follow creation by at most 30 minutes")
        for key, value in self.fields.items():
            if key in {"existing_loan_amount", "property_value"}:
                low, high = (
                    (150000, 10000000)
                    if key == "existing_loan_amount"
                    else (176471, 20000000)
                )
                if type(value) is not int or not low <= value <= high:
                    raise ValueError(f"invalid {key}")
            elif value not in {"house", "terraced_house", "apartment", "holiday_home"}:
                raise ValueError("invalid property type")
        if set(self.field_provenance) != set(self.fields):
            raise ValueError("field provenance must match present fields")
        has_intent_proofs = set(self.intent_provenance) == {
            "personal_refinance",
            "no_increase",
        }
        if (self.intent is not None) != has_intent_proofs:
            raise ValueError(
                "completed intent and both intent proofs must be present together"
            )
        if len(set(self.missing_fields)) != len(self.missing_fields):
            raise ValueError("missing fields must be unique")
        expected = {"existing_loan_amount", "property_value", "property_type"} - set(
            self.fields
        )
        if set(self.missing_fields) != expected:
            raise ValueError("missing fields must match supported fields")
        if len(set(self.issues)) != len(self.issues):
            raise ValueError("issues must be unique")
        complete = not expected and self.intent is not None and has_intent_proofs
        if self.ready != (complete and not self.issues):
            raise ValueError(
                "ready must reflect complete supported fields, intent proofs, and no issues"
            )
        return self


class MortgageDraftState(BaseModel):
    model_config = ConfigDict(extra="forbid")
    draft: MortgageDraft
    source_turns: list[UserTurn] = Field(max_length=5)

    @model_validator(mode="after")
    def proofs_match_sources(self):
        by_id = {turn.message_id: turn for turn in self.source_turns}
        if len(by_id) != len(self.source_turns):
            raise ValueError("source turn IDs must be unique")
        proofs = [
            *self.draft.field_provenance.values(),
            *self.draft.intent_provenance.values(),
        ]
        for proof in proofs:
            turn = by_id.get(proof.message_id)
            if turn is None or turn.text[proof.start : proof.end] != proof.quote:
                raise ValueError("provenance must match an original source turn span")
        return self
