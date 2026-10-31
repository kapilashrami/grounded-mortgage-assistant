"""Additive, deliberately small wire types for the isolated memory backend."""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    model_validator,
)

from grounded_mortgage_assistant.state.memory import FieldName

MessageId = Annotated[
    str, Field(min_length=1, max_length=80, pattern=r"^[A-Za-z0-9_-]+$")
]


class MemoryChange(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    field: FieldName
    fact_id: str = Field(min_length=1)
    field_revision: int = Field(ge=1)
    status: Literal["known", "ambiguous", "withdrawn"]


class MemoryReceipt(BaseModel):
    """Safe summary only - no values, proofs, text, or client identifiers."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    revision: int = Field(ge=0)
    changes: list[MemoryChange] = Field(default_factory=list, max_length=3)
    invalidated: list[Literal["calculation", "offer", "review"]] = Field(
        default_factory=list, max_length=3
    )
    redacted_message_ids: list[str] = Field(default_factory=list, max_length=100)


class MemoryChatRequest(BaseModel):
    """The legacy chat request shape. page_context is accepted but never model input."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)
    session_id: MessageId
    message_id: MessageId
    message: str = Field(min_length=1, max_length=1000)
    page_context: str | None = Field(default=None, max_length=1000)


class ForgetScope(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    kind: Literal["all_chat", "messages"]
    message_ids: list[MessageId] | None = Field(
        default=None, max_length=100, min_length=1
    )

    @model_validator(mode="after")
    def complete_scope(self):
        if self.kind == "all_chat" and self.message_ids is not None:
            raise ValueError("all_chat has no message ids")
        if self.kind == "messages" and (
            self.message_ids is None
            or len(set(self.message_ids)) != len(self.message_ids)
        ):
            raise ValueError("messages needs unique message ids")
        return self


class ForgetRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    message_id: MessageId
    scope: ForgetScope


class ResetRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    message_id: MessageId
