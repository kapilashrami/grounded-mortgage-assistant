"""Native-tool interpretation boundary. It never commits memory."""

from __future__ import annotations

import json
import os
import re
import time
from collections.abc import Mapping
from dataclasses import dataclass
from dataclasses import field as dataclass_field
from hashlib import sha256
from typing import Any

import httpx
from langchain_openai import ChatOpenAI
from langsmith.run_helpers import tracing_context
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from grounded_mortgage_assistant.llm.interpretation_candidates import (
    CandidateCatalog,
    CandidateSelectionError,
    TurnSelection,
    build_candidate_catalog,
    expand_selection,
    knowledge_query_is_eligible,
)
from grounded_mortgage_assistant.llm.model_context import (
    ModelContextError,
    pack_model_context,
    token_count,
)
from grounded_mortgage_assistant.state.memory import (
    Memory,
    MemoryError,
    SanitizedTurn,
    TurnInterpretation,
)
from grounded_mortgage_assistant.tracing import trace_event

LOCAL_MODEL = "Qwen3.5-9B-MLX-4bit-938d891"
LOCAL_URL = "http://127.0.0.1:8011/v1"
OPENAI_URL = "https://api.openai.com/v1"
DEFAULT_OPENAI_MODEL = "gpt-4.1"
# Chat Completions rejects function tools with any other effort for GPT-5.4 models
# ("use /v1/responses or set reasoning_effort to 'none'"), checked 31 October 2026.
REASONING_EFFORTS = frozenset({"none"})
OUTPUT_TOKENS = 1800
MAX_PROVIDER_RESPONSE_BYTES = 1_000_000
SchemaDiagnostic = tuple[str, str]

_INTERPRETATION_FIELDS = frozenset(
    {
        "subject",
        "conditional",
        "requested_action",
        "fact_candidate_ids",
        "scope_candidate_ids",
    }
)
_NORMALIZED_SCHEMA_CODES = frozenset(
    {
        "bool_type",
        "duplicate_fact_candidate_ids",
        "duplicate_scope_candidate_ids",
        "extra_forbidden",
        "greater_than",
        "greater_than_equal",
        "list_type",
        "literal_error",
        "missing",
        "model_type",
        "string_type",
        "too_long",
        "value_error",
    }
)
_DIAGNOSTIC_FIELDS = {
    "duplicate_fact_candidate_ids": "fact_candidate_ids",
    "duplicate_scope_candidate_ids": "scope_candidate_ids",
}


def _schema_diagnostics(error: ValidationError) -> tuple[SchemaDiagnostic, ...]:
    """Return only fixed schema paths and normalized validation codes."""
    diagnostics: list[SchemaDiagnostic] = []
    for item in error.errors(
        include_input=False, include_context=False, include_url=False
    ):
        location = item.get("loc", ())
        code = item.get("type")
        field = location[0] if location else _DIAGNOSTIC_FIELDS.get(code)
        path = str(field) if field in _INTERPRETATION_FIELDS else "$"
        normalized = code if code in _NORMALIZED_SCHEMA_CODES else "schema_error"
        diagnostic = (path, normalized)
        if diagnostic not in diagnostics:
            diagnostics.append(diagnostic)
    return tuple(diagnostics)


class InterpretationError(RuntimeError):
    def __init__(
        self, category: str, *, schema_diagnostics: tuple[SchemaDiagnostic, ...] = ()
    ):
        self.category = category
        self.schema_diagnostics = schema_diagnostics
        super().__init__(category)


class RejectSyncTransport(httpx.BaseTransport):
    def handle_request(self, request: httpx.Request) -> httpx.Response:
        raise InterpretationError("SYNC_REQUEST_REJECTED")


@dataclass(frozen=True)
class InterpretationResult:
    interpretation: TurnInterpretation
    call_id: str
    model: str
    input_tokens: int
    output_tokens: int | None
    provider_input_tokens: int
    elapsed_ms: int
    prompt_hash: str
    schema_hash: str


class EvidenceSelection(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)
    evidence_ids: list[str] = Field(max_length=5)
    needs_clarification: bool


def interpretation_tool(catalog: CandidateCatalog) -> dict[str, Any]:
    schema = TurnSelection.model_json_schema()
    for field, candidate_ids, maximum in (
        ("fact_candidate_ids", tuple(catalog.fact_candidates), 3),
        ("scope_candidate_ids", tuple(catalog.scope_candidates), 4),
    ):
        property_schema = schema["properties"][field]
        items = property_schema["items"]
        property_schema["uniqueItems"] = True
        property_schema["maxItems"] = min(maximum, len(candidate_ids))
        if candidate_ids:
            items["enum"] = list(candidate_ids)
    return {
        "type": "function",
        "function": {
            "name": "interpret_turn",
            "description": "Return the strictly validated turn interpretation.",
            "parameters": schema,
        },
    }


@dataclass(frozen=True)
class Provider:
    """One chat-completions endpoint. The selection is fixed at startup."""

    name: str
    model: str
    base_url: str
    api_key: str = dataclass_field(repr=False)
    extra_body: dict[str, Any] | None = None
    # GPT-5 reasoning models reject temperature=0; they take a reasoning effort.
    temperature: float | None = 0
    reasoning_effort: str | None = None

    def owns(self, model: Any) -> bool:
        # OpenAI answers with a dated snapshot of the requested alias.
        return isinstance(model, str) and (
            model == self.model
            or (
                self.name == "openai"
                and re.fullmatch(
                    rf"{re.escape(self.model)}-\d{{4}}-\d{{2}}-\d{{2}}", model
                )
                is not None
            )
        )

    def finished(self, reason: Any) -> bool:
        # OpenAI reports "stop" when tool_choice forces a named function.
        return reason == "tool_calls" or (self.name == "openai" and reason == "stop")


LOCAL_PROVIDER = Provider(
    "local",
    LOCAL_MODEL,
    LOCAL_URL,
    "local",
    {"chat_template_kwargs": {"enable_thinking": False}},
)


def provider_from_env(env: Mapping[str, str] = os.environ) -> Provider:
    """Local Qwen unless LLM_PROVIDER=openai. There is no fallback."""
    name = env.get("LLM_PROVIDER", "local")
    if name == "local":
        return LOCAL_PROVIDER
    if name != "openai":
        raise RuntimeError("LLM_PROVIDER must be local or openai")
    key = env.get("OPENAI_API_KEY", "").strip()
    if not key:
        raise RuntimeError("OPENAI_API_KEY is required for the openai provider")
    model = env.get("OPENAI_MODEL", DEFAULT_OPENAI_MODEL)
    effort = env.get("OPENAI_REASONING_EFFORT")
    if not model.startswith("gpt-5"):
        if effort is not None:
            raise RuntimeError("OPENAI_REASONING_EFFORT applies to gpt-5 models")
        return Provider("openai", model, OPENAI_URL, key)
    if effort not in REASONING_EFFORTS:
        raise RuntimeError(
            "OPENAI_REASONING_EFFORT must be one of "
            + ", ".join(sorted(REASONING_EFFORTS))
        )
    return Provider(
        "openai", model, OPENAI_URL, key, temperature=None, reasoning_effort=effort
    )


class GuardedTransport(httpx.AsyncBaseTransport):
    def __init__(
        self,
        transport: httpx.AsyncBaseTransport,
        tokenizer: Any,
        capacity: int,
        tool: dict[str, Any],
        provider: Provider = LOCAL_PROVIDER,
    ):
        self.transport, self.tokenizer, self.capacity, self.tool = (
            transport,
            tokenizer,
            capacity,
            tool,
        )
        self.provider = provider
        self.calls = 0
        self.input_tokens = 0
        self.tool_name = tool["function"]["name"]

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        if (
            request.method != "POST"
            or str(request.url) != f"{self.provider.base_url}/chat/completions"
        ):
            raise InterpretationError("UNEXPECTED_ENDPOINT")
        try:
            body = json.loads(request.content)
        except (ValueError, UnicodeDecodeError) as error:
            raise InterpretationError("INVALID_PROVIDER_REQUEST") from error
        if (
            body.get("model") != self.provider.model
            or body.get("stream") is not False
            or body.get("n") != 1
            or body.get("temperature") != self.provider.temperature
            or body.get("reasoning_effort") != self.provider.reasoning_effort
            or body.get("max_completion_tokens") != OUTPUT_TOKENS
            or body.get("parallel_tool_calls") is not False
            or body.get("tool_choice")
            != {"type": "function", "function": {"name": self.tool_name}}
            or body.get("tools") != [self.tool]
            or body.get("chat_template_kwargs")
            != (self.provider.extra_body or {}).get("chat_template_kwargs")
        ):
            raise InterpretationError("INVALID_PROVIDER_REQUEST")
        messages = tuple(body.get("messages", ()))
        self.input_tokens = token_count(self.tokenizer, messages, body["tools"])
        if (
            self.input_tokens > 8192
            or self.input_tokens + OUTPUT_TOKENS > self.capacity
        ):
            raise InterpretationError("CONTEXT_BUDGET_EXCEEDED")
        self.calls += 1
        started = time.monotonic()
        response: httpx.Response | None = None
        try:
            response = await self.transport.handle_async_request(request)
            declared_size = response.headers.get("content-length")
            if declared_size is not None and (
                not declared_size.isdecimal()
                or int(declared_size) > MAX_PROVIDER_RESPONSE_BYTES
            ):
                raise InterpretationError("PROVIDER_RESPONSE_TOO_LARGE")
            payload = await response.aread()
            if len(payload) > MAX_PROVIDER_RESPONSE_BYTES:
                raise InterpretationError("PROVIDER_RESPONSE_TOO_LARGE")
            self._validate_response(payload, self.tool_name, self.provider)
            body = json.loads(payload)
            usage = body.get("usage") if isinstance(body, dict) else None
            provider_input = (
                usage.get("prompt_tokens") if isinstance(usage, dict) else None
            )
            provider_output = (
                usage.get("completion_tokens") if isinstance(usage, dict) else None
            )
            if (
                isinstance(provider_input, bool)
                or not isinstance(provider_input, int)
                or provider_input < 0
            ):
                provider_input = None
            if (
                isinstance(provider_output, bool)
                or not isinstance(provider_output, int)
                or provider_output < 0
            ):
                provider_output = None
            trace_event(
                {
                    "name": self.tool_name,
                    "result": "success",
                    "model": body.get("model"),
                    "input_tokens": self.input_tokens,
                    "input_token_source": "provider"
                    if provider_input is not None
                    else "local_tokenizer_estimate",
                    "provider_input_tokens": provider_input,
                    "output_tokens": provider_output,
                    "available_tokens": self.capacity
                    - self.input_tokens
                    - OUTPUT_TOKENS,
                    "duration_ms": int((time.monotonic() - started) * 1000),
                }
            )
        except BaseException:
            trace_event(
                {
                    "name": self.tool_name,
                    "result": "failure",
                    "duration_ms": int((time.monotonic() - started) * 1000),
                    "input_tokens": self.input_tokens,
                    "input_token_source": "local_tokenizer_estimate",
                }
            )
            if response is not None:
                await response.aclose()
            raise
        if response is None:
            raise InterpretationError("PROVIDER_FAILURE")
        return response

    @staticmethod
    def _validate_response(
        payload: bytes, tool_name: str, provider: Provider = LOCAL_PROVIDER
    ) -> None:
        try:
            body = json.loads(payload)
            choices = body["choices"]
        except (KeyError, TypeError, ValueError, UnicodeDecodeError) as error:
            raise InterpretationError("MALFORMED_PROVIDER_RESPONSE") from error
        if not provider.owns(body.get("model")):
            raise InterpretationError("MODEL_IDENTITY_MISMATCH")
        if not isinstance(choices, list) or len(choices) != 1:
            raise InterpretationError("INVALID_CHOICE_COUNT")
        choice = choices[0]
        if not provider.finished(choice.get("finish_reason")):
            raise InterpretationError("TRUNCATED_RESPONSE")
        calls = choice.get("message", {}).get("tool_calls")
        if not isinstance(calls, list) or len(calls) != 1:
            raise InterpretationError("INVALID_TOOL_CALL_COUNT")
        call = calls[0]
        function = call.get("function", {}) if isinstance(call, dict) else {}
        if (
            not isinstance(call.get("id"), str)
            or not call["id"].strip()
            or function.get("name") != tool_name
            or not isinstance(function.get("arguments"), str)
        ):
            raise InterpretationError("INVALID_TOOL_CALL")
        try:
            arguments = json.loads(function["arguments"])
            (
                TurnSelection if tool_name == "interpret_turn" else EvidenceSelection
            ).model_validate(arguments, strict=True)
        except ValidationError as error:
            raise InterpretationError(
                "INVALID_TOOL_ARGUMENTS",
                schema_diagnostics=_schema_diagnostics(error),
            ) from error
        except (TypeError, ValueError) as error:
            raise InterpretationError("INVALID_TOOL_ARGUMENTS") from error

    async def aclose(self) -> None:
        await self.transport.aclose()


def build_interpreter(
    *,
    tokenizer: Any,
    provider_capacity: int,
    tool: dict[str, Any],
    transport: httpx.AsyncBaseTransport | None = None,
    tool_name: str = "interpret_turn",
    provider: Provider = LOCAL_PROVIDER,
) -> tuple[ChatOpenAI, GuardedTransport, httpx.AsyncClient, httpx.Client]:
    guarded = GuardedTransport(
        transport or httpx.AsyncHTTPTransport(),
        tokenizer,
        provider_capacity,
        tool,
        provider,
    )
    async_client = httpx.AsyncClient(
        transport=guarded,
        trust_env=False,
        follow_redirects=False,
        timeout=httpx.Timeout(30),
    )
    sync_client = httpx.Client(
        transport=RejectSyncTransport(), trust_env=False, follow_redirects=False
    )
    model = ChatOpenAI(
        model=provider.model,
        api_key=provider.api_key,
        base_url=provider.base_url,
        use_responses_api=False,
        max_retries=0,
        streaming=False,
        n=1,
        max_tokens=OUTPUT_TOKENS,
        temperature=provider.temperature,
        reasoning_effort=provider.reasoning_effort,
        extra_body=provider.extra_body,
        http_async_client=async_client,
        http_client=sync_client,
    ).bind_tools([tool], tool_choice=tool_name, parallel_tool_calls=False)
    return model, guarded, async_client, sync_client


async def interpret_turn(
    state: Memory,
    turn: SanitizedTurn,
    *,
    tokenizer: Any,
    provider_capacity: int,
    transport: httpx.AsyncBaseTransport | None = None,
    provider: Provider = LOCAL_PROVIDER,
) -> InterpretationResult:
    try:
        catalog = build_candidate_catalog(state, turn)
        tool = interpretation_tool(catalog)
        packed, input_tokens = pack_model_context(
            state,
            turn,
            catalog,
            tokenizer=tokenizer,
            tools=[tool],
            provider_capacity=provider_capacity,
        )
    except ModelContextError as error:
        raise InterpretationError(str(error)) from error
    except MemoryError as error:
        raise InterpretationError("INVALID_CANDIDATE_CATALOG") from error
    model, guarded, async_client, sync_client = build_interpreter(
        tokenizer=tokenizer,
        provider_capacity=provider_capacity,
        tool=tool,
        transport=transport,
        provider=provider,
    )
    started = time.monotonic()
    try:
        # This overrides both ambient LangSmith settings and any inherited trace.
        with tracing_context(enabled=False):
            response = await model.ainvoke(
                list(packed.messages), config={"callbacks": [], "metadata": {}}
            )
        metadata = response.response_metadata or {}
        if not provider.owns(metadata.get("model_name")):
            raise InterpretationError("MODEL_IDENTITY_MISMATCH")
        if response.invalid_tool_calls:
            raise InterpretationError("INVALID_TOOL_CALL")
        calls = response.tool_calls
        if len(calls) != 1:
            raise InterpretationError("INVALID_TOOL_CALL_COUNT")
        call = calls[0]
        if (
            call.get("name") != "interpret_turn"
            or not isinstance(call.get("id"), str)
            or not call["id"].strip()
        ):
            raise InterpretationError("INVALID_TOOL_CALL")
        selection = TurnSelection.model_validate(call.get("args"), strict=True)
        proposal = expand_selection(catalog, selection)
        if proposal.requested_action == "knowledge" and not knowledge_query_is_eligible(
            proposal
        ):
            raise InterpretationError("KNOWLEDGE_QUERY_UNAVAILABLE")
        usage = metadata.get("token_usage") or {}
        prompt_hash = sha256(
            json.dumps(packed.messages, separators=(",", ":"), sort_keys=True).encode()
        ).hexdigest()
        schema_hash = sha256(
            json.dumps(tool, separators=(",", ":"), sort_keys=True).encode()
        ).hexdigest()
        return InterpretationResult(
            proposal,
            call["id"],
            provider.model,
            guarded.input_tokens or input_tokens,
            usage.get("completion_tokens"),
            provider_capacity - OUTPUT_TOKENS,
            int((time.monotonic() - started) * 1000),
            prompt_hash,
            schema_hash,
        )
    except InterpretationError:
        raise
    except CandidateSelectionError as error:
        raise InterpretationError(error.category) from error
    except ValidationError as error:
        raise InterpretationError(
            "INVALID_TOOL_ARGUMENTS",
            schema_diagnostics=_schema_diagnostics(error),
        ) from error
    except Exception as error:
        raise InterpretationError("PROVIDER_FAILURE") from error
    finally:
        await async_client.aclose()
        sync_client.close()


def evidence_selection_tool(record_ids: list[str]) -> dict[str, Any]:
    schema = EvidenceSelection.model_json_schema()
    items = schema["properties"]["evidence_ids"]["items"]
    items["enum"] = record_ids
    schema["properties"]["evidence_ids"]["uniqueItems"] = True
    schema["properties"]["evidence_ids"]["maxItems"] = min(5, len(record_ids))
    return {
        "type": "function",
        "function": {
            "name": "select_evidence",
            "description": "Select only IDs supporting the answer, or clarify.",
            "parameters": schema,
        },
    }


async def select_evidence(
    question: str,
    records: list[dict[str, Any]],
    *,
    tokenizer: Any,
    provider_capacity: int,
    transport: httpx.AsyncBaseTransport | None = None,
    provider: Provider = LOCAL_PROVIDER,
) -> EvidenceSelection:
    ids = [row["id"] for row in records]
    if not ids:
        raise InterpretationError("EMPTY_RETRIEVAL")
    tool = evidence_selection_tool(ids)
    messages = (
        {
            "role": "system",
            "content": "Select public evidence only. Call select_evidence exactly once. A clarification has no IDs; an answer has one or more unique IDs.",
        },
        {
            "role": "user",
            "content": json.dumps(
                {
                    "question": question,
                    "evidence": [
                        {"id": row["id"], "text": row["metadata"]["generation_text"]}
                        for row in records
                    ],
                },
                separators=(",", ":"),
                sort_keys=True,
            ),
        },
    )
    # Template counting occurs in GuardedTransport before I/O.
    model, _, async_client, sync_client = build_interpreter(
        tokenizer=tokenizer,
        provider_capacity=provider_capacity,
        tool=tool,
        transport=transport,
        tool_name="select_evidence",
        provider=provider,
    )
    try:
        with tracing_context(enabled=False):
            response = await model.ainvoke(
                list(messages), config={"callbacks": [], "metadata": {}}
            )
        if (
            not provider.owns((response.response_metadata or {}).get("model_name"))
            or len(response.tool_calls) != 1
        ):
            raise InterpretationError("INVALID_TOOL_CALL")
        call = response.tool_calls[0]
        if call.get("name") != "select_evidence":
            raise InterpretationError("INVALID_TOOL_CALL")
        selected = EvidenceSelection.model_validate(call.get("args"), strict=True)
        if len(set(selected.evidence_ids)) != len(selected.evidence_ids) or not set(
            selected.evidence_ids
        ) <= set(ids):
            raise InterpretationError("INVALID_EVIDENCE_SELECTION")
        if selected.needs_clarification != (not selected.evidence_ids):
            raise InterpretationError("INVALID_EVIDENCE_SELECTION")
        return selected
    except InterpretationError:
        raise
    except Exception as error:
        raise InterpretationError("PROVIDER_FAILURE") from error
    finally:
        await async_client.aclose()
        sync_client.close()
