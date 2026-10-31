import asyncio
import json

import httpx
import pytest

from grounded_mortgage_assistant.guardrails.privacy import ingest
from grounded_mortgage_assistant.llm.conversation import (
    LOCAL_MODEL,
    InterpretationError,
    interpret_turn,
    interpretation_tool,
    select_evidence,
)
from grounded_mortgage_assistant.llm.interpretation_candidates import (
    build_candidate_catalog,
)
from grounded_mortgage_assistant.state.memory import Memory


class Tokenizer:
    def apply_chat_template(self, messages, **kwargs):
        return {"input_ids": [[1] * 10]}


def response(request: httpx.Request) -> httpx.Response:
    body = json.loads(request.content)
    assert body["model"] == LOCAL_MODEL
    assert body["max_completion_tokens"] == 1800
    assert body["parallel_tool_calls"] is False
    assert body["temperature"] == 0
    assert body["chat_template_kwargs"] == {"enable_thinking": False}
    assert [message["role"] for message in body["messages"]] == ["system", "user"]
    return httpx.Response(
        200,
        json={
            "model": LOCAL_MODEL,
            "choices": [
                {
                    "finish_reason": "tool_calls",
                    "message": {
                        "role": "assistant",
                        "tool_calls": [
                            {
                                "id": "call_1",
                                "type": "function",
                                "function": {
                                    "name": "interpret_turn",
                                    "arguments": json.dumps(
                                        {
                                            "subject": "self",
                                            "conditional": False,
                                            "requested_action": "acknowledge",
                                            "fact_candidate_ids": [],
                                            "scope_candidate_ids": [],
                                        }
                                    ),
                                },
                            }
                        ],
                    },
                }
            ],
            "usage": {"completion_tokens": 4},
        },
    )


def test_acknowledgement_omits_optional_knowledge_question_without_mutating_state():
    state = Memory()
    result = asyncio.run(
        interpret_turn(
            state,
            ingest("Thanks."),
            tokenizer=Tokenizer(),
            provider_capacity=262144,
            transport=httpx.MockTransport(response),
        )
    )
    assert result.call_id == "call_1"
    assert result.interpretation.knowledge_question is None
    assert state == Memory()
    assert result.provider_input_tokens == 260344
    assert len(result.prompt_hash) == len(result.schema_hash) == 64


def test_evidence_selector_uses_only_current_retrieved_id_descriptor():
    records = [{"id": "public-1", "metadata": {"generation_text": "Public evidence"}}]

    def selected(request):
        body = json.loads(request.content)
        assert body["tool_choice"] == {
            "type": "function",
            "function": {"name": "select_evidence"},
        }
        assert body["tools"][0]["function"]["parameters"]["properties"]["evidence_ids"][
            "items"
        ]["enum"] == ["public-1"]
        user = json.loads(body["messages"][-1]["content"])
        assert user == {
            "evidence": [{"id": "public-1", "text": "Public evidence"}],
            "question": "What is a mortgage loan?",
        }
        return httpx.Response(
            200,
            json={
                "model": LOCAL_MODEL,
                "choices": [
                    {
                        "finish_reason": "tool_calls",
                        "message": {
                            "role": "assistant",
                            "content": "",
                            "tool_calls": [
                                {
                                    "id": "call",
                                    "function": {
                                        "name": "select_evidence",
                                        "arguments": json.dumps(
                                            {
                                                "evidence_ids": ["public-1"],
                                                "needs_clarification": False,
                                            }
                                        ),
                                    },
                                }
                            ],
                        },
                    }
                ],
            },
        )

    result = asyncio.run(
        select_evidence(
            "What is a mortgage loan?",
            records,
            tokenizer=Tokenizer(),
            provider_capacity=262144,
            transport=httpx.MockTransport(selected),
        )
    )
    assert result.evidence_ids == ["public-1"]


def test_unbuffered_async_stream_response_is_buffered_before_validation():
    class Stream(httpx.AsyncByteStream):
        def __init__(self, content):
            self.content = content

        async def __aiter__(self):
            yield self.content

        async def aclose(self):
            return None

    def streamed(request):
        return httpx.Response(200, stream=Stream(response(request).content))

    result = asyncio.run(
        interpret_turn(
            Memory(),
            ingest("Thanks."),
            tokenizer=Tokenizer(),
            provider_capacity=262144,
            transport=httpx.MockTransport(streamed),
        )
    )
    assert result.call_id == "call_1"


def test_actual_serialized_request_excludes_private_amount_and_original_canary():
    canary = "1820000"

    def no_private_canary(request):
        serialized = request.content.decode()
        assert canary not in serialized
        assert "client-message-id" not in serialized
        return response(request)

    asyncio.run(
        interpret_turn(
            Memory(),
            ingest(f"My mortgage is SEK {canary}.", turn_ref="server_turn"),
            tokenizer=Tokenizer(),
            provider_capacity=262144,
            transport=httpx.MockTransport(no_private_canary),
        )
    )


def test_wire_descriptor_uses_exact_current_candidate_ids_and_empty_maximum():
    fact_turn = ingest("My mortgage is SEK 1820000.", turn_ref="current")
    descriptor = interpretation_tool(build_candidate_catalog(Memory(), fact_turn))
    properties = descriptor["function"]["parameters"]["properties"]
    assert properties["fact_candidate_ids"]["items"]["enum"] == ["fc_current_1"]
    assert properties["fact_candidate_ids"]["maxItems"] == 1
    assert properties["fact_candidate_ids"]["uniqueItems"] is True
    assert "enum" not in properties["scope_candidate_ids"]["items"]
    assert properties["scope_candidate_ids"]["maxItems"] == 0
    assert "1820000" not in json.dumps(descriptor)


def test_empty_knowledge_projection_fails_after_one_native_call():
    calls = 0

    def knowledge(request):
        nonlocal calls
        calls += 1
        body = response(request).json()
        arguments = json.loads(
            body["choices"][0]["message"]["tool_calls"][0]["function"]["arguments"]
        )
        arguments["requested_action"] = "knowledge"
        body["choices"][0]["message"]["tool_calls"][0]["function"]["arguments"] = (
            json.dumps(arguments)
        )
        return httpx.Response(200, json=body)

    state = Memory()
    with pytest.raises(InterpretationError, match="KNOWLEDGE_QUERY_UNAVAILABLE"):
        asyncio.run(
            interpret_turn(
                state,
                ingest("Thanks."),
                tokenizer=Tokenizer(),
                provider_capacity=262144,
                transport=httpx.MockTransport(knowledge),
            )
        )
    assert calls == 1
    assert state == Memory()


def test_explicit_knowledge_projection_returns_a_valid_interpretation():
    def knowledge(request):
        body = response(request).json()
        arguments = json.loads(
            body["choices"][0]["message"]["tool_calls"][0]["function"]["arguments"]
        )
        arguments["requested_action"] = "knowledge"
        body["choices"][0]["message"]["tool_calls"][0]["function"]["arguments"] = (
            json.dumps(arguments)
        )
        return httpx.Response(200, json=body)

    result = asyncio.run(
        interpret_turn(
            Memory(),
            ingest("What information about mortgage?"),
            tokenizer=Tokenizer(),
            provider_capacity=262144,
            transport=httpx.MockTransport(knowledge),
        )
    )
    assert result.interpretation.knowledge_question == ["INFORMATION", "MORTGAGE"]


def test_calculation_with_no_current_candidates_accepts_empty_selection():
    def calculation(request):
        body = response(request).json()
        arguments = json.loads(
            body["choices"][0]["message"]["tool_calls"][0]["function"]["arguments"]
        )
        arguments["requested_action"] = "calculate"
        body["choices"][0]["message"]["tool_calls"][0]["function"]["arguments"] = (
            json.dumps(arguments)
        )
        return httpx.Response(200, json=body)

    result = asyncio.run(
        interpret_turn(
            Memory(),
            ingest("Calculate LTV."),
            tokenizer=Tokenizer(),
            provider_capacity=262144,
            transport=httpx.MockTransport(calculation),
        )
    )
    assert result.interpretation.requested_action == "calculate"
    assert not result.interpretation.operations


def test_wrong_kind_candidate_id_is_rejected_after_one_native_call():
    calls = 0

    def wrong_kind(request):
        nonlocal calls
        calls += 1
        body = response(request).json()
        arguments = json.loads(
            body["choices"][0]["message"]["tool_calls"][0]["function"]["arguments"]
        )
        arguments["fact_candidate_ids"] = ["sc_scope_1"]
        body["choices"][0]["message"]["tool_calls"][0]["function"]["arguments"] = (
            json.dumps(arguments)
        )
        return httpx.Response(200, json=body)

    with pytest.raises(InterpretationError, match="INVALID_CANDIDATE_SELECTION"):
        asyncio.run(
            interpret_turn(
                Memory(),
                ingest("I want to refinance my existing mortgage.", turn_ref="scope"),
                tokenizer=Tokenizer(),
                provider_capacity=262144,
                transport=httpx.MockTransport(wrong_kind),
            )
        )
    assert calls == 1


def test_wrong_model_is_fixed_category_without_retry():
    def wrong_model(request):
        return httpx.Response(
            200,
            json={
                "model": "wrong",
                "choices": [
                    {
                        "finish_reason": "stop",
                        "message": {"role": "assistant", "tool_calls": []},
                    }
                ],
            },
        )

    with pytest.raises(InterpretationError, match="MODEL_IDENTITY_MISMATCH"):
        asyncio.run(
            interpret_turn(
                Memory(),
                ingest("Thanks."),
                tokenizer=Tokenizer(),
                provider_capacity=262144,
                transport=httpx.MockTransport(wrong_model),
            )
        )


@pytest.mark.parametrize(
    "text",
    [
        "My mortgage is SEK 1000000 or SEK 2000000 or SEK 3000000.",
        "My mortgage is SEK 1000000 or SEK 2000000 or SEK 3000000 and my property value is SEK 2600000.",
        "My property is house or apartment or terraced house or holiday property or house.",
        "My property is house or house.",
    ],
)
def test_unrepresentable_catalog_fails_before_any_provider_call(text):
    calls = 0

    def provider(request):
        nonlocal calls
        calls += 1
        return response(request)

    state = Memory()
    with pytest.raises(InterpretationError, match="INVALID_CANDIDATE_CATALOG"):
        asyncio.run(
            interpret_turn(
                state,
                ingest(text),
                tokenizer=Tokenizer(),
                provider_capacity=262144,
                transport=httpx.MockTransport(provider),
            )
        )
    assert calls == 0
    assert state == Memory()


@pytest.mark.parametrize(
    ("arguments", "expected_diagnostic", "secret"),
    [
        (
            {"requested_action": "top_secret_canary"},
            ("requested_action", "literal_error"),
            "top_secret_canary",
        ),
        ({"knowledge_tokens": "canary"}, ("$", "extra_forbidden"), "canary"),
        (
            {"fact_candidate_ids": ["dynamic_candidate", "dynamic_candidate"]},
            ("fact_candidate_ids", "duplicate_fact_candidate_ids"),
            "dynamic_candidate",
        ),
        (
            {"unexpected_dynamic_key": "input_canary"},
            ("$", "extra_forbidden"),
            "unexpected_dynamic_key",
        ),
    ],
)
def test_invalid_schema_arguments_have_safe_diagnostics_without_retry_or_mutation(
    arguments, expected_diagnostic, secret
):
    calls = 0

    def invalid(request):
        nonlocal calls
        calls += 1
        body = response(request).json()
        tool_arguments = json.loads(
            body["choices"][0]["message"]["tool_calls"][0]["function"]["arguments"]
        )
        tool_arguments.update(arguments)
        body["choices"][0]["message"]["tool_calls"][0]["function"]["arguments"] = (
            json.dumps(tool_arguments)
        )
        return httpx.Response(200, json=body)

    state = Memory()
    with pytest.raises(InterpretationError, match="INVALID_TOOL_ARGUMENTS") as raised:
        asyncio.run(
            interpret_turn(
                state,
                ingest("Thanks."),
                tokenizer=Tokenizer(),
                provider_capacity=262144,
                transport=httpx.MockTransport(invalid),
            )
        )
    assert calls == 1
    assert state == Memory()
    assert expected_diagnostic in raised.value.schema_diagnostics
    rendered = repr(raised.value.schema_diagnostics)
    assert secret not in rendered
    assert "input_canary" not in rendered


@pytest.mark.parametrize(
    ("mutate", "category"),
    [
        (lambda body: body.update(choices=[]), "INVALID_CHOICE_COUNT"),
        (
            lambda body: body["choices"][0].update(finish_reason="length"),
            "TRUNCATED_RESPONSE",
        ),
        (
            lambda body: body["choices"][0]["message"].update(tool_calls=[]),
            "INVALID_TOOL_CALL_COUNT",
        ),
        (
            lambda body: body["choices"][0]["message"]["tool_calls"][0].update(id=""),
            "INVALID_TOOL_CALL",
        ),
        (
            lambda body: body["choices"][0]["message"]["tool_calls"][0][
                "function"
            ].update(arguments="not-json"),
            "INVALID_TOOL_ARGUMENTS",
        ),
    ],
)
def test_raw_invalid_responses_are_rejected_once_without_mutation(mutate, category):
    calls = 0

    def invalid(request):
        nonlocal calls
        calls += 1
        body = response(request).json()
        mutate(body)
        return httpx.Response(200, json=body)

    state = Memory()
    with pytest.raises(InterpretationError, match=category):
        asyncio.run(
            interpret_turn(
                state,
                ingest("Thanks."),
                tokenizer=Tokenizer(),
                provider_capacity=262144,
                transport=httpx.MockTransport(invalid),
            )
        )
    assert calls == 1
    assert state == Memory()


def test_ambient_proxy_keys_and_tracing_do_not_change_local_wire(monkeypatch):
    monkeypatch.setenv("HTTPS_PROXY", "http://not-local.invalid")
    monkeypatch.setenv("OPENAI_API_KEY", "hosted-secret")
    monkeypatch.setenv("LANGSMITH_TRACING", "true")
    monkeypatch.setenv("LANGCHAIN_TRACING_V2", "true")
    seen = []

    def local_only(request):
        seen.append(request)
        assert request.url.host == "127.0.0.1"
        assert request.headers["authorization"] == "Bearer local"
        return response(request)

    asyncio.run(
        interpret_turn(
            Memory(),
            ingest("Thanks."),
            tokenizer=Tokenizer(),
            provider_capacity=262144,
            transport=httpx.MockTransport(local_only),
        )
    )
    assert len(seen) == 1


def test_transport_closes_after_success_and_provider_validation_failure():
    class ClosingTransport(httpx.AsyncBaseTransport):
        def __init__(self, handler):
            self.handler = handler
            self.closed = False

        async def handle_async_request(self, request):
            return self.handler(request)

        async def aclose(self):
            self.closed = True

    for handler, category in (
        (response, None),
        (lambda request: httpx.Response(200), "MALFORMED_PROVIDER_RESPONSE"),
    ):
        transport = ClosingTransport(handler)
        if category is None:
            asyncio.run(
                interpret_turn(
                    Memory(),
                    ingest("Thanks."),
                    tokenizer=Tokenizer(),
                    provider_capacity=262144,
                    transport=transport,
                )
            )
        else:
            with pytest.raises(InterpretationError, match=category):
                asyncio.run(
                    interpret_turn(
                        Memory(),
                        ingest("Thanks."),
                        tokenizer=Tokenizer(),
                        provider_capacity=262144,
                        transport=transport,
                    )
                )
        assert transport.closed
