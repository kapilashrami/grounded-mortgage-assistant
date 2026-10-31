import asyncio
import json

import httpx
import pytest

from grounded_mortgage_assistant.app import model_cost, verify_openai_provider
from grounded_mortgage_assistant.guardrails.privacy import ingest
from grounded_mortgage_assistant.llm.conversation import (
    LOCAL_PROVIDER,
    OPENAI_URL,
    InterpretationError,
    Provider,
    interpret_turn,
    provider_from_env,
)
from grounded_mortgage_assistant.state.memory import Memory

OPENAI = Provider("openai", "gpt-4.1", OPENAI_URL, "test-key")


class Tokenizer:
    def apply_chat_template(self, messages, **kwargs):
        return {"input_ids": [[1] * 10]}


def openai_response(model="gpt-4.1-2025-04-14", finish_reason="stop"):
    def respond(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        assert str(request.url) == f"{OPENAI_URL}/chat/completions"
        assert request.headers["authorization"] == "Bearer test-key"
        assert body["model"] == "gpt-4.1"
        assert "chat_template_kwargs" not in body
        return httpx.Response(
            200,
            json={
                "model": model,
                "choices": [
                    {
                        "finish_reason": finish_reason,
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
                "usage": {"prompt_tokens": 12, "completion_tokens": 3},
            },
        )

    return respond


def run(transport, provider=OPENAI):
    return asyncio.run(
        interpret_turn(
            Memory(),
            ingest("Thanks."),
            tokenizer=Tokenizer(),
            provider_capacity=262144,
            transport=httpx.MockTransport(transport),
            provider=provider,
        )
    )


def test_provider_defaults_to_local():
    assert provider_from_env({}) is LOCAL_PROVIDER


def test_openai_provider_requires_a_key():
    with pytest.raises(RuntimeError, match="OPENAI_API_KEY"):
        provider_from_env({"LLM_PROVIDER": "openai"})


def test_openai_provider_reads_model_and_hides_key():
    provider = provider_from_env(
        {
            "LLM_PROVIDER": "openai",
            "OPENAI_API_KEY": "secret-key",
            "OPENAI_MODEL": "gpt-4.1-mini",
        }
    )
    assert (provider.name, provider.model, provider.base_url) == (
        "openai",
        "gpt-4.1-mini",
        OPENAI_URL,
    )
    assert "secret-key" not in repr(provider)


def test_unknown_provider_is_rejected():
    with pytest.raises(RuntimeError, match="local or openai"):
        provider_from_env({"LLM_PROVIDER": "cohere"})


def test_openai_forced_tool_call_with_stop_and_dated_model_is_accepted():
    result = run(openai_response())
    assert result.interpretation.requested_action == "acknowledge"
    assert result.model == "gpt-4.1"


def test_openai_truncation_is_still_rejected():
    with pytest.raises(InterpretationError, match="TRUNCATED_RESPONSE"):
        run(openai_response(finish_reason="length"))


def test_openai_other_model_is_rejected():
    with pytest.raises(InterpretationError, match="MODEL_IDENTITY_MISMATCH"):
        run(openai_response(model="gpt-4.1-mini-2025-04-14"))


def test_local_provider_still_rejects_stop():
    def local_stop(request):
        body = openai_response()(request).json()
        body["model"] = LOCAL_PROVIDER.model
        return httpx.Response(200, json=body)

    with pytest.raises(InterpretationError):
        run(local_stop, provider=LOCAL_PROVIDER)


def test_openai_startup_verifies_model_visibility():
    def models(request):
        assert str(request.url) == f"{OPENAI_URL}/models/gpt-4.1"
        return httpx.Response(200, json={"id": "gpt-4.1"})

    asyncio.run(verify_openai_provider(OPENAI, transport=httpx.MockTransport(models)))

    with pytest.raises(RuntimeError, match="OPENAI_PROVIDER_STARTUP_INVALID"):
        asyncio.run(
            verify_openai_provider(
                OPENAI,
                transport=httpx.MockTransport(lambda _: httpx.Response(401, json={})),
            )
        )


def test_cost_is_zero_locally_and_estimated_for_openai():
    assert model_cost(LOCAL_PROVIDER, 100, 10)["external_api_usd"] == 0
    assert model_cost(OPENAI, 1_000_000, 1_000_000)["external_api_usd"] == 10.0
    unpriced = Provider("openai", "gpt-5.5", OPENAI_URL, "k")
    assert model_cost(unpriced, 1, 1)["external_api_usd"] is None


def test_gpt5_provider_sends_reasoning_effort_and_no_temperature():
    provider = provider_from_env(
        {
            "LLM_PROVIDER": "openai",
            "OPENAI_API_KEY": "test-key",
            "OPENAI_MODEL": "gpt-5.4-mini",
            "OPENAI_REASONING_EFFORT": "none",
        }
    )
    assert (provider.temperature, provider.reasoning_effort) == (None, "none")

    def respond(request):
        body = json.loads(request.content)
        assert body["model"] == "gpt-5.4-mini"
        assert "temperature" not in body
        assert body["reasoning_effort"] == "none"
        reply = openai_response(model="gpt-5.4-mini-2026-03-17")
        request_for_gpt41 = httpx.Request(
            "POST",
            request.url,
            headers=request.headers,
            content=json.dumps({**body, "model": "gpt-4.1"}),
        )
        return reply(request_for_gpt41)

    result = run(respond, provider=provider)
    assert result.model == "gpt-5.4-mini"


def test_gpt5_requires_a_valid_reasoning_effort():
    env = {
        "LLM_PROVIDER": "openai",
        "OPENAI_API_KEY": "k",
        "OPENAI_MODEL": "gpt-5.4-mini",
    }
    with pytest.raises(RuntimeError, match="REASONING_EFFORT"):
        provider_from_env(env)
    with pytest.raises(RuntimeError, match="REASONING_EFFORT"):
        provider_from_env({**env, "OPENAI_REASONING_EFFORT": "low"})


def test_reasoning_effort_is_rejected_for_non_reasoning_models():
    with pytest.raises(RuntimeError, match="gpt-5"):
        provider_from_env(
            {
                "LLM_PROVIDER": "openai",
                "OPENAI_API_KEY": "k",
                "OPENAI_REASONING_EFFORT": "low",
            }
        )
