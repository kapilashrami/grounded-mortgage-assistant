import asyncio

import httpx
import pytest

from grounded_mortgage_assistant.app import verify_local_provider
from grounded_mortgage_assistant.llm.conversation import LOCAL_MODEL


def test_provider_startup_verifies_literal_model_and_capacity():
    def response(request):
        assert str(request.url) == "http://127.0.0.1:8011/v1/models"
        assert request.headers["authorization"] == "Bearer local"
        return httpx.Response(
            200,
            json={
                "object": "list",
                "data": [
                    {
                        "id": LOCAL_MODEL,
                        "object": "model",
                        "created": 1793371279,
                        "owned_by": "omlx",
                        "max_model_len": 262144,
                    }
                ],
            },
        )

    assert (
        asyncio.run(verify_local_provider(transport=httpx.MockTransport(response)))
        == 262144
    )


def test_provider_startup_rejects_missing_or_invalid_capacity():
    with pytest.raises(RuntimeError, match="LOCAL_PROVIDER_STARTUP_INVALID"):
        asyncio.run(
            verify_local_provider(
                transport=httpx.MockTransport(
                    lambda _: httpx.Response(
                        200, json={"data": [{"id": LOCAL_MODEL, "max_model_len": True}]}
                    )
                )
            )
        )
