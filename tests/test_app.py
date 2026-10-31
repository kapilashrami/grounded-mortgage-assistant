from __future__ import annotations

import asyncio

import httpx
import pytest
from fastapi.testclient import TestClient
from support import FixtureEvidence

from grounded_mortgage_assistant.app import create_app
from grounded_mortgage_assistant.config import not_found_answer
from grounded_mortgage_assistant.guardrails.pii import PiiDecision
from grounded_mortgage_assistant.llm.conversation import (
    EvidenceSelection,
    InterpretationResult,
)
from grounded_mortgage_assistant.llm.interpretation_candidates import (
    TurnSelection,
    build_candidate_catalog,
    expand_selection,
)
from grounded_mortgage_assistant.state.memory import (
    FactOperation,
    TurnInterpretation,
    build_context,
)
from grounded_mortgage_assistant.state.runtime import MemoryRuntime


async def fake_pii_check(text: str) -> PiiDecision:
    """Deterministic stand-in for the loopback detector: flags only email addresses."""
    if "@" in text:
        return PiiDecision(pii_detected=True, category="email_address")
    return PiiDecision(pii_detected=False, category=None)


def make_app(**kwargs):
    kwargs.setdefault("evidence", FixtureEvidence())
    return create_app(pii_checker=fake_pii_check, **kwargs)


def native_result(proposal):
    return InterpretationResult(
        proposal, "call", "local", 12, 2, 260344, 1, "a" * 64, "b" * 64
    )


def proposal(turn, *, action="acknowledge", operations=(), knowledge_question=None):
    return TurnInterpretation(
        subject="self",
        conditional=False,
        operations=list(operations),
        goal="unchanged",
        borrowing="unchanged",
        ownership="unchanged",
        requested_action=action,
        knowledge_question=knowledge_question,
        unresolved=[],
    )


def operation(turn, field, value_index):
    return FactOperation(
        field=field,
        operation="set",
        value_refs=[turn.protected_values[value_index].value_ref],
        source_span={"token_start": 0, "token_end": len(turn.tokens) - 1},
    )


def test_negative_correction_restricts_prior_fact_and_never_calculates():
    async def native(_, turn):
        if len(turn.protected_values) == 2:
            return native_result(
                proposal(
                    turn,
                    action="calculate",
                    operations=[
                        operation(turn, "existing_loan_amount", 0),
                        operation(turn, "property_value", 1),
                    ],
                )
            )
        return native_result(proposal(turn))

    client = TestClient(make_app(interpreter=native))
    client.post(
        "/api/chat",
        json={
            "session_id": "s",
            "message_id": "one",
            "message": "My mortgage is SEK 1820000 and my property value is SEK 2600000.",
        },
    )
    response = client.post(
        "/api/chat",
        json={
            "session_id": "s",
            "message_id": "two",
            "message": "My mortgage is not SEK 1560000.",
        },
    ).json()
    assert response["status"] == "CLARIFY"
    assert "calculations" not in response
    assert response["memory_receipt"]["invalidated"] == [
        "calculation",
        "offer",
        "review",
    ]


def test_wrong_native_candidate_is_a_clarification_not_a_positive_acknowledgement():
    async def native(_, turn):
        forged = FactOperation(
            field="existing_loan_amount",
            operation="set",
            value_refs=["v_foreign"],
            source_span={"token_start": 0, "token_end": len(turn.tokens) - 1},
        )
        return native_result(proposal(turn, action="calculate", operations=[forged]))

    client = TestClient(make_app(interpreter=native))
    response = client.post(
        "/api/chat",
        json={
            "session_id": "s",
            "message_id": "one",
            "message": "My mortgage is SEK 1820000.",
        },
    ).json()
    assert response["status"] == "CLARIFY"
    assert "calculations" not in response


def test_reset_during_awaited_generation_does_not_commit_late_response():
    started, release = asyncio.Event(), asyncio.Event()

    async def native(_, turn):
        started.set()
        await release.wait()
        return native_result(proposal(turn))

    async def exercise():
        # Runtime behavior is exercised directly because TestClient is synchronous.
        from grounded_mortgage_assistant.state.runtime import MemoryRuntime

        runtime = MemoryRuntime()
        session, generation, _ = await runtime.admit("s", "m", {"message": "x"})
        await runtime.reset("s")
        assert not await runtime.commit(
            session, generation, session.memory, "m", {"message": "x"}, ("{}",)
        )

    asyncio.run(exercise())


def knowledge_client(calls, score):
    """A public-question app whose retriever returns one sample record at `score`."""

    async def native(_, turn):
        return native_result(
            proposal(
                turn,
                action="knowledge",
                knowledge_question=["INFORMATION", "MORTGAGE"],
            )
        )

    evidence = FixtureEvidence()
    record_id = "sample-mortgage-01"
    metadata = evidence.records[record_id]
    generation = FixtureEvidence.generation_text(metadata)

    async def retrieve(query):
        calls.append(("retrieve", query))
        return [
            {
                "id": record_id,
                "score": score,
                "metadata": {"generation_text": generation},
            }
        ]

    async def select(question, records):
        calls.append(("select", question))
        assert [row["id"] for row in records] == [record_id]
        return EvidenceSelection(evidence_ids=[record_id], needs_clarification=False)

    client = TestClient(
        make_app(
            interpreter=native, retriever=retrieve, selector=select, evidence=evidence
        )
    )
    return client, record_id, metadata


def test_knowledge_retrieves_the_raw_public_question_and_renders_selected_evidence():
    calls = []
    client, record_id, metadata = knowledge_client(calls, score=0.9)
    response = chat(client, "s", "one", "What is a mortgage loan?").json()
    # Retrieval and selection receive the raw question; only the trace is corpus-owned.
    assert calls == [
        ("retrieve", "What is a mortgage loan?"),
        ("select", "What is a mortgage loan?"),
    ]
    assert response["status"] == "ANSWERED"
    assert response["answer"] == metadata["final_answer"]
    assert response["citations"][0]["evidence_id"] == record_id
    (trace,) = client.get("/api/traces").json()
    assert trace["question"] == metadata["question"]


def test_public_question_the_lexical_matcher_missed_reaches_the_retriever():
    calls = []
    client, _, _ = knowledge_client(calls, score=0.9)
    response = chat(client, "s", "founded", "When was the company founded?").json()
    assert calls[0] == ("retrieve", "When was the company founded?")
    assert response["status"] == "ANSWERED"


def test_low_relevance_public_question_says_the_corpus_does_not_cover_it():
    calls = []
    client, _, _ = knowledge_client(calls, score=0.01)
    response = chat(
        client, "s", "weather", "What is the weather in Paris today?"
    ).json()
    assert calls == [("retrieve", "What is the weather in Paris today?")]
    assert response["status"] == "CLARIFY"
    assert response["answer"] == (not_found_answer(""))
    assert response["citations"] == []


def test_forget_replays_exact_receipt_and_blocks_forgotten_chat_replay():
    calls = []

    async def native(_, turn):
        calls.append(turn.turn_ref)
        return native_result(proposal(turn))

    async def exercise():
        app = make_app(interpreter=native)
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(
            transport=transport, base_url="http://test"
        ) as client:
            chat = await client.post(
                "/api/chat",
                json={"session_id": "s", "message_id": "one", "message": "Thanks."},
            )
            assert chat.status_code == 200
            payload = {"message_id": "forget-one", "scope": {"kind": "all_chat"}}
            first = await client.post("/api/sessions/s/forget", json=payload)
            duplicate = await client.post("/api/sessions/s/forget", json=payload)
            changed = await client.post(
                "/api/sessions/s/forget",
                json={
                    "message_id": "forget-one",
                    "scope": {"kind": "messages", "message_ids": ["one"]},
                },
            )
            replay = await client.post(
                "/api/chat",
                json={"session_id": "s", "message_id": "one", "message": "Thanks."},
            )

        assert first.status_code == duplicate.status_code == 200
        assert first.json() == duplicate.json()
        assert first.json()["memory_receipt"]["redacted_message_ids"] == ["one"]
        assert changed.status_code == 409
        assert replay.json()["detail"]["code"] == "FORGOTTEN_TURN"
        assert calls and len(calls) == 1

    asyncio.run(exercise())


def test_forget_targeted_rejected_ingress_and_unknown_target_do_not_erase_chat():
    async def native(_, turn):
        return native_result(proposal(turn))

    async def exercise():
        runtime = MemoryRuntime()
        rejected_payload = {"session_id": "rejected", "message_id": "blocked"}
        session, generation, _ = await runtime.admit(
            "rejected", "blocked", rejected_payload
        )
        await runtime.reject(session, generation, "blocked", rejected_payload, "{}")
        app = make_app(interpreter=native, runtime=runtime)
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(
            transport=transport, base_url="http://test"
        ) as client:
            targeted = await client.post(
                "/api/sessions/rejected/forget",
                json={
                    "message_id": "forget-blocked",
                    "scope": {"kind": "messages", "message_ids": ["blocked"]},
                },
            )
            replay = await client.post(
                "/api/chat",
                json={
                    "session_id": "rejected",
                    "message_id": "blocked",
                    "message": "Thanks.",
                },
            )
            chat = await client.post(
                "/api/chat",
                json={"session_id": "live", "message_id": "one", "message": "Thanks."},
            )
            assert chat.status_code == 200
            before = runtime.sessions["live"].memory
            unknown = await client.post(
                "/api/sessions/live/forget",
                json={
                    "message_id": "forget-unknown",
                    "scope": {"kind": "messages", "message_ids": ["missing"]},
                },
            )

        assert targeted.status_code == 200
        assert targeted.json()["memory_receipt"]["redacted_message_ids"] == ["blocked"]
        assert replay.json()["detail"]["code"] == "FORGOTTEN_TURN"
        assert unknown.status_code == 422
        assert runtime.sessions["live"].memory == before
        assert not runtime.sessions["live"].chats["one"].forgotten

    asyncio.run(exercise())


def test_invalid_forget_is_terminal_replay_and_does_not_reserve_again():
    async def native(_, turn):
        return native_result(proposal(turn))

    async def exercise():
        runtime = MemoryRuntime()
        app = make_app(interpreter=native, runtime=runtime)
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(
            transport=transport, base_url="http://test"
        ) as client:
            await client.post(
                "/api/chat",
                json={"session_id": "s", "message_id": "one", "message": "Thanks."},
            )
            payload = {
                "message_id": "invalid-forget",
                "scope": {"kind": "messages", "message_ids": ["missing"]},
            }
            first = await client.post("/api/sessions/s/forget", json=payload)
            second = await client.post("/api/sessions/s/forget", json=payload)
            changed = await client.post(
                "/api/sessions/s/forget",
                json={**payload, "scope": {"kind": "all_chat"}},
            )
        assert first.status_code == second.status_code == 422
        assert first.json() == second.json()
        assert changed.status_code == 409
        record = runtime.sessions["s"].commands["invalid-forget"]
        assert record.response is not None and not record.pending

    asyncio.run(exercise())


def test_forget_cancels_inflight_chat_before_ack_and_late_commit_cannot_win():
    started, finalized = asyncio.Event(), asyncio.Event()

    async def native(_, turn):
        started.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            finalized.set()
            return native_result(proposal(turn))
        raise AssertionError("unreachable")

    async def exercise():
        runtime = MemoryRuntime()
        app = make_app(interpreter=native, runtime=runtime)
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(
            transport=transport, base_url="http://test"
        ) as client:
            pending = asyncio.create_task(
                client.post(
                    "/api/chat",
                    json={"session_id": "s", "message_id": "one", "message": "Thanks."},
                )
            )
            await started.wait()
            forgotten = await client.post(
                "/api/sessions/s/forget",
                json={"message_id": "forget-one", "scope": {"kind": "all_chat"}},
            )
            late = await pending

        assert finalized.is_set()
        assert forgotten.status_code == 200
        assert late.json()["detail"]["code"] == "SESSION_RESET"
        assert runtime.sessions["s"].chats["one"].forgotten

    asyncio.run(exercise())


def test_forget_command_id_cannot_collide_with_chat_id():
    async def native(_, turn):
        return native_result(proposal(turn))

    async def exercise():
        app = make_app(interpreter=native)
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(
            transport=transport, base_url="http://test"
        ) as client:
            await client.post(
                "/api/chat",
                json={"session_id": "s", "message_id": "one", "message": "Thanks."},
            )
            response = await client.post(
                "/api/sessions/s/forget",
                json={"message_id": "one", "scope": {"kind": "all_chat"}},
            )
        assert response.status_code == 409
        assert response.json()["detail"]["code"] == "IDENTITY_CONFLICT"

    asyncio.run(exercise())


def test_ingress_rejection_is_replayed_and_can_be_forgotten():
    calls = []

    async def native(_, turn):
        calls.append(turn)
        return native_result(proposal(turn))

    async def exercise():
        runtime = MemoryRuntime()
        app = make_app(interpreter=native, runtime=runtime)
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(
            transport=transport, base_url="http://test"
        ) as client:
            payload = {
                "session_id": "s",
                "message_id": "blocked",
                "message": "email me at a@b.se",
            }
            first = await client.post("/api/chat", json=payload)
            replay = await client.post("/api/chat", json=payload)
            changed = await client.post(
                "/api/chat",
                json={**payload, "message": "Thanks."},
            )
            forgotten = await client.post(
                "/api/sessions/s/forget",
                json={
                    "message_id": "forget-blocked",
                    "scope": {"kind": "messages", "message_ids": ["blocked"]},
                },
            )
            after = await client.post("/api/chat", json=payload)
        assert first.status_code == replay.status_code == 422
        assert first.json() == replay.json()
        assert changed.status_code == 409
        assert forgotten.status_code == 200
        assert after.json()["detail"]["code"] == "FORGOTTEN_TURN"
        assert not calls

    asyncio.run(exercise())


def test_cancelled_http_chat_releases_only_its_own_reservation():
    started, release = asyncio.Event(), asyncio.Event()

    async def native(_, turn):
        started.set()
        await release.wait()
        return native_result(proposal(turn))

    async def exercise():
        app = make_app(interpreter=native)
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(
            transport=transport, base_url="http://test"
        ) as client:
            pending = asyncio.create_task(
                client.post(
                    "/api/chat",
                    json={"session_id": "s", "message_id": "one", "message": "Thanks."},
                )
            )
            await started.wait()
            pending.cancel()
            with __import__("pytest").raises(asyncio.CancelledError):
                await pending
            next_chat = asyncio.create_task(
                client.post(
                    "/api/chat",
                    json={"session_id": "s", "message_id": "two", "message": "Thanks."},
                )
            )
            await started.wait()
            release.set()
            accepted = await next_chat
        assert accepted.status_code == 200

    asyncio.run(exercise())


def test_cancelled_forget_waits_for_chat_cleanup_then_exact_retry_replays_terminal_receipt():
    started, cleanup_held, release = asyncio.Event(), asyncio.Event(), asyncio.Event()

    async def native(_, turn):
        started.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            cleanup_held.set()
            await release.wait()
            raise

    async def exercise():
        app = make_app(interpreter=native)
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(
            transport=transport, base_url="http://test"
        ) as client:
            chat = asyncio.create_task(
                client.post(
                    "/api/chat",
                    json={
                        "session_id": "s",
                        "message_id": "chat",
                        "message": "Thanks.",
                    },
                )
            )
            await started.wait()
            payload = {"message_id": "forget", "scope": {"kind": "all_chat"}}
            forgetting = asyncio.create_task(
                client.post("/api/sessions/s/forget", json=payload)
            )
            await cleanup_held.wait()
            forgetting.cancel()
            release.set()
            with pytest.raises(asyncio.CancelledError):
                await forgetting
            with pytest.raises(asyncio.CancelledError):
                await chat
            replay = await client.post("/api/sessions/s/forget", json=payload)
        assert replay.status_code == 200
        assert replay.json()["message_id"] == "forget"

    asyncio.run(exercise())


def test_late_finish_command_cannot_repopulate_reset_session():
    async def exercise():
        runtime = MemoryRuntime()
        session, _, _ = await runtime.admit("s", "chat", {})
        await runtime.reset("s", "reset", {"id": "reset"}, "{}")
        stored = await runtime.finish_command(
            session, "late", {"id": "late"}, "{}", generation=session.generation - 1
        )
        assert not stored
        assert not session.commands

    asyncio.run(exercise())


def test_reset_rejects_existing_chat_or_command_identity():
    async def native(_, turn):
        return native_result(proposal(turn))

    async def exercise():
        app = make_app(interpreter=native)
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(
            transport=transport, base_url="http://test"
        ) as client:
            await client.post(
                "/api/chat",
                json={"session_id": "s", "message_id": "chat", "message": "Thanks."},
            )
            chat_collision = await client.post(
                "/api/sessions/s/reset", json={"message_id": "chat"}
            )
            await client.post(
                "/api/sessions/s/forget",
                json={
                    "message_id": "forget",
                    "scope": {"kind": "messages", "message_ids": ["chat"]},
                },
            )
            command_collision = await client.post(
                "/api/sessions/s/reset", json={"message_id": "forget"}
            )
        assert chat_collision.status_code == command_collision.status_code == 409
        assert chat_collision.json()["detail"]["code"] == "IDENTITY_CONFLICT"
        assert command_collision.json()["detail"]["code"] == "IDENTITY_CONFLICT"

    asyncio.run(exercise())


# Chat-first mortgage journey: the customer types amounts and intent in chat; the
# server answers with an LTV card or a ready draft. There is no guided question flow.
AMOUNTS = "What is my loan to value, my existing mortgage is 2000000 and property is worth 2500000"
SCOPE = "I want to refinance my existing mortgage without borrowing more for my house."
ENVELOPE = {
    "session_id",
    "message_id",
    "trace_id",
    "citations",
    "related_articles",
    "actions",
}


def chat_first_native(captured=None):
    """Stand in for Qwen: select every server-built candidate, as production expands."""

    async def native(memory, turn):
        if captured is not None:
            captured.append(build_context(memory, current_turn=turn))
        catalog = build_candidate_catalog(memory, turn)
        projection = turn.model_projection
        action = (
            "calculate"
            if "LTV" in projection or len(turn.protected_values) == 2
            else "acknowledge"
        )
        return native_result(
            expand_selection(
                catalog,
                TurnSelection(
                    subject="self",
                    conditional=False,
                    requested_action=action,
                    fact_candidate_ids=list(catalog.fact_candidates),
                    scope_candidate_ids=list(catalog.scope_candidates),
                ),
            )
        )

    return native


def chat(client, session_id, message_id, message):
    return client.post(
        "/api/chat",
        json={"session_id": session_id, "message_id": message_id, "message": message},
    )


def test_chat_first_journey_calculates_then_drafts_with_replay_and_exact_provenance():
    captured = []
    client = TestClient(make_app(interpreter=chat_first_native(captured)))

    calculated = chat(client, "s", "amounts", AMOUNTS)
    assert calculated.status_code == 200
    body = calculated.json()
    assert ENVELOPE <= set(body)
    assert body["calculations"][0]["percent"] == "80.00"
    assert "pending_question" not in body and "mortgage_draft" not in body

    # Exact replay returns the stored body; an altered payload conflicts.
    assert chat(client, "s", "amounts", AMOUNTS).json() == body
    assert chat(client, "s", "amounts", "Thanks.").status_code == 409

    scoped = chat(client, "s", "scope", SCOPE).json()
    assert scoped["status"] == "ANSWERED"
    assert "calculations" not in scoped and "mortgage_draft" not in scoped

    drafted = chat(client, "s", "draft", "Draft an application for me").json()
    assert drafted["mortgage_draft_result"] == "ready"
    assert "calculations" not in drafted
    draft = drafted["mortgage_draft"]
    assert draft["fields"] == {
        "existing_loan_amount": 2000000,
        "property_value": 2500000,
        "property_type": "house",
    }
    assert {k: v["message_id"] for k, v in draft["field_provenance"].items()} == {
        "existing_loan_amount": "amounts",
        "property_value": "amounts",
        "property_type": "scope",
    }
    assert {v["message_id"] for v in draft["intent_provenance"].values()} == {"scope"}

    # The model context never sees client identifiers or raw amounts.
    packed = repr(captured)
    assert "amounts" not in packed and "2000000" not in packed


def test_calculate_without_amounts_asks_for_both_in_chat():
    client = TestClient(make_app(interpreter=chat_first_native()))
    body = chat(client, "s", "calc", "Calculate my loan to value.").json()
    assert body["status"] == "CLARIFY"
    assert body["answer"] == (
        "To calculate your loan-to-value, please provide your existing mortgage "
        "amount and your property value."
    )
    assert "calculations" not in body and "pending_question" not in body


def test_draft_without_refinance_scope_explains_the_supported_draft():
    client = TestClient(make_app(interpreter=chat_first_native()))
    chat(client, "s", "amounts", AMOUNTS)
    body = chat(client, "s", "draft", "Draft an application for me").json()
    assert body["status"] == "CLARIFY"
    assert "mortgage_draft" not in body
    assert body["answer"].startswith("I can prepare a draft only for refinancing")


def test_corrected_amount_recalculates_active_ltv_with_valid_references():
    client = TestClient(make_app(interpreter=chat_first_native()))
    first = chat(
        client,
        "s",
        "one",
        "My mortgage is SEK 1820000 and my property value is SEK 2600000.",
    ).json()
    assert first["calculations"][0]["percent"] == "70.00"
    assert {
        item["message_id"] for item in first["calculations"][0]["input_references"]
    } == {"one"}
    corrected = chat(client, "s", "two", "My mortgage is SEK 1560000.").json()
    assert corrected["calculations"][0]["percent"] == "60.00"
    assert {
        item["message_id"] for item in corrected["calculations"][0]["input_references"]
    } == {"one", "two"}


def test_forgetting_a_source_message_purges_derived_private_bodies():
    async def exercise():
        runtime = MemoryRuntime()
        app = make_app(interpreter=chat_first_native(), runtime=runtime)
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(
            transport=transport, base_url="http://test"
        ) as client:
            for message_id, message in (
                ("amounts", AMOUNTS),
                ("scope", SCOPE),
                ("draft", "Draft an application for me"),
            ):
                response = await client.post(
                    "/api/chat",
                    json={
                        "session_id": "s",
                        "message_id": message_id,
                        "message": message,
                    },
                )
                assert response.status_code == 200
            assert "mortgage_draft" in response.text
            forget = await client.post(
                "/api/sessions/s/forget",
                json={
                    "message_id": "forget-amounts",
                    "scope": {"kind": "messages", "message_ids": ["amounts"]},
                },
            )
            replay = await client.post(
                "/api/chat",
                json={"session_id": "s", "message_id": "amounts", "message": AMOUNTS},
            )
        assert forget.status_code == 200
        assert replay.json()["detail"]["code"] == "FORGOTTEN_TURN"
        bodies = [
            record.response
            for record in (
                *runtime.sessions["s"].chats.values(),
                *runtime.sessions["s"].commands.values(),
            )
            if record.response
        ]
        assert not any("2000000" in str(body) for body in bodies)

    asyncio.run(exercise())


def test_engineering_console_and_its_assets_are_served():
    client = TestClient(make_app(interpreter=chat_first_native()))
    page = client.get("/inspect")
    assert page.status_code == 200 and "text/html" in page.headers["content-type"]
    for asset in ("inspect.js", "inspect.css"):
        assert client.get(f"/inspect/static/{asset}").status_code == 200


def test_console_trace_label_reads_as_english_with_punctuation():
    client = TestClient(make_app(interpreter=chat_first_native()))
    chat(client, "s", "calc", "Calculate my loan to value.")
    (trace,) = client.get("/api/traces").json()
    assert trace["question"] == "Calculate my LTV."
    assert trace["question_projection"] is True


def test_console_trace_label_masks_amounts_and_keeps_commas():
    client = TestClient(make_app(interpreter=chat_first_native()))
    chat(client, "s", "amounts", AMOUNTS)
    (trace,) = client.get("/api/traces").json()
    assert trace["question"] == (
        "What is my LTV, my existing mortgage is ‹amount› "
        "and property is worth ‹amount›"
    )
    assert "2000000" not in trace["question"]


def test_console_trace_label_uses_me_after_a_preposition():
    client = TestClient(make_app(interpreter=chat_first_native()))
    chat(client, "s", "scope", SCOPE)
    chat(client, "s", "draft", "Draft an application for me")
    titles = [trace["question"] for trace in client.get("/api/traces").json()]
    assert (
        "I want to refinance my existing mortgage without borrowing more for my house."
        in titles
    )
    assert "Draft an application for me" in titles


@pytest.mark.parametrize("model_action", ["acknowledge", "clarify"])
def test_explicit_ltv_request_calculates_even_when_the_model_only_acknowledges(
    model_action,
):
    # Observed with local Qwen at temperature 0: the explicit question was
    # classified as acknowledge, so the amounts were stored but no card was shown.
    async def native(memory, turn):
        catalog = build_candidate_catalog(memory, turn)
        return native_result(
            expand_selection(
                catalog,
                TurnSelection(
                    subject="self",
                    conditional=False,
                    requested_action=model_action,
                    fact_candidate_ids=list(catalog.fact_candidates),
                    scope_candidate_ids=[],
                ),
            )
        )

    client = TestClient(make_app(interpreter=native))
    body = chat(client, "s", "amounts", AMOUNTS).json()
    assert body["calculations"][0]["percent"] == "80.00"
    (trace,) = client.get("/api/traces").json()
    detail = client.get(f"/api/traces/{trace['trace_id']}").json()
    assert {
        "name": "explicit_ltv_request",
        "result": "calculate",
        "model_action": model_action,
    } in detail["events"]


def always_calculate_native():
    """Stand in for a model that labels every personal turn as calculate."""

    async def native(memory, turn):
        catalog = build_candidate_catalog(memory, turn)
        return native_result(
            expand_selection(
                catalog,
                TurnSelection(
                    subject="self",
                    conditional=False,
                    requested_action="calculate",
                    fact_candidate_ids=list(catalog.fact_candidates),
                    scope_candidate_ids=list(catalog.scope_candidates),
                ),
            )
        )

    return native


def test_stated_intent_is_acknowledged_even_when_the_model_says_calculate():
    # Observed with OpenAI gpt-4.1: after an LTV card, the refinance intent was
    # labelled calculate and a second card replaced the acknowledgement.
    client = TestClient(make_app(interpreter=always_calculate_native()))
    assert chat(client, "s", "amounts", AMOUNTS).json()["calculations"]
    body = chat(client, "s", "scope", SCOPE).json()
    assert "calculations" not in body
    assert body["answer"] == "I have recorded the supported details."
    events = [
        event
        for trace in client.get("/api/traces").json()
        for event in client.get(f"/api/traces/{trace['trace_id']}").json()["events"]
    ]
    assert {
        "name": "unrequested_calculation",
        "result": "acknowledge",
        "model_action": "calculate",
    } in events


def test_stating_intent_after_ltv_keeps_the_calculation_current():
    client = TestClient(make_app(interpreter=chat_first_native()))
    assert chat(client, "s", "amounts", AMOUNTS).json()["calculations"]
    body = chat(client, "s", "scope", SCOPE).json()
    assert body["memory_receipt"]["invalidated"] == ["offer", "review"]


def test_amount_correction_after_ltv_still_recalculates():
    client = TestClient(make_app(interpreter=always_calculate_native()))
    assert chat(client, "s", "amounts", AMOUNTS).json()["calculations"]
    body = chat(client, "s", "fix", "My property is worth 2600000.").json()
    assert body["calculations"][0]["percent"] == "76.92"


def test_question_first_order_reversed_sentence_calculates_the_same_ltv():
    client = TestClient(make_app(interpreter=chat_first_native()))
    body = chat(
        client,
        "s",
        "reversed",
        "What is my loan to value, my existing mortgage is 2000000 and property is worth 2500000",
    ).json()
    assert body["calculations"][0]["percent"] == "80.00"


def test_plain_amount_statement_is_not_promoted_to_a_calculation():
    async def native(memory, turn):
        catalog = build_candidate_catalog(memory, turn)
        return native_result(
            expand_selection(
                catalog,
                TurnSelection(
                    subject="self",
                    conditional=False,
                    requested_action="acknowledge",
                    fact_candidate_ids=list(catalog.fact_candidates),
                    scope_candidate_ids=[],
                ),
            )
        )

    client = TestClient(make_app(interpreter=native))
    body = chat(
        client,
        "s",
        "one",
        "My mortgage is SEK 1820000 and my property value is SEK 2600000.",
    ).json()
    assert "calculations" not in body
