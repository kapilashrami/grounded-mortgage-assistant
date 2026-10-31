import asyncio

import pytest

from grounded_mortgage_assistant.state.memory import Memory
from grounded_mortgage_assistant.state.runtime import (
    MAX_BODIES,
    MemoryRuntime,
    RuntimeErrorCode,
)


def test_replay_conflict_eviction_and_cross_namespace_identity():
    async def exercise():
        runtime = MemoryRuntime(monotonic=lambda: 0)
        session, generation, replay = await runtime.admit("s", "chat", {"a": 1})
        assert replay is None
        assert await runtime.commit(
            session, generation, Memory(), "chat", {"a": 1}, ('{"ok":true}',)
        )
        _, _, replay = await runtime.admit("s", "chat", {"a": 1})
        assert replay == '{"ok":true}'
        with pytest.raises(RuntimeErrorCode, match="IDENTITY_CONFLICT"):
            await runtime.admit("s", "chat", {"a": 2})
        with pytest.raises(RuntimeErrorCode, match="IDENTITY_CONFLICT"):
            await runtime.command("s", "chat", {"command": 1})

    asyncio.run(exercise())


def test_reset_checks_existing_identities_and_expiry_precedes_reset_replay():
    now = [0.0]

    async def exercise():
        runtime = MemoryRuntime(monotonic=lambda: now[0])
        session, generation, _ = await runtime.admit("s", "chat", {"a": 1})
        await runtime.commit(session, generation, Memory(), "chat", {"a": 1}, ("{}",))
        with pytest.raises(RuntimeErrorCode, match="IDENTITY_CONFLICT"):
            await runtime.reset("s", "chat", {}, "{}")
        await runtime.finish_command(session, "forget", {"scope": "all"}, "{}")
        with pytest.raises(RuntimeErrorCode, match="IDENTITY_CONFLICT"):
            await runtime.reset("s", "forget", {}, "{}")
        await runtime.reset("s", "reset", {"id": "reset"}, '{"ok":true}')
        assert (
            await runtime.reset("s", "reset", {"id": "reset"}, "ignored")
            == '{"ok":true}'
        )

        session, _, _ = await runtime.admit("expired", "one", {})
        now[0] = 1801
        with pytest.raises(RuntimeErrorCode, match="SESSION_EXPIRED"):
            await runtime.reset("expired", "reset", {}, "{}")
        assert session.closed
        now[0] += 86401
        # Once the expiry tombstone is swept, Start over still acknowledges the
        # unknown session (and reserves a reset tombstone) instead of failing.
        assert (
            await runtime.reset("expired", "reset", {}, '{"ok":true}') == '{"ok":true}'
        )
        assert runtime.tombstones["expired"].reason == "reset"

    asyncio.run(exercise())


def test_combined_fifo_bodies_and_sweep_cleanup_clear_private_memory():
    now = [0.0]
    finalized = asyncio.Event()

    async def delayed():
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            finalized.set()
            raise

    async def exercise():
        runtime = MemoryRuntime(monotonic=lambda: now[0])
        session, generation, _ = await runtime.admit("s", "first", {})
        await runtime.commit(session, generation, Memory(), "first", {}, ("one",))
        for index in range(MAX_BODIES + 1):
            await runtime.finish_command(session, f"cmd{index}", {"i": index}, "body")
        assert session.chats["first"].response is None

        session.task = asyncio.create_task(delayed())
        await asyncio.sleep(0)
        session.inflight = True
        now[0] = 1801
        runtime.sweep()
        assert session.closed and session.memory == Memory()
        assert not session.chats and not session.commands
        await runtime.await_cleanup()
        assert finalized.is_set()

    asyncio.run(exercise())


def test_clock_sweep_expires_and_late_generation_cannot_commit():
    now = [0.0]

    async def exercise():
        runtime = MemoryRuntime(monotonic=lambda: now[0])
        session, generation, _ = await runtime.admit("s", "one", {})
        now[0] = 1801
        runtime.sweep()
        assert "s" not in runtime.sessions
        assert not await runtime.commit(
            session, generation + 1, Memory(), "one", {}, ("{}",)
        )
        with pytest.raises(RuntimeErrorCode, match="SESSION_EXPIRED"):
            await runtime.admit("s", "two", {})

    asyncio.run(exercise())


def test_late_cancellation_cannot_clear_a_new_generation_reservation():
    async def exercise():
        runtime = MemoryRuntime()
        session, generation, _ = await runtime.admit("s", "one", {})
        await runtime.cancel(session, generation, "one")
        _, next_generation, _ = await runtime.admit("s", "two", {})
        await runtime.cancel(session, generation, "one")
        assert session.inflight
        assert session.chats["two"].pending
        await runtime.cancel(session, next_generation, "two")

    asyncio.run(exercise())


def test_close_reserves_identity_before_waiting_for_old_session_lock():
    async def exercise():
        runtime = MemoryRuntime()
        session, generation, _ = await runtime.admit("s", "one", {})
        session.inflight, session.task = False, None
        await session.lock.acquire()
        closing = asyncio.create_task(
            runtime.close("s", forgotten=False, reset_id="reset", reset_payload={})
        )
        await asyncio.sleep(0)
        with pytest.raises(RuntimeErrorCode, match="SESSION_RESET"):
            await runtime.admit("s", "replacement", {})
        session.lock.release()
        await closing
        assert not await runtime.commit(
            session, generation, Memory(), "one", {}, ("{}",)
        )
        with pytest.raises(RuntimeErrorCode, match="SESSION_RESET"):
            await runtime.admit("s", "replacement", {})

    asyncio.run(exercise())


def test_reset_callers_queued_on_lock_share_cleanup_before_any_ack():
    async def exercise():
        runtime = MemoryRuntime()
        session, _, _ = await runtime.admit("s", "chat", {})
        started, cleaning, release = asyncio.Event(), asyncio.Event(), asyncio.Event()

        async def generation():
            started.set()
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                cleaning.set()
                await release.wait()

        session.task = asyncio.create_task(generation())
        await started.wait()
        await session.lock.acquire()
        first = asyncio.create_task(runtime.reset("s", "reset", {}, "receipt"))
        second = asyncio.create_task(runtime.reset("s", "reset", {}, "receipt"))
        await asyncio.sleep(0)
        session.lock.release()
        await cleaning.wait()
        # Both queued callers have had a chance to pass through the old lock.
        for _ in range(5):
            await asyncio.sleep(0)
        replay = asyncio.create_task(runtime.reset("s", "reset", {}, "receipt"))
        await asyncio.sleep(0)
        premature = first.done() or second.done() or replay.done()
        release.set()
        results = await asyncio.gather(first, second, replay)
        assert not premature
        assert results == ["receipt"] * 3

    asyncio.run(exercise())
