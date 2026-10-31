"""Bounded RAM-only admission, replay and cancellation for the candidate backend."""

from __future__ import annotations

import asyncio
import hashlib
import json
import time
from dataclasses import dataclass, field

from grounded_mortgage_assistant.state.memory import Memory

SESSION_SECONDS = 1800
TOMBSTONE_SECONDS = 86400
MAX_SESSIONS = 128
MAX_RESERVED = 1024
MAX_CHAT_IDS = 100
MAX_COMMAND_IDS = 128
MAX_BODIES = 16
MAX_BODY_BYTES = 1024 * 1024


class RuntimeErrorCode(ValueError):
    def __init__(self, code: str):
        self.code = code
        super().__init__(code)


def fingerprint(payload: object) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


@dataclass(frozen=True)
class LedgerRecord:
    fingerprint: str | None
    response: str | None
    forgotten: bool = False
    pending: bool = False
    sequence: int = 0
    status: int = 200


@dataclass
class Session:
    memory: Memory = field(default_factory=Memory)
    touched: float = 0.0
    generation: int = 0
    inflight: bool = False
    task: asyncio.Task | None = None
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    chats: dict[str, LedgerRecord] = field(default_factory=dict)
    commands: dict[str, LedgerRecord] = field(default_factory=dict)
    next_sequence: int = 0
    closed: bool = False
    closing: bool = False


@dataclass(frozen=True)
class Tombstone:
    created: float
    reason: str
    message_id: str | None = None
    fingerprint: str | None = None
    response: str | None = None


class MemoryRuntime:
    def __init__(
        self, *, monotonic=time.monotonic, max_sessions: int = MAX_SESSIONS
    ) -> None:
        self.monotonic, self.max_sessions = monotonic, max_sessions
        self.sessions: dict[str, Session] = {}
        self.tombstones: dict[str, Tombstone] = {}
        self._cleanup_tasks: set[asyncio.Task] = set()
        self._reset_tasks: dict[str, asyncio.Task] = {}

    def _expire(self, session_id: str, session: Session, now: float) -> None:
        self.sessions.pop(session_id, None)
        session.closed = True
        session.generation += 1
        task = session.task
        session.inflight, session.task, session.memory = False, None, Memory()
        session.chats, session.commands = {}, {}
        self.tombstones[session_id] = Tombstone(now, "expired")
        if task is not None and task is not asyncio.current_task():
            task.cancel()
            self._cleanup_tasks.add(task)

    def sweep(self) -> None:
        now = self.monotonic()
        for session_id, session in list(self.sessions.items()):
            if now - session.touched >= SESSION_SECONDS:
                self._expire(session_id, session, now)
        self.tombstones = {
            key: value
            for key, value in self.tombstones.items()
            if now - value.created < TOMBSTONE_SECONDS
        }

    async def await_cleanup(self) -> None:
        tasks, self._cleanup_tasks = self._cleanup_tasks, set()
        for task in tasks:
            try:
                await task
            except asyncio.CancelledError:
                pass

    def _session(self, session_id: str, *, create: bool) -> Session:
        self.sweep()
        tombstone = self.tombstones.get(session_id)
        if tombstone:
            raise RuntimeErrorCode(
                "SESSION_RESET" if tombstone.reason == "reset" else "SESSION_EXPIRED"
            )
        session = self.sessions.get(session_id)
        if session is not None and session.closing:
            raise RuntimeErrorCode("SESSION_RESET")
        if session is None and create:
            if (
                len(self.sessions) >= self.max_sessions
                or len(self.sessions) + len(self.tombstones) >= MAX_RESERVED
            ):
                raise RuntimeErrorCode("SESSION_CAPACITY_EXCEEDED")
            session = Session(touched=self.monotonic())
            self.sessions[session_id] = session
        if session is None:
            raise RuntimeErrorCode("SESSION_NOT_FOUND")
        return session

    @staticmethod
    def _lookup(
        ledger: dict[str, LedgerRecord],
        other: dict[str, LedgerRecord],
        identity: str,
        payload: object,
    ) -> str | None:
        if identity in other:
            raise RuntimeErrorCode("IDENTITY_CONFLICT")
        record = ledger.get(identity)
        if record is None:
            return None
        if record.forgotten:
            raise RuntimeErrorCode("FORGOTTEN_TURN")
        if record.fingerprint != fingerprint(payload):
            raise RuntimeErrorCode("IDENTITY_CONFLICT")
        if record.pending:
            raise RuntimeErrorCode("IN_PROGRESS")
        if record.response is None:
            raise RuntimeErrorCode("REPLAY_EXPIRED")
        return record.response

    @classmethod
    def _store(
        cls,
        session: Session,
        ledger: dict[str, LedgerRecord],
        identity: str,
        record: LedgerRecord,
        limit: int,
    ) -> None:
        existing = ledger.get(identity)
        if existing is None and len(ledger) >= limit:
            raise RuntimeErrorCode("IDENTITY_CAPACITY_EXCEEDED")
        if existing is None:
            session.next_sequence += 1
            sequence = session.next_sequence
        else:
            sequence = existing.sequence
        ledger[identity] = LedgerRecord(
            record.fingerprint,
            record.response,
            record.forgotten,
            record.pending,
            sequence,
            record.status,
        )
        bodies = [
            (name, key, item)
            for name, entries in (
                ("chat", session.chats),
                ("command", session.commands),
            )
            for key, item in entries.items()
            if item.response is not None
        ]
        while (
            len(bodies) > MAX_BODIES
            or sum(len(item.response.encode()) for _, _, item in bodies)
            > MAX_BODY_BYTES
        ):
            name, key, prior = min(bodies, key=lambda item: item[2].sequence)
            entries = session.chats if name == "chat" else session.commands
            entries[key] = LedgerRecord(
                prior.fingerprint,
                None,
                prior.forgotten,
                prior.pending,
                prior.sequence,
                prior.status,
            )
            bodies = [item for item in bodies if item[0] != name or item[1] != key]

    async def admit(
        self, session_id: str, message_id: str, payload: object
    ) -> tuple[Session, int, str | None]:
        session = self._session(session_id, create=True)
        async with session.lock:
            replay = self._lookup(session.chats, session.commands, message_id, payload)
            if replay is not None:
                session.touched = self.monotonic()
                return session, session.generation, replay
            if session.inflight:
                raise RuntimeErrorCode("IN_PROGRESS")
            self._store(
                session,
                session.chats,
                message_id,
                LedgerRecord(fingerprint(payload), None, pending=True),
                MAX_CHAT_IDS,
            )
            session.inflight, session.task, session.touched = (
                True,
                asyncio.current_task(),
                self.monotonic(),
            )
            return session, session.generation, None

    async def commit(
        self,
        session: Session,
        generation: int,
        memory: Memory,
        message_id: str,
        payload: object,
        response: tuple[str, ...],
    ) -> bool:
        async with session.lock:
            record = session.chats.get(message_id)
            if (
                session.closed
                or generation != session.generation
                or record is None
                or record.forgotten
                or not record.pending
            ):
                return False
            self._store(
                session,
                session.chats,
                message_id,
                LedgerRecord(fingerprint(payload), response[0]),
                MAX_CHAT_IDS,
            )
            session.memory, session.inflight, session.task, session.touched = (
                memory,
                False,
                None,
                self.monotonic(),
            )
            return True

    async def reject(
        self,
        session: Session,
        generation: int,
        message_id: str,
        payload: object,
        response: str,
        status: int = 200,
    ) -> bool:
        async with session.lock:
            record = session.chats.get(message_id)
            if (
                session.closed
                or generation != session.generation
                or record is None
                or record.forgotten
                or not record.pending
            ):
                return False
            self._store(
                session,
                session.chats,
                message_id,
                LedgerRecord(fingerprint(payload), response, status=status),
                MAX_CHAT_IDS,
            )
            session.inflight, session.task, session.touched = (
                False,
                None,
                self.monotonic(),
            )
            return True

    async def cancel(self, session: Session, generation: int, message_id: str) -> None:
        async with session.lock:
            record = session.chats.get(message_id)
            if (
                generation == session.generation
                and record is not None
                and record.pending
            ):
                session.chats.pop(message_id, None)
                session.inflight, session.task, session.touched = (
                    False,
                    None,
                    self.monotonic(),
                )

    async def command(
        self,
        session_id: str,
        message_id: str,
        payload: object,
        *,
        require_idle: bool = False,
    ) -> tuple[Session, str | None, bool, int]:
        session = self._session(session_id, create=False)
        async with session.lock:
            if (
                session.closed
                or session.closing
                or self.sessions.get(session_id) is not session
            ):
                raise RuntimeErrorCode("SESSION_RESET")
            replay = self._lookup(session.commands, session.chats, message_id, payload)
            if replay is not None:
                return session, replay, False, session.generation
            if require_idle and session.inflight:
                raise RuntimeErrorCode("IN_PROGRESS")
            self._store(
                session,
                session.commands,
                message_id,
                LedgerRecord(fingerprint(payload), None, pending=True),
                MAX_COMMAND_IDS,
            )
            return session, None, True, session.generation

    async def finish_command(
        self,
        session: Session,
        message_id: str,
        payload: object,
        response: str,
        *,
        status: int = 200,
        generation: int | None = None,
    ) -> bool:
        async with session.lock:
            if session.closed or (
                generation is not None and generation != session.generation
            ):
                return False
            self._store(
                session,
                session.commands,
                message_id,
                LedgerRecord(fingerprint(payload), response, status=status),
                MAX_COMMAND_IDS,
            )
            session.touched = self.monotonic()
            return True

    async def close(
        self,
        session_id: str,
        *,
        forgotten: bool,
        reset_id: str | None = None,
        reset_payload: object | None = None,
        reset_response: str | None = None,
    ) -> Session | None:
        session = self.sessions.get(session_id)
        if session is None:
            return None
        if session.closing:
            return session
        # Reserve the identity before waiting for the per-session lock.  This
        # prevents a same-id admission from appearing between pop and tombstone.
        session.closing = True
        async with session.lock:
            if self.sessions.get(session_id) is session:
                self.sessions.pop(session_id, None)
            session.generation += 1
            session.closed = True
            task = session.task
            session.inflight, session.task = False, None
            if forgotten:
                session.chats = {
                    key: LedgerRecord(
                        None, None, True, sequence=record.sequence, status=record.status
                    )
                    for key, record in session.chats.items()
                }
            session.memory = Memory()
            session.chats, session.commands = {}, {}
        self.tombstones[session_id] = Tombstone(
            self.monotonic(),
            "reset" if reset_id else "expired",
            reset_id,
            fingerprint(reset_payload) if reset_payload is not None else None,
            reset_response,
        )
        if task is not None and task is not asyncio.current_task():
            task.cancel()
            self._cleanup_tasks.add(task)
        await self.await_cleanup()
        return session

    async def reset(
        self,
        session_id: str,
        message_id: str | None = None,
        payload: object | None = None,
        response: str | None = None,
    ) -> str | None:
        self.sweep()
        pending = self._reset_tasks.get(session_id)
        if pending is not None:
            try:
                await asyncio.shield(pending)
            except asyncio.CancelledError:
                await asyncio.shield(pending)
                raise
            # A completed task's callback may still be queued. Resolve its
            # receipt below without recursively awaiting the same done task.
            if self._reset_tasks.get(session_id) is pending:
                self._reset_tasks.pop(session_id, None)
        tombstone = self.tombstones.get(session_id)
        if tombstone:
            if tombstone.reason == "reset":
                payload_fingerprint = (
                    fingerprint(payload) if payload is not None else None
                )
                if tombstone.message_id == message_id:
                    if tombstone.fingerprint != payload_fingerprint:
                        raise RuntimeErrorCode("IDENTITY_CONFLICT")
                    return tombstone.response
                # A reset is idempotent for its own command identity. A later,
                # distinct reset can still acknowledge the already-closed
                # session, without replacing that original replay receipt.
                return response
            raise RuntimeErrorCode(
                "SESSION_EXPIRED" if tombstone.reason == "expired" else "SESSION_RESET"
            )
        session = self.sessions.get(session_id)
        if session is None:
            if len(self.tombstones) >= MAX_RESERVED:
                raise RuntimeErrorCode("SESSION_CAPACITY_EXCEEDED")
            self.tombstones[session_id] = Tombstone(
                self.monotonic(),
                "reset",
                message_id,
                fingerprint(payload) if payload is not None else None,
                response,
            )
            return response
        async with session.lock:
            # Another reset may have reserved cleanup while this caller was
            # queued for the lock. Registration belongs to the same critical
            # section as this recheck, before either caller can await cleanup.
            pending = self._reset_tasks.get(session_id)
            if pending is None:
                if message_id in session.chats or message_id in session.commands:
                    raise RuntimeErrorCode("IDENTITY_CONFLICT")
                if len(self.tombstones) >= MAX_RESERVED:
                    raise RuntimeErrorCode("SESSION_CAPACITY_EXCEEDED")
                self.tombstones[session_id] = Tombstone(
                    self.monotonic(),
                    "reset",
                    message_id,
                    fingerprint(payload) if payload is not None else None,
                    response,
                )
                # Runtime-level callers can own their synthetic admitted chat.
                # The detached closer must not cancel its own waiting caller.
                if session.task is asyncio.current_task():
                    session.inflight, session.task = False, None
                task = asyncio.create_task(
                    self.close(
                        session_id,
                        forgotten=False,
                        reset_id=message_id,
                        reset_payload=payload,
                        reset_response=response,
                    )
                )
                self._reset_tasks[session_id] = task

                def finished(done: asyncio.Task) -> None:
                    if self._reset_tasks.get(session_id) is done:
                        self._reset_tasks.pop(session_id, None)

                task.add_done_callback(finished)
            else:
                task = pending
        try:
            await asyncio.shield(task)
        except asyncio.CancelledError:
            await asyncio.shield(task)
            raise
        if pending is not None:
            # Resolve this caller's identity against the completed reset. A
            # different payload must not inherit the first caller's receipt.
            return await self.reset(session_id, message_id, payload, response)
        return response
