"""FastAPI app: chat, sources, session commands and the engineering console."""

from __future__ import annotations

import asyncio
import json
import re
import time
from collections import deque
from collections.abc import Awaitable, Callable
from contextlib import asynccontextmanager
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

import httpx
from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from langsmith.run_helpers import tracing_context

from grounded_mortgage_assistant.config import (
    corpus_name,
    not_found_answer,
    tokenizer_path,
    unavailable_answer,
)
from grounded_mortgage_assistant.contracts import (
    ForgetRequest,
    MemoryChatRequest,
    ResetRequest,
)
from grounded_mortgage_assistant.guardrails.pii import (
    PiiCheckUnavailable,
    PiiDecision,
    check_pii,
)
from grounded_mortgage_assistant.guardrails.privacy import (
    IngressError,
    detector_text_with_approved_amounts,
    ingest,
)
from grounded_mortgage_assistant.llm.conversation import (
    LOCAL_MODEL,
    LOCAL_URL,
    Provider,
    interpret_turn,
    provider_from_env,
    select_evidence,
)
from grounded_mortgage_assistant.rag.answer import answer_public_question
from grounded_mortgage_assistant.rag.public_evidence import PublicEvidence
from grounded_mortgage_assistant.rag.public_questions import PublicQuestionScope
from grounded_mortgage_assistant.rag.retrieval import FrozenRetriever
from grounded_mortgage_assistant.state.memory import (
    MemoryError,
    StageResult,
    analyze_financial_turn,
    forget,
    record_control_event,
)
from grounded_mortgage_assistant.state.runtime import (
    LedgerRecord,
    MemoryRuntime,
    RuntimeErrorCode,
    fingerprint,
)
from grounded_mortgage_assistant.tracing import (
    TRACE_EVENTS,
    clarify_decision,
    trace_event,
)
from grounded_mortgage_assistant.workflow.graph import (
    build_graph,
    complete_memory_draft,
    memory_receipt,
)


def _error(
    code: str, *, status: int = 409, retry: str = "new_message"
) -> HTTPException:
    return HTTPException(status, {"code": code, "retry": retry})


# Punctuation tokens add noise to the console's safe question label.
_TRACE_BOUNDARY_TOKENS = frozenset({"SENTENCE_END", "QUESTION", "EXCLAMATION", "COMMA"})


def _is_draft_command(turn: Any) -> bool:
    symbols = tuple(token for token in turn.model_projection if token != "SENTENCE_END")
    if symbols[:1] == ("PLEASE",):
        symbols = symbols[1:]
    return symbols in {
        ("DRAFT",),
        ("DRAFT", "APPLICATION"),
        ("DRAFT", "A", "APPLICATION"),
        ("DRAFT", "APPLICATION", "FOR", "SELF"),
        ("DRAFT", "A", "APPLICATION", "FOR", "SELF"),
    }


def _envelope(body: dict[str, Any], session_id: str, message_id: str) -> dict[str, Any]:
    return {
        "citations": [],
        "related_articles": [],
        "actions": [],
        **body,
        "trace_id": str(uuid4()),
        "session_id": session_id,
        "message_id": message_id,
    }


def _private_replay_is_current(body: dict[str, Any], session: Any) -> bool:
    """Never replay a fact-derived result after its memory basis has moved."""
    if "calculations" not in body and "mortgage_draft" not in body:
        return True
    receipt = body.get("memory_receipt")
    if (
        not isinstance(receipt, dict)
        or receipt.get("revision") != session.memory.revision
    ):
        return False
    draft = body.get("mortgage_draft")
    if not isinstance(draft, dict):
        return True
    try:
        return datetime.fromisoformat(draft["expires_at"]) > datetime.now(UTC)
    except KeyError, TypeError, ValueError:
        return False


async def verify_local_provider(
    *, transport: httpx.AsyncBaseTransport | None = None
) -> int:
    """Verify only the pinned local model and capacity, without inference."""
    try:
        async with httpx.AsyncClient(
            transport=transport, trust_env=False, follow_redirects=False, timeout=5
        ) as client:
            response = await client.get(
                f"{LOCAL_URL}/models", headers={"Authorization": "Bearer local"}
            )
        if response.status_code != 200 or response.is_redirect:
            raise ValueError("provider status")
        data = response.json().get("data")
        if not isinstance(data, list):
            raise TypeError("provider data")
        entry = next(
            (
                item
                for item in data
                if isinstance(item, dict) and item.get("id") == LOCAL_MODEL
            ),
            None,
        )
        capacity = entry.get("max_model_len") if entry else None
        if type(capacity) is not int or capacity <= 0 or capacity < 8192 + 1800:
            raise ValueError("provider capacity")
        return capacity
    except (httpx.HTTPError, TypeError, ValueError, KeyError) as error:
        raise RuntimeError("LOCAL_PROVIDER_STARTUP_INVALID") from error


# USD per million tokens (input, output), OpenAI list prices checked 31 October 2026.
OPENAI_PRICES = {
    "gpt-4.1": (2.00, 8.00),
    "gpt-4.1-mini": (0.40, 1.60),
    "gpt-4.1-nano": (0.10, 0.40),
    "gpt-4o-mini": (0.15, 0.60),
    "gpt-5.4-mini": (0.75, 4.50),
}


async def verify_openai_provider(
    provider: Provider, *, transport: httpx.AsyncBaseTransport | None = None
) -> None:
    """Verify the key can see the configured model, without inference."""
    try:
        async with httpx.AsyncClient(
            transport=transport, trust_env=False, follow_redirects=False, timeout=10
        ) as client:
            response = await client.get(
                f"{provider.base_url}/models/{provider.model}",
                headers={"Authorization": f"Bearer {provider.api_key}"},
            )
        if response.status_code != 200 or response.json().get("id") != provider.model:
            raise ValueError("provider model")
    except (httpx.HTTPError, TypeError, ValueError) as error:
        raise RuntimeError("OPENAI_PROVIDER_STARTUP_INVALID") from error


def model_cost(
    provider: Provider, input_tokens: int | None, output_tokens: int | None
) -> dict[str, Any]:
    if provider.name == "local":
        return {
            "external_api_usd": 0,
            "hardware": None,
            "note": "Local hardware cost is unmeasured.",
        }
    price = OPENAI_PRICES.get(provider.model)
    if price is None or input_tokens is None or output_tokens is None:
        return {
            "external_api_usd": None,
            "hardware": None,
            "note": "Hosted OpenAI usage; no list price is configured for this model.",
        }
    usd = (input_tokens * price[0] + output_tokens * price[1]) / 1_000_000
    return {
        "external_api_usd": round(usd, 6),
        "hardware": None,
        "note": "Estimate at the 31 October 2026 OpenAI list price.",
    }


def create_app(
    *,
    interpreter: Any | None = None,
    retriever: Any | None = None,
    selector: Any | None = None,
    runtime: MemoryRuntime | None = None,
    evidence: PublicEvidence | None = None,
    pii_checker: Callable[[str], Awaitable[PiiDecision]] | None = None,
    provider: Provider | None = None,
) -> FastAPI:
    provider = provider or provider_from_env()
    runtime = runtime or MemoryRuntime()
    evidence = evidence or PublicEvidence()
    checker = pii_checker or check_pii
    public_questions = PublicQuestionScope(evidence.records)
    graph = build_graph()
    traces: deque[dict[str, Any]] = deque(maxlen=100)

    def bounded(value: object, limit: int) -> tuple[str, bool]:
        text = value if isinstance(value, str) else ""
        return (text[:limit], len(text) > limit)

    def redact_question(value: object) -> str:
        question, clipped = bounded(value, 1000)
        question = re.sub(
            r"[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}", "[redacted email]", question
        )
        question = re.sub(r"(?:\+?\d[\s-]?){8,}\d", "[redacted phone]", question)
        return f"{question} [clipped]" if clipped else question

    def purge_session_traces(session_id: str) -> None:
        retained = [row for row in traces if row["session_id"] != session_id]
        traces.clear()
        traces.extend(retained)

    def prune_traces() -> None:
        now = runtime.monotonic()
        retained = [row for row in traces if now - row["created_monotonic"] < 1800]
        if len(retained) != len(traces):
            traces.clear()
            traces.extend(retained)

    def sweep_runtime() -> None:
        active = set(runtime.sessions)
        runtime.sweep()
        for session_id in active - set(runtime.sessions):
            purge_session_traces(session_id)
        prune_traces()

    def record_trace(
        *,
        trace_id: str,
        session_id: str,
        session: Any,
        generation: int,
        message_id: str,
        request_kind: str,
        status: str,
        question: object,
        answer: object,
        started: float,
        events: list[dict[str, Any]] | None = None,
        evidence_count: int = 0,
        error: str | None = None,
    ) -> dict[str, Any] | None:
        if (
            session.closed
            or runtime.sessions.get(session_id) is not session
            or session.generation != generation
        ):
            return None
        safe_answer, answer_clipped = bounded(answer, 16_000)
        safe_events = [
            {
                key: value
                for key, value in event.items()
                if key not in {"question", "model_projection", "callback", "payload"}
            }
            for event in events or []
        ]
        provider_events = [
            event
            for event in safe_events
            if event.get("name") in {"interpret_turn", "select_evidence"}
        ]
        local_inputs = [
            event.get("input_tokens")
            if isinstance(event.get("input_tokens"), int)
            and not isinstance(event.get("input_tokens"), bool)
            else None
            for event in provider_events
        ]
        provider_inputs = [
            event.get("provider_input_tokens")
            if isinstance(event.get("provider_input_tokens"), int)
            and not isinstance(event.get("provider_input_tokens"), bool)
            else None
            for event in provider_events
        ]
        if not provider_events:
            input_tokens, input_source = 0, "no model calls"
        elif all(value is not None for value in provider_inputs):
            input_tokens, input_source = sum(provider_inputs), "provider usage"
        elif all(value is not None for value in local_inputs):
            input_tokens = sum(
                provider if provider is not None else local
                for provider, local in zip(provider_inputs, local_inputs, strict=True)
            )
            input_source = "partial: provider usage and local tokenizer estimate"
        else:
            input_tokens, input_source = None, "partial: input usage unavailable"
        output_values = [event.get("output_tokens") for event in provider_events]
        output_tokens = (
            0
            if not provider_events
            else sum(value for value in output_values if isinstance(value, int))
            if all(
                isinstance(value, int) and not isinstance(value, bool)
                for value in output_values
            )
            else None
        )
        output_source = (
            "no model calls"
            if not provider_events
            else "provider usage"
            if output_tokens is not None
            else "partial: output usage unavailable"
        )
        row = {
            "trace_id": trace_id,
            "session_id": session_id,
            "message_id": message_id,
            "request_kind": request_kind,
            "status": status,
            "question": redact_question(question),
            "answer": safe_answer,
            "answer_clipped": answer_clipped,
            "error": error,
            "duration_ms": int((time.monotonic() - started) * 1000),
            "created_at": datetime.now(UTC).isoformat(),
            "created_monotonic": runtime.monotonic(),
            "events": safe_events,
            "retrieval": {"count": evidence_count},
            "usage": {
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "input_token_source": input_source,
                "output_token_source": output_source,
            },
            "cost": model_cost(provider, input_tokens, output_tokens),
            "model": {"id": provider.model, "provider": provider.name},
        }
        traces.append(row)
        return row

    tokenizer: Any | None = None
    frozen: FrozenRetriever | None = None
    retrieval_lock = asyncio.Lock()
    sweeper: asyncio.Task | None = None
    brand = corpus_name()

    async def production_interpreter(memory: Any, turn: Any) -> Any:
        if tokenizer is None:
            raise RuntimeError("offline tokenizer unavailable")
        return await interpret_turn(
            memory,
            turn,
            tokenizer=tokenizer,
            provider_capacity=262144,
            provider=provider,
        )

    async def production_retriever(query: str) -> list[dict[str, Any]]:
        if frozen is None:
            raise RuntimeError("frozen retriever unavailable")
        started = time.monotonic()
        async with retrieval_lock:
            result = await asyncio.to_thread(frozen.retrieve, query)
        trace_event(
            {
                "name": "frozen_retrieval",
                "result": "success",
                "final_ids": [row["id"] for row in result.records],
                "count": len(result.records),
                "duration_ms": int((time.monotonic() - started) * 1000),
            }
        )
        return result.records

    async def production_selector(question: str, records: list[dict[str, Any]]) -> Any:
        if tokenizer is None:
            raise RuntimeError("offline tokenizer unavailable")
        return await select_evidence(
            question,
            records,
            tokenizer=tokenizer,
            provider_capacity=262144,
            provider=provider,
        )

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        nonlocal frozen, sweeper, tokenizer
        if interpreter is None:
            from transformers import AutoTokenizer

            local_tokenizer = tokenizer_path()
            if not local_tokenizer.is_dir():
                raise RuntimeError("offline tokenizer unavailable")
            tokenizer = AutoTokenizer.from_pretrained(
                local_tokenizer, local_files_only=True, trust_remote_code=False
            )
            if provider.name == "local":
                app.state.provider_capacity = await verify_local_provider()
            else:
                await verify_openai_provider(provider)
            frozen = FrozenRetriever()

        async def sweep_loop() -> None:
            while True:
                await asyncio.sleep(60)
                sweep_runtime()
                await runtime.await_cleanup()

        sweeper = asyncio.create_task(sweep_loop())
        try:
            yield
        finally:
            sweeper.cancel()
            try:
                await sweeper
            except asyncio.CancelledError:
                pass

    app = FastAPI(title="Grounded mortgage assistant", lifespan=lifespan)
    app.state.provider_capacity = None
    app.mount(
        "/inspect/static",
        StaticFiles(directory=Path(__file__).parent / "console/static"),
        name="inspect-static",
    )

    @app.get("/inspect")
    async def inspect() -> FileResponse:
        return FileResponse(Path(__file__).parent / "console/static/inspect.html")

    @app.get("/api/traces")
    async def list_traces(response: Response) -> list[dict[str, Any]]:
        response.headers["Cache-Control"] = "no-store"
        sweep_runtime()
        return [
            {
                key: row[key]
                for key in (
                    "trace_id",
                    "session_id",
                    "message_id",
                    "request_kind",
                    "status",
                    "question",
                    "duration_ms",
                    "created_at",
                    "usage",
                    "cost",
                    "retrieval",
                )
            }
            for row in reversed(traces)
        ]

    @app.get("/api/traces/{trace_id}")
    async def trace(trace_id: str, response: Response) -> dict[str, Any]:
        response.headers["Cache-Control"] = "no-store"
        sweep_runtime()
        for row in traces:
            if row["trace_id"] == trace_id:
                return row
        raise HTTPException(404, "Trace not found")

    @app.exception_handler(RequestValidationError)
    async def invalid_request(_: Request, __: RequestValidationError):
        # Do not return Pydantic's input echo for this privacy boundary.
        return JSONResponse(
            status_code=422,
            content={"detail": {"code": "INVALID_REQUEST", "retry": "none"}},
        )

    @app.get("/health")
    async def health() -> dict[str, Any]:
        return {
            "status": "ok",
            "mode": "candidate_memory",
            "provider": "local_only" if provider.name == "local" else provider.name,
            "model": provider.model,
            "provider_capacity": app.state.provider_capacity,
            "input_limit": 8192,
            "output_reservation": 1800,
            "records": len(evidence.records),
        }

    @app.get("/api/scope")
    async def scope() -> dict[str, Any]:
        return {
            "mode": "candidate_memory",
            "records": len(evidence.records),
            "hosted_fallback": False,
            "live_cutover": False,
            "provider": provider.name,
            "model": provider.model,
            "provider_capacity": app.state.provider_capacity,
            "input_limit": 8192,
            "output_reservation": 1800,
        }

    @app.get("/api/sources/{source_id}.pdf")
    async def source(source_id: str, sha256: str) -> FileResponse:
        path = evidence.pdf(source_id, sha256)
        if path is None:
            raise HTTPException(404, "Source not found")
        return FileResponse(path, media_type="application/pdf")

    @app.post("/api/chat")
    async def chat(request: MemoryChatRequest) -> dict[str, Any]:
        payload = request.model_dump()
        try:
            session, generation, replay = await runtime.admit(
                request.session_id, request.message_id, payload
            )
        except RuntimeErrorCode as error:
            code = error.code
            raise _error(
                code,
                status=410
                if code in {"SESSION_EXPIRED", "SESSION_RESET"}
                else 429
                if code.endswith("CAPACITY_EXCEEDED")
                else 409,
            ) from error
        if replay is not None:
            # Stored response bodies are server-owned JSON fragments.
            stored = session.chats[request.message_id]
            body = json.loads(replay)
            if not _private_replay_is_current(body, session):
                raise _error("REPLAY_EXPIRED")
            if stored.status != 200:
                return JSONResponse(content=body, status_code=stored.status)
            return body
        finished = False
        started = time.monotonic()
        trace_token = TRACE_EVENTS.set([])
        trace_question = ""
        try:
            detector_text = request.message
            try:
                detector_turn = ingest(request.message)
                approved_refs = tuple(
                    value_ref
                    for clause in analyze_financial_turn(
                        session.memory, detector_turn
                    ).clauses
                    if clause.kind in {"affirmed", "alternatives", "negated"}
                    for value_ref in clause.value_refs
                )
                if approved_refs:
                    detector_text = detector_text_with_approved_amounts(
                        detector_turn, approved_refs
                    )
            except IngressError:
                pass
            try:
                pii_decision = await checker(detector_text)
                if pii_decision.pii_detected and pii_decision.category is None:
                    raise PiiCheckUnavailable
            except PiiCheckUnavailable as error:
                trace_event(
                    {
                        "name": "privacy",
                        "result": "unavailable",
                        "reason": "detector_unavailable",
                    }
                )
                rejected = {
                    "detail": {"code": "PII_CHECK_UNAVAILABLE", "retry": "new_message"}
                }
                finished = await runtime.reject(
                    session,
                    generation,
                    request.message_id,
                    payload,
                    json.dumps(rejected, separators=(",", ":")),
                    503,
                )
                if finished:
                    record_trace(
                        trace_id=str(uuid4()),
                        session_id=request.session_id,
                        session=session,
                        generation=generation,
                        message_id=request.message_id,
                        request_kind="chat",
                        status="REJECTED",
                        question="",
                        answer="",
                        started=started,
                        events=TRACE_EVENTS.get() or [],
                        error="PII_CHECK_UNAVAILABLE",
                    )
                raise _error("PII_CHECK_UNAVAILABLE", status=503) from error
            if pii_decision.pii_detected:
                trace_event(
                    {
                        "name": "privacy",
                        "result": "blocked",
                        "reason": pii_decision.category,
                    }
                )
                rejected = {
                    "detail": {"code": "SENSITIVE_TEXT", "retry": "new_message"}
                }
                finished = await runtime.reject(
                    session,
                    generation,
                    request.message_id,
                    payload,
                    json.dumps(rejected, separators=(",", ":")),
                    422,
                )
                if finished:
                    record_trace(
                        trace_id=str(uuid4()),
                        session_id=request.session_id,
                        session=session,
                        generation=generation,
                        message_id=request.message_id,
                        request_kind="chat",
                        status="REJECTED",
                        question="",
                        answer="",
                        started=started,
                        events=TRACE_EVENTS.get() or [],
                        error="SENSITIVE_TEXT",
                    )
                raise _error("SENSITIVE_TEXT", status=422) from None
            trace_event({"name": "privacy", "result": "pass"})
            if public_questions.is_public_question(request.message):
                control = record_control_event(
                    session.memory,
                    client_message_id=request.message_id,
                    control_ref=f"control_{uuid4().hex}",
                )
                staged = StageResult(control, (), ())
                answer = await answer_public_question(
                    request.message,
                    retriever=retriever or production_retriever,
                    selector=selector or production_selector,
                    evidence=evidence,
                    scope=public_questions,
                )
                top_score = (
                    round(answer.top_score, 4) if answer.top_score is not None else None
                )
                trace_event(
                    {
                        "name": "public_projection",
                        "kind": {
                            "answered": "matched",
                            "unsupported": "unsupported",
                            "unavailable": "unavailable",
                        }[answer.status],
                        "record_id": answer.top_record_id,
                        "top_score": top_score,
                    }
                )
                if answer.status == "answered":
                    # Trace only the corpus question, never the raw customer text.
                    trace_question = evidence.records[answer.top_record_id]["question"]
                    result = {
                        "memory": control,
                        "staged": staged,
                        "public_knowledge": True,
                        "response": {
                            "status": "ANSWERED",
                            **answer.rendered,
                            "memory_receipt": memory_receipt(control, staged),
                        },
                    }
                elif answer.status == "unsupported":
                    clarify_decision("public_projection", "unsupported")
                    result = {
                        "memory": control,
                        "staged": staged,
                        "public_knowledge": True,
                        "response": {
                            "status": "CLARIFY",
                            "answer": not_found_answer(brand),
                            "memory_receipt": memory_receipt(control, staged),
                        },
                    }
                else:
                    clarify_decision("evidence", "evidence_unavailable")
                    result = {
                        "memory": control,
                        "staged": staged,
                        "public_knowledge": True,
                        "response": {
                            "status": "CLARIFY",
                            "answer": unavailable_answer(brand),
                            "memory_receipt": memory_receipt(control, staged),
                            "error": {
                                "code": "EVIDENCE_UNAVAILABLE",
                                "retry": "new_message",
                            },
                        },
                    }
            else:
                turn = ingest(request.message)
                trace_question = " ".join(
                    "AMOUNT" if token.startswith("VALUE(") else token
                    for token in turn.model_projection
                    if token not in _TRACE_BOUNDARY_TOKENS
                )
                if _is_draft_command(turn):
                    control = record_control_event(
                        session.memory,
                        client_message_id=request.message_id,
                        control_ref=f"control_{uuid4().hex}",
                    )
                    staged = StageResult(control, (), ())
                    requested = replace(
                        control, draft_requested=True, active_operation=None
                    )
                    memory, draft_response = complete_memory_draft(requested, staged)
                    result = {"memory": memory, "response": draft_response}
                else:
                    with tracing_context(enabled=False):
                        result = await graph.ainvoke(
                            {
                                "memory": session.memory,
                                "turn": turn,
                                "message_id": request.message_id,
                                "interpreter": interpreter or production_interpreter,
                                "retriever": retriever or production_retriever,
                                "selector": selector or production_selector,
                                "evidence": evidence,
                                "public_questions": public_questions,
                            },
                            config={"callbacks": []},
                        )
            response = _envelope(
                result["response"], request.session_id, request.message_id
            )
            finished = await runtime.commit(
                session,
                generation,
                result.get("memory", session.memory),
                request.message_id,
                payload,
                (json.dumps(response, separators=(",", ":")),),
            )
            if not finished:
                raise _error("SESSION_RESET")
            record_trace(
                trace_id=response["trace_id"],
                session_id=request.session_id,
                session=session,
                generation=generation,
                message_id=request.message_id,
                request_kind="chat",
                status=response["status"],
                question=trace_question,
                answer=response.get("answer"),
                started=started,
                events=[
                    *(TRACE_EVENTS.get() or []),
                    {"name": "memory_commit", "result": "success"},
                ],
                evidence_count=len(response.get("citations", [])),
            )
            return response
        except IngressError as error:
            privacy_event = {"name": "privacy", "result": "blocked"}
            if error.code == "SENSITIVE_TEXT":
                privacy_event["reason"] = "personal_information"
            trace_event(privacy_event)
            rejected = {"detail": {"code": error.code, "retry": "new_message"}}
            finished = await runtime.reject(
                session,
                generation,
                request.message_id,
                payload,
                json.dumps(rejected, separators=(",", ":")),
                422,
            )
            if finished:
                record_trace(
                    trace_id=str(uuid4()),
                    session_id=request.session_id,
                    session=session,
                    generation=generation,
                    message_id=request.message_id,
                    request_kind="chat",
                    status="REJECTED",
                    question="",
                    answer="",
                    started=started,
                    events=TRACE_EVENTS.get() or [],
                    error=error.code,
                )
            raise _error(error.code, status=422, retry="new_message") from error
        except asyncio.CancelledError:
            trace_event({"name": "request", "result": "cancelled"})
            raise
        except HTTPException:
            raise
        except Exception as error:
            trace_event(
                {
                    "name": "request",
                    "result": "failure",
                    "category": type(error).__name__,
                }
            )
            finished = await runtime.reject(
                session,
                generation,
                request.message_id,
                payload,
                json.dumps(
                    {"detail": {"code": "PROCESSING_FAILED", "retry": "new_message"}},
                    separators=(",", ":"),
                ),
                409,
            )
            if finished:
                record_trace(
                    trace_id=str(uuid4()),
                    session_id=request.session_id,
                    session=session,
                    generation=generation,
                    message_id=request.message_id,
                    request_kind="chat",
                    status="ERROR",
                    question=trace_question,
                    answer="",
                    started=started,
                    events=TRACE_EVENTS.get() or [],
                    error="PROCESSING_FAILED",
                )
            raise _error("PROCESSING_FAILED") from error
        finally:
            if not finished:
                await runtime.cancel(session, generation, request.message_id)
            TRACE_EVENTS.reset(trace_token)

    @app.post("/api/sessions/{session_id}/forget")
    async def forget_session(session_id: str, request: ForgetRequest) -> dict[str, Any]:
        payload = request.model_dump()
        try:
            session, replay, admitted, _admitted_generation = await runtime.command(
                session_id, request.message_id, payload
            )
        except RuntimeErrorCode as error:
            raise _error(
                error.code,
                status=404
                if error.code == "SESSION_NOT_FOUND"
                else 410
                if error.code in {"SESSION_EXPIRED", "SESSION_RESET"}
                else 429
                if error.code.endswith("CAPACITY_EXCEEDED")
                else 409,
                retry="none" if error.code == "SESSION_NOT_FOUND" else "new_message",
            ) from error
        if replay is not None:
            status = session.commands[request.message_id].status
            body = json.loads(replay)
            return (
                JSONResponse(content=body, status_code=status)
                if status != 200
                else body
            )

        task: asyncio.Task | None = None
        error_body: dict[str, Any] | None = None
        async with session.lock:
            if session.closed or runtime.sessions.get(session_id) is not session:
                raise _error("SESSION_RESET", status=410, retry="none")
            # A duplicate may have passed command() while its first request was
            # awaiting cancellation. Resolve it before changing memory again.
            record = session.commands.get(request.message_id)
            if record is not None and not admitted:
                if record.fingerprint != fingerprint(payload):
                    raise _error("IDENTITY_CONFLICT")
                if record.response is None:
                    raise _error("IN_PROGRESS")
                return json.loads(record.response)

            chat_ids = set(session.chats)
            client_ids = set(session.memory.client_turns)
            if request.scope.kind == "messages":
                selected_ids = set(request.scope.message_ids or ())
                if not selected_ids <= (chat_ids | client_ids):
                    error_body = {
                        "detail": {"code": "INVALID_REQUEST", "retry": "none"}
                    }
                refs = {
                    session.memory.client_turns[item]
                    for item in selected_ids
                    if item in session.memory.client_turns
                }
                if error_body is None and refs:
                    try:
                        updated, helper_ids = forget(session.memory, refs)
                    except MemoryError:
                        error_body = {
                            "detail": {"code": "INVALID_REQUEST", "retry": "none"}
                        }
                if error_body is None and not refs:
                    updated, helper_ids = session.memory, ()
                if error_body is None:
                    redacted_ids = sorted(set(helper_ids) | selected_ids)
                    tombstoned_chat_ids = set(redacted_ids)
            else:
                updated, helper_ids = forget(session.memory, clear_all=True)
                redacted_ids = sorted(set(helper_ids) | chat_ids | client_ids)
                tombstoned_chat_ids = chat_ids | client_ids

            if error_body is not None:
                updated = session.memory
                helper_ids = ()
                redacted_ids = []
                tombstoned_chat_ids = set()
                forgotten_fields = []
            else:
                forgotten_fields = sorted(
                    set(session.memory.facts) - set(updated.facts)
                )
            # Reserve the command identity before mutating. A concurrent
            # duplicate will observe IN_PROGRESS rather than apply twice.
            if error_body is None:
                session.memory = updated
                session.generation += 1
                task, session.inflight, session.task = session.task, False, None
            # A remaining chat identity cannot replay a response that may have
            # derived from forgotten material, but only affected identities are
            # made permanently unusable.
            if error_body is None:
                session.chats = {
                    key: LedgerRecord(
                        None,
                        None,
                        True,
                        sequence=record.sequence,
                        status=record.status,
                    )
                    if key in tombstoned_chat_ids
                    else LedgerRecord(
                        record.fingerprint,
                        None,
                        record.forgotten,
                        sequence=record.sequence,
                        status=record.status,
                    )
                    for key, record in session.chats.items()
                }
                session.commands = {
                    key: LedgerRecord(
                        None,
                        None,
                        True,
                        sequence=record.sequence,
                        status=record.status,
                    )
                    if key in tombstoned_chat_ids
                    else LedgerRecord(
                        record.fingerprint,
                        None,
                        record.forgotten,
                        sequence=record.sequence,
                        status=record.status,
                    )
                    for key, record in session.commands.items()
                }

        if error_body is not None:
            encoded = json.dumps(error_body, separators=(",", ":"))
            await runtime.finish_command(
                session, request.message_id, payload, encoded, status=422
            )
            return JSONResponse(content=error_body, status_code=422)

        # Forget has changed the memory generation. Remove every associated
        # diagnostic before any cancellation point can let an in-flight task run.
        purge_session_traces(session_id)

        response = _envelope(
            {
                "status": "ANSWERED",
                "answer": "The requested chat memory has been removed.",
                "memory_receipt": {
                    "revision": updated.revision,
                    "changes": [],
                    "invalidated": ["calculation", "offer", "review"],
                    "redacted_message_ids": redacted_ids,
                },
                "forget_receipt": {
                    "forgotten_fields": forgotten_fields,
                    "form_values_cleared": False,
                },
            },
            session_id,
            request.message_id,
        )

        async def cleanup_and_finish() -> None:
            if task is not None and task is not asyncio.current_task():
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
            await runtime.finish_command(
                session,
                request.message_id,
                payload,
                json.dumps(response, separators=(",", ":")),
            )

        completion = asyncio.create_task(cleanup_and_finish())
        try:
            await asyncio.shield(completion)
        except asyncio.CancelledError:
            # A committed forget is not allowed to strand its command record or
            # acknowledge before the cancelled chat has finished unwinding.
            await asyncio.shield(completion)
            raise
        return response

    @app.post("/api/sessions/{session_id}/reset")
    async def reset_session(session_id: str, request: ResetRequest) -> dict[str, Any]:
        response = _envelope(
            {
                "status": "ANSWERED",
                "answer": "This conversation has been reset.",
                "reset_receipt": {
                    "reset_id": f"reset_{request.message_id}",
                    "closed": True,
                    "form_values_cleared": False,
                },
                "memory_receipt": {
                    "revision": 0,
                    "changes": [],
                    "invalidated": ["calculation", "offer", "review"],
                    "redacted_message_ids": [],
                },
            },
            session_id,
            request.message_id,
        )
        import json

        try:
            replay = await runtime.reset(
                session_id,
                request.message_id,
                request.model_dump(),
                json.dumps(response, separators=(",", ":")),
            )
        except RuntimeErrorCode as error:
            raise _error(
                error.code,
                status=429
                if error.code.endswith("CAPACITY_EXCEEDED")
                else 409
                if error.code == "IDENTITY_CONFLICT"
                else 410,
                retry="none",
            ) from error
        purge_session_traces(session_id)
        return json.loads(replay) if replay is not None else response

    return app
