"""Loopback-only local PII detector for the candidate backend."""

from __future__ import annotations

import asyncio
import importlib
import json
import logging
import os
import re
from importlib.metadata import version
from pathlib import Path
from threading import Lock
from typing import Any

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse


def _disable_guardrails_logging(scope_handler: type[Any]) -> None:
    for name in ("guardrails-ai", "guardrails", "presidio-analyzer"):
        logger = logging.getLogger(name)
        for handler in tuple(logger.handlers):
            if isinstance(handler, scope_handler):
                logger.removeHandler(handler)
                handler.close()
        logger.handlers.clear()
        logger.disabled = True
        logger.propagate = False


def _build_detector() -> tuple[Any, type[Any], type[Any]]:
    if (Path.home() / ".guardrailsrc").exists():
        raise RuntimeError("local Guardrails configuration is not allowed")
    nltk_data_value = os.environ.get("NLTK_DATA")
    nltk_data = Path(nltk_data_value) if nltk_data_value else None
    if not (
        nltk_data is not None
        and nltk_data.is_absolute()
        and nltk_data.is_dir()
        and (nltk_data / "tokenizers" / "punkt").is_dir()
        and (nltk_data / "tokenizers" / "punkt_tab").is_dir()
    ):
        raise RuntimeError("local tokenizer data is unavailable")
    if version("en-core-web-lg") != "3.8.0":
        raise RuntimeError("local language model is unavailable")
    importlib.import_module("en_core_web_lg")
    from guardrails.logger import ScopeHandler
    from guardrails.settings import settings
    from guardrails.validator_base import FailResult, PassResult
    from guardrails_ai.detect_pii import DetectPII

    settings.rc.enable_metrics = False
    settings.disable_tracing = True
    settings.use_server = False
    _disable_guardrails_logging(ScopeHandler)
    detector = DetectPII(
        pii_entities=[
            "PERSON",
            "EMAIL_ADDRESS",
            "PHONE_NUMBER",
            "IP_ADDRESS",
            *DetectPII.PII_ENTITIES_MAP["spi"],
        ],
        use_local=True,
        load_nltk_data=False,
    )
    return detector, FailResult, PassResult


try:
    detector, fail_result_type, pass_result_type = _build_detector()
except Exception:  # noqa: BLE001 - fail closed without retaining diagnostics.
    detector = None
    fail_result_type = None
    pass_result_type = None

detector_lock = Lock()
app = FastAPI(docs_url=None, redoc_url=None, openapi_url=None)
_EMAIL = re.compile(r"[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}")


def _result(status: int, body: dict[str, Any]) -> JSONResponse:
    return JSONResponse(
        status_code=status, content=body, headers={"Cache-Control": "no-store"}
    )


def _check(text: str) -> tuple[bool, str | None]:
    if detector is None or fail_result_type is None or pass_result_type is None:
        raise RuntimeError("detector unavailable")
    if _EMAIL.search(text):
        return True, "email_address"
    with detector_lock:
        result = detector.validate(text, metadata={})
    if isinstance(result, fail_result_type):
        return True, "personal_information"
    if isinstance(result, pass_result_type):
        return False, None
    raise RuntimeError("invalid detector result")


@app.on_event("startup")
async def startup() -> None:
    print(
        "PII gate ready" if detector is not None else "PII gate unavailable", flush=True
    )


@app.get("/health")
async def health(request: Request) -> JSONResponse:
    if request.client is None or request.client.host != "127.0.0.1":
        return _result(403, {"detail": {"code": "LOOPBACK_ONLY"}})
    return _result(
        200 if detector is not None else 503,
        {"status": "ready" if detector is not None else "unavailable"},
    )


@app.post("/check")
async def check(request: Request) -> JSONResponse:
    if request.client is None or request.client.host != "127.0.0.1":
        return _result(403, {"detail": {"code": "LOOPBACK_ONLY"}})
    raw = bytearray()
    try:
        async for chunk in request.stream():
            raw.extend(chunk)
            if len(raw) > 8192:
                return _result(422, {"detail": {"code": "INVALID_REQUEST"}})
        body = json.loads(raw)
    except Exception:  # noqa: BLE001 - never echo malformed request content.
        return _result(422, {"detail": {"code": "INVALID_REQUEST"}})
    if (
        not isinstance(body, dict)
        or set(body) != {"text"}
        or not isinstance(body.get("text"), str)
        or not 1 <= len(body["text"]) <= 1000
    ):
        return _result(422, {"detail": {"code": "INVALID_REQUEST"}})
    try:
        pii_detected, category = await asyncio.to_thread(_check, body["text"])
    except Exception:  # noqa: BLE001 - detector failures have one fixed response.
        return _result(503, {"detail": {"code": "PII_CHECK_UNAVAILABLE"}})
    return _result(200, {"pii_detected": pii_detected, "category": category})
