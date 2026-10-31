"""Strict client for the local-only PII gate."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import httpx


class PiiCheckUnavailable(RuntimeError):
    """The loopback detector did not provide a valid result."""


@dataclass(frozen=True)
class PiiDecision:
    """Validated fixed-category detector decision."""

    pii_detected: bool
    category: Literal["email_address", "personal_information"] | None


async def check_pii(text: str) -> PiiDecision:
    """Return a validated fixed-category decision, never a detector response body."""
    try:
        timeout = httpx.Timeout(connect=0.3, read=2.0, write=0.3, pool=0.3)
        async with httpx.AsyncClient(
            base_url="http://127.0.0.1:8097",
            follow_redirects=False,
            timeout=timeout,
            trust_env=False,
        ) as client:
            response = await client.post("/check", json={"text": text})
        if response.status_code != 200 or response.is_redirect:
            raise PiiCheckUnavailable
        body = response.json()
        if (
            set(body) != {"pii_detected", "category"}
            or type(body["pii_detected"]) is not bool
            or body["category"] not in {"email_address", "personal_information", None}
            or (body["pii_detected"] != (body["category"] is not None))
        ):
            raise PiiCheckUnavailable
        return PiiDecision(body["pii_detected"], body["category"])
    except (httpx.HTTPError, TypeError, ValueError, KeyError) as error:
        raise PiiCheckUnavailable from error
