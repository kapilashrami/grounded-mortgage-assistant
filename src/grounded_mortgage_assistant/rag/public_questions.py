"""Public-question routing and partner scoping for corpus retrieval results."""

from __future__ import annotations

import re
from collections.abc import Mapping
from typing import Any

_WORD = re.compile(r"[a-z]+")
_STOP = frozenset(
    [
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "by",
        "can",
        "could",
        "did",
        "do",
        "does",
        "for",
        "from",
        "how",
        "i",
        "if",
        "in",
        "is",
        "it",
        "may",
        "me",
        "might",
        "my",
        "of",
        "on",
        "or",
        "our",
        "should",
        "tell",
        "that",
        "the",
        "their",
        "there",
        "these",
        "this",
        "to",
        "was",
        "we",
        "what",
        "when",
        "where",
        "who",
        "why",
        "will",
        "with",
        "would",
        "you",
        "your",
    ]
)
_EXECUTION = re.compile(
    r"\b(?:draft|calculate|review|forget|delete|remove|correct|change|update)\b",
    re.IGNORECASE,
)
_ASSERTION = re.compile(
    r"\b(?:my mortgage is|i owe|change my property|my property value is|i have a mortgage)\b",
    re.IGNORECASE,
)
_QUESTION = re.compile(
    r"\?|\b(?:what|when|where|who|why|how|can|could|would|is|are|do|does|explain|tell me about)\b",
    re.IGNORECASE,
)


def _terms(value: str) -> tuple[str, ...]:
    terms = []
    for word in _WORD.findall(value.casefold()):
        if word in _STOP:
            continue
        if (
            len(word) > 3
            and word.endswith("s")
            and not word.endswith(("ss", "status", "business"))
        ):
            word = word[:-1]
        terms.append(word)
    return tuple(terms)


class PublicQuestionScope:
    def __init__(self, records: dict[str, dict[str, Any]]) -> None:
        self.partner_urls: dict[str, tuple[str, ...]] = {}
        self.partner_titles: dict[str, tuple[str, ...]] = {}
        for record_id, metadata in records.items():
            contexts = [
                item
                for item in metadata.get("source_context", [])
                if isinstance(item, Mapping)
            ]
            self.partner_urls[record_id] = tuple(
                sorted(
                    {
                        item.get("resolved_url") or item.get("url")
                        for item in contexts
                        if isinstance(item.get("resolved_url") or item.get("url"), str)
                    }
                )
            )
            self.partner_titles[record_id] = tuple(
                item.get("title", "") for item in contexts
            )

    @staticmethod
    def is_public_question(raw: str) -> bool:
        return bool(_QUESTION.search(raw)) and not bool(
            re.search(r"\d", raw)
            or _EXECUTION.search(raw)
            or _ASSERTION.search(raw)
            or re.search(r"\b(?:my|our)\s+ltv\b", raw, re.IGNORECASE)
        )

    def scope_partner_records(
        self, raw: str, records: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """Drop partner-specific records unless the question names that partner."""
        return [
            row
            for row in records
            if not self._partner_scoped(
                self.partner_urls.get(row["id"], ()),
                self.partner_titles.get(row["id"], ()),
                raw,
            )
        ]

    @staticmethod
    def _partner_scoped(
        urls: tuple[str, ...], titles: tuple[str, ...], raw: str
    ) -> bool:
        slugs = [
            url.rsplit("/partners/", 1)[1]
            .split("/", 1)[0]
            .replace("-", " ")
            .replace("_", " ")
            for url in urls
            if "/partners/" in url
        ]
        if not slugs:
            return False
        words = " ".join(_terms(raw))
        return not any(
            label
            and (
                re.search(rf"\b{re.escape(' '.join(_terms(label)))}\b", words)
                or re.search(rf"\b{re.escape(' '.join(_terms(slug)))}\b", words)
            )
            for label in titles
            for slug in slugs
        )
