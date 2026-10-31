"""Test doubles that let the suite run without the private corpus."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from grounded_mortgage_assistant.config import ROOT
from grounded_mortgage_assistant.rag.public_evidence import PublicEvidence

SAMPLE = ROOT / "data" / "sample" / "records.jsonl"


class FixtureEvidence(PublicEvidence):
    """The synthetic sample records behind the real rendering logic; no PDFs."""

    def __init__(self, path: Path = SAMPLE) -> None:
        self.records = {}
        for line in path.read_text().splitlines():
            row = json.loads(line)
            self.records[row["id"]] = {
                "question": row["question"],
                "final_answer": row["answer"],
                "qualifications": row["qualifications"],
                "product": row["product"],
                "source_context": [
                    {"url": row["source"]["url"], "title": row["source"]["title"]}
                ],
            }
        self.sources = {}
        self.english = None
        self.english_sources = {}
        self._citable = {}

    def citation(self, record_id: str) -> dict[str, Any]:
        source = self.records[record_id]["source_context"][0]
        return {
            "evidence_id": record_id,
            "url": source["url"],
            "title": source["title"],
            "pages": [],
            "pdf_url": None,
        }

    def pdf(self, source_id: str, digest: str) -> Path | None:
        return None

    @staticmethod
    def generation_text(metadata: dict[str, Any]) -> str:
        return " ".join([metadata["final_answer"], *metadata.get("qualifications", [])])
