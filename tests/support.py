"""Test doubles that let the suite run without the private corpus."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from grounded_mortgage_assistant.config import ROOT
from grounded_mortgage_assistant.rag.public_evidence import SampleEvidence
from grounded_mortgage_assistant.rag.sample_bundle import records_from_jsonl, views

SAMPLE = ROOT / "data" / "sample" / "records.jsonl"


class FixtureEvidence(SampleEvidence):
    """The synthetic sample records behind the real rendering logic; no PDFs."""

    def __init__(self, path: Path = SAMPLE) -> None:
        super().__init__(records_from_jsonl(path))

    @staticmethod
    def generation_text(metadata: dict[str, Any]) -> str:
        return views(metadata)["generation"]
