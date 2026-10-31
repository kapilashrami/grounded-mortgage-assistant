"""Public-source metadata and PDF serving helpers, independent of chat runtime."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from grounded_mortgage_assistant.rag.retrieval import ARTIFACT, validate_frozen_bundle


class PublicEvidence:
    def __init__(self, artifact: Path = ARTIFACT) -> None:
        manifest = validate_frozen_bundle(artifact)
        self.root = artifact.parents[3]
        self.records = {
            row["id"]: row["metadata"]
            for row in json.loads((artifact / "union/chunks.json").read_text())
        }
        self.sources = {row["id"]: row for row in manifest["sources"]}
        self.by_path = {row["path"]: row for row in manifest["sources"]}
        derived_path = self.root / "output/pdf/english-citations/manifest.json"
        derived = (
            json.loads(derived_path.read_text()) if derived_path.is_file() else None
        )
        if derived is not None and (
            derived.get("union_sha256")
            != hashlib.sha256(
                (artifact / "union/manifest.json").read_bytes()
            ).hexdigest()
            or derived.get("chunks_sha256")
            != hashlib.sha256((artifact / "union/chunks.json").read_bytes()).hexdigest()
        ):
            raise ValueError(
                "English citation manifest is not anchored to the frozen bundle."
            )
        if derived is not None:
            expected_records = set(self.records)
            if set(derived.get("records", {})) != expected_records:
                raise ValueError(
                    "English citation manifest does not cover the frozen records."
                )
            for source in derived.get("sources", []):
                path = (self.root / source["path"]).resolve()
                if not path.is_relative_to(
                    (self.root / "output/pdf/english-citations").resolve()
                ):
                    raise ValueError(
                        "English citation path is outside the owned archive."
                    )
                original = self.sources.get(source.get("original_source_id"))
                if (
                    original is None
                    or source.get("original_sha256") != original["sha256"]
                ):
                    raise ValueError(
                        "English citation source is not bound to its frozen original."
                    )
            derived_sources = {
                source["id"]: source for source in derived.get("sources", [])
            }
            for record_id, mapping in derived["records"].items():
                metadata = self.records[record_id]
                original = self.by_path.get(
                    str(Path("knowledge") / metadata["pdf_path"])
                )
                source = derived_sources.get(mapping.get("source_id"))
                if (
                    original is None
                    or source is None
                    or source["original_source_id"] != original["id"]
                    or type(mapping.get("page")) is not int
                    or mapping["page"] < 1
                    or mapping.get("original_pages") != metadata.get("pdf_pages")
                ):
                    raise ValueError("English citation record binding is invalid.")
        self.english = derived
        self.english_sources = {
            row["id"]: row for row in (derived or {}).get("sources", [])
        }
        self._citable: dict[str, bool] = {}

    def is_citable(self, record_id: str) -> bool:
        """Whether citation() can render this record. Cached per record."""
        if record_id not in self._citable:
            try:
                self.citation(record_id)
                self._citable[record_id] = True
            except KeyError, ValueError:
                self._citable[record_id] = False
        return self._citable[record_id]

    def citation(self, record_id: str) -> dict[str, Any]:
        metadata = self.records[record_id]
        source = self.by_path.get(str(Path("knowledge") / metadata["pdf_path"]))
        if source is None:
            raise ValueError("Citation has no owned PDF source.")
        path = self.root / source["path"]
        if (
            not path.is_file()
            or hashlib.sha256(path.read_bytes()).hexdigest() != source["sha256"]
        ):
            raise ValueError("Citation PDF is unavailable or changed.")
        contexts, pages = metadata.get("source_context", []), metadata.get("pdf_pages")
        if not contexts or not pages:
            raise ValueError("Citation metadata is incomplete.")
        context = contexts[0]
        url = context.get("resolved_url") or context.get("url")
        if not isinstance(url, str) or not url.startswith("https://"):
            raise ValueError("Citation source URL is unavailable.")
        versions = [
            {
                key: item[key]
                for key in ("title", "source_id", "published_at", "captured_at")
                if item.get(key)
            }
            for item in contexts
        ]
        english = (self.english or {}).get("records", {}).get(record_id)
        english_source = (
            self.english_sources.get(english["source_id"]) if english else None
        )
        if english_source is not None:
            english_path = self.root / english_source["path"]
            if (
                not english_path.is_file()
                or hashlib.sha256(english_path.read_bytes()).hexdigest()
                != english_source["sha256"]
            ):
                raise ValueError("English citation PDF is unavailable or changed.")
            pdf_url = f"/api/sources/{english_source['id']}.pdf?sha256={english_source['sha256']}#page={english['page']}"
            pages = [english["page"]]
            pdf_language = "en"
        else:
            pdf_url = f"/api/sources/{source['id']}.pdf?sha256={source['sha256']}#page={pages[0]}"
            pdf_language = "sv"
        result = {
            "evidence_id": record_id,
            "pages": pages,
            "qualifications": metadata.get("qualifications", []),
            "source_id": context["source_id"],
            "url": url,
            "pdf_url": pdf_url,
            "pdf_language": pdf_language,
            "section": context.get("scope")
            or "Translated English record; original Swedish PDF",
            "source_versions": [item for item in versions if item.get("title")],
        }
        if english_source is not None:
            result["original_pdf_url"] = (
                f"/api/sources/{source['id']}.pdf?sha256={source['sha256']}#page={metadata['pdf_pages'][0]}"
            )
            result["original_pages"] = metadata["pdf_pages"]
        return result

    def pdf(self, source_id: str, digest: str) -> Path | None:
        source = self.sources.get(source_id) or self.english_sources.get(source_id)
        if source is None or digest != source["sha256"]:
            return None
        path = self.root / source["path"]
        return (
            path
            if path.is_file()
            and hashlib.sha256(path.read_bytes()).hexdigest() == digest
            else None
        )

    def render(
        self, selected_ids: list[str], records: list[dict[str, Any]]
    ) -> dict[str, Any]:
        """Render only reviewed extracts whose returned pack contains them intact."""
        blocks: list[str] = []
        by_id = {
            row["id"]: row.get("metadata", {}).get("generation_text", "")
            for row in records
        }
        if (
            len(set(selected_ids)) != len(selected_ids)
            or not set(selected_ids) <= by_id.keys()
        ):
            raise ValueError("Citation selection is unavailable.")
        # Selection chooses evidence; the frozen retriever retains its ordering.
        ordered_ids = [row["id"] for row in records if row["id"] in selected_ids]
        for record_id in ordered_ids:
            metadata = self.records[record_id]
            text, qualifications = (
                metadata["final_answer"],
                metadata.get("qualifications", []),
            )
            generation = by_id.get(record_id, "")
            if (
                not generation
                or text not in generation
                or any(item not in generation for item in qualifications)
            ):
                raise ValueError("Citation extract is unavailable.")
            blocks.append(text)
        return {
            "answer": "\n\n".join(blocks),
            "citations": [self.citation(record_id) for record_id in ordered_ids],
            "related_articles": [],
            "actions": [],
        }
