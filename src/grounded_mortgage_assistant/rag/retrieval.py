"""Frozen two-view BGE retrieval. No index construction occurs at runtime."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import faiss
import numpy as np
import torch
from sentence_transformers import CrossEncoder, SentenceTransformer

from grounded_mortgage_assistant.config import EMBEDDING_MODEL, RERANKER_MODEL, data_dir

# The frozen corpus lives outside the repository (DATA_DIR); manifest paths are relative to it.
DATA = data_dir()
ARTIFACT = DATA / "artifacts/m3/2026-10-29-english-balanced-demo/retrieval-final-v2"


def _device() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    return "cuda" if torch.cuda.is_available() else "cpu"


@dataclass(frozen=True)
class RetrievalResult:
    records: list[dict[str, Any]]
    trace: dict[str, Any]


def _matches(path: Path, digest: str) -> bool:
    return hashlib.sha256(path.read_bytes()).hexdigest() == digest


def validate_frozen_bundle(artifact: Path = ARTIFACT) -> dict[str, Any]:
    """Validate the pinned union and all manifest-declared local inputs."""
    expected_union = "e55f0d932508063d11d8e98f2087a8d6ead91a73ef1a69e0f36e5f9ea8d04554"
    union_path = artifact / "union/manifest.json"
    if not _matches(union_path, expected_union):
        raise RuntimeError("frozen union manifest changed")
    union = json.loads(union_path.read_text())
    for relative, digest in union["files"].items():
        if not _matches(artifact / "union" / relative, digest):
            raise RuntimeError(f"frozen union/{relative} changed")
    root = artifact.parents[3]
    for relative, digest in union["inputs"].items():
        if not _matches(root / relative, digest):
            raise RuntimeError(f"frozen union input {relative} changed")
    return union


class FrozenRetriever:
    def __init__(self) -> None:
        torch.set_num_threads(1)
        faiss.omp_set_num_threads(1)
        self.base_chunks = self._load("base/chunks.json")
        self.clean_chunks = self._load("clean/chunks.json")
        self.projections = self._load("clean/projections.json")
        self.equivalence = self._load("equivalence-keys.json")
        self._validate()
        self.base = faiss.read_index(str(ARTIFACT / "base/index.faiss"))
        self.clean = faiss.read_index(str(ARTIFACT / "clean/index.faiss"))
        self.base.hnsw.efSearch = 32
        self.clean.hnsw.efSearch = 32
        self.embedder = SentenceTransformer(
            EMBEDDING_MODEL[0],
            revision=EMBEDDING_MODEL[1],
            device="cpu",
            local_files_only=True,
        )
        self.embedder.max_seq_length = 8192
        self.reranker = CrossEncoder(
            RERANKER_MODEL[0],
            revision=RERANKER_MODEL[1],
            device=_device(),
            max_length=8192,
            local_files_only=True,
        )
        self.reranker.model.float().eval()

    def _load(self, relative: str) -> Any:
        return json.loads((ARTIFACT / relative).read_text())

    def _validate(self) -> None:
        # The reviewed union manifest anchors both views and their row order.
        union = validate_frozen_bundle()
        for name, chunks in (("base", self.base_chunks), ("clean", self.clean_chunks)):
            expected = union["strategy"][
                "legacy_manifest_sha256" if name == "base" else "clean_manifest_sha256"
            ]
            if not _matches(ARTIFACT / name / "manifest.json", expected):
                raise RuntimeError(f"frozen {name} manifest changed")
            manifest = self._load(f"{name}/manifest.json")
            for relative, digest in manifest["files"].items():
                if not _matches(ARTIFACT / name / relative, digest):
                    raise RuntimeError(f"frozen {name}/{relative} changed")
            if [row["id"] for row in chunks] != manifest["selection"]["record_ids"]:
                raise RuntimeError(f"frozen {name} row order changed")
            if (
                len(chunks) != 1476
                or manifest.get("english_runtime", {}).get("corpus_sha256")
                != "49c5c544e7e502f39895fa41297d9d584405b9da7fe537b574efb6949b16e473"
            ):
                raise RuntimeError(f"invalid frozen {name} artifact")
            if not all(item["id"] in self.projections for item in chunks):
                raise RuntimeError(f"{name} chunk/projection ID mismatch")
        if (
            hashlib.sha256(
                (ARTIFACT / "clean/projections.json").read_bytes()
            ).hexdigest()
            != "7faf5a8434476c16ca530b52ebf07ba6da6953fbe17cab98a13ea3124f8ded19"
        ):
            raise RuntimeError("frozen projections hash mismatch")

    @staticmethod
    def _view(
        index: faiss.Index, chunks: list[dict[str, Any]], vector: np.ndarray
    ) -> list[dict[str, Any]]:
        scores, positions = index.search(vector, 10)
        return [
            {"id": chunks[int(position)]["id"], "score": float(score)}
            for score, position in zip(scores[0], positions[0])
            if position >= 0
        ]

    def retrieve(self, query: str) -> RetrievalResult:
        vector = self.embedder.encode(
            [query], normalize_embeddings=True, convert_to_numpy=True
        ).astype("float32")
        clean = self._view(self.clean, self.clean_chunks, vector)
        base = self._view(self.base, self.base_chunks, vector)
        union: list[str] = []
        for item in clean + base:
            if item["id"] not in union:
                union.append(item["id"])
        ordered_ids = sorted(union)
        texts = [self.projections[item]["clean_search_text"] for item in ordered_ids]
        # Enforce the explicit model token limit before the one permitted batch.
        lengths = [
            self.reranker.tokenizer(
                query, text, truncation=False, add_special_tokens=True
            )["input_ids"]
            for text in texts
        ]
        if any(len(item) > 8192 for item in lengths):
            raise RuntimeError("frozen clean_search_text exceeds reranker max length")
        values = self.reranker.predict([(query, text) for text in texts], batch_size=32)
        reranked = sorted(
            (
                {"id": record_id, "score": float(score)}
                for record_id, score in zip(ordered_ids, values)
            ),
            key=lambda item: (-item["score"], item["id"]),
        )
        packed: list[dict[str, Any]] = []
        chars = 0
        seen: set[str] = set()
        skipped_equivalents: list[str] = []
        skipped_budget: list[str] = []
        for item in reranked:
            record_id = item["id"]
            key = self.equivalence[record_id]
            generation = self.projections[record_id]["generation_text"]
            if key in seen:
                skipped_equivalents.append(record_id)
                continue
            if chars + len(generation) > 12000:
                skipped_budget.append(record_id)
                continue
            seen.add(key)
            chars += len(generation)
            packed.append({"id": record_id, "text": generation, "score": item["score"]})
            if len(packed) == 5:
                break
        trace = {
            "query": query,
            "models": {
                "embedder": "BAAI/bge-m3@5617a9f61b028005a4858fdac845db406aefb181",
                "reranker": "BAAI/bge-reranker-v2-m3@953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e",
                "embedder_device": "cpu",
                "reranker_device": "mps",
                "max_length": 8192,
                "batch_size": 32,
            },
            "views": {"clean": clean, "base": base},
            "union_ids_clean_first": union,
            "rerank_input_ids_lexicographic": ordered_ids,
            "reranked": reranked,
            "final_ids": [item["id"] for item in packed],
            "generation_texts": [item["text"] for item in packed],
            "characters": chars,
            "skipped_equivalents": skipped_equivalents,
            "skipped_budget": skipped_budget,
        }
        return RetrievalResult(
            records=[
                {
                    "id": item["id"],
                    "score": item["score"],
                    "metadata": {
                        "record_id": item["id"],
                        "generation_text": item["text"],
                    },
                }
                for item in packed
            ],
            trace=trace,
        )
