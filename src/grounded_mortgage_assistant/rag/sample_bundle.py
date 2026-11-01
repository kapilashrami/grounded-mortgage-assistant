"""Build and validate a small retrieval bundle from the synthetic sample records.

The sample bundle uses the same two-view layout as the frozen corpus: a plain
question-and-answer view and a labelled view, each in its own FAISS HNSW index,
with the reranker scoring the labelled text. Its manifest hashes its own files,
so a changed file stops startup, but it is never mistaken for the pinned corpus.
"""

from __future__ import annotations

import hashlib
import json
import tempfile
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np

from grounded_mortgage_assistant.config import EMBEDDING_MODEL, RERANKER_MODEL

KIND = "sample"
FILES = ("records.json", "projections.json", "base/index.faiss", "clean/index.faiss")
Encoder = Callable[[list[str]], np.ndarray]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def records_from_jsonl(path: Path) -> dict[str, dict[str, Any]]:
    """Read sample records into the metadata shape PublicEvidence renders."""
    records: dict[str, dict[str, Any]] = {}
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if row["id"] in records:
            raise ValueError(f"duplicate sample record {row['id']}")
        records[row["id"]] = {
            "question": row["question"],
            "final_answer": row["answer"],
            "qualifications": row["qualifications"],
            "product": row["product"],
            "source_context": [
                {
                    "url": row["source"]["url"],
                    "title": row["source"]["title"],
                    "source_id": row["id"],
                }
            ],
        }
    return records


def views(metadata: dict[str, Any]) -> dict[str, str]:
    """The texts one record contributes: two search views and the answer pack."""
    title = metadata["source_context"][0]["title"]
    return {
        "base": f"{metadata['question']} {metadata['final_answer']}",
        "clean": "\n".join(
            [
                f"Question: {metadata['question']}",
                f"Answer: {metadata['final_answer']}",
                f"Product: {metadata['product']}",
                f"Source: {title}",
            ]
        ),
        "generation": " ".join(
            [metadata["final_answer"], *metadata.get("qualifications", [])]
        ),
    }


def _index(vectors: np.ndarray) -> Any:
    import faiss

    index = faiss.IndexHNSWFlat(vectors.shape[1], 32, faiss.METRIC_INNER_PRODUCT)
    index.hnsw.efConstruction = 80
    index.add(vectors)
    return index


def build_sample_bundle(records_path: Path, out: Path, encode: Encoder) -> dict:
    """Write a sample bundle to `out`; refuses to replace an existing folder."""
    if out.exists():
        raise FileExistsError(f"{out} already exists; move it away to rebuild")
    records = records_from_jsonl(records_path)
    if not records:
        raise ValueError("no sample records")
    ids = list(records)
    texts = {record_id: views(records[record_id]) for record_id in ids}
    out.parent.mkdir(parents=True, exist_ok=True)
    # Build in a temporary folder next to `out`; a failed build leaves nothing.
    with tempfile.TemporaryDirectory(dir=out.parent, prefix=".sample-bundle-") as tmp:
        staging = Path(tmp) / "bundle"
        _write(staging, ids, records, texts, records_path, encode)
        manifest = json.loads((staging / "manifest.json").read_text())
        staging.rename(out)
    return manifest


def _write(
    staging: Path,
    ids: list[str],
    records: dict[str, dict[str, Any]],
    texts: dict[str, dict[str, str]],
    records_path: Path,
    encode: Encoder,
) -> None:
    import faiss

    (staging / "base").mkdir(parents=True)
    (staging / "clean").mkdir()
    for view in ("base", "clean"):
        vectors = np.asarray(encode([texts[i][view] for i in ids]), dtype="float32")
        if vectors.ndim != 2 or vectors.shape[0] != len(ids):
            raise ValueError("encoder returned the wrong shape")
        faiss.write_index(_index(vectors), str(staging / view / "index.faiss"))
    (staging / "records.json").write_text(
        json.dumps([{"id": i, "metadata": records[i]} for i in ids], indent=2) + "\n"
    )
    (staging / "projections.json").write_text(
        json.dumps(
            {
                i: {
                    "clean_search_text": texts[i]["clean"],
                    "generation_text": texts[i]["generation"],
                }
                for i in ids
            },
            indent=2,
        )
        + "\n"
    )
    manifest = {
        "kind": KIND,
        "schema_version": 1,
        "source_sha256": _sha256(records_path),
        "embedding_model": "@".join(EMBEDDING_MODEL),
        "reranker_model": "@".join(RERANKER_MODEL),
        "dimensions": int(vectors.shape[1]),
        "record_ids": ids,
        "files": {name: _sha256(staging / name) for name in FILES},
    }
    (staging / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


def validate_sample_bundle(path: Path) -> dict[str, Any]:
    """Check a sample bundle's own hashes, models and record order."""
    manifest = json.loads((path / "manifest.json").read_text())
    if manifest.get("kind") != KIND or manifest.get("schema_version") != 1:
        raise RuntimeError("not a sample bundle")
    if manifest.get("embedding_model") != "@".join(EMBEDDING_MODEL) or manifest.get(
        "reranker_model"
    ) != "@".join(RERANKER_MODEL):
        raise RuntimeError("sample bundle was built with different models")
    if set(manifest.get("files", {})) != set(FILES):
        raise RuntimeError("sample bundle file list is incomplete")
    for name, digest in manifest["files"].items():
        if not (path / name).is_file() or _sha256(path / name) != digest:
            raise RuntimeError(f"sample bundle {name} changed")
    records = json.loads((path / "records.json").read_text())
    if [row["id"] for row in records] != manifest["record_ids"]:
        raise RuntimeError("sample bundle record order changed")
    return manifest
