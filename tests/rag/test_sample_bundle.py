import hashlib
import json
import re

import numpy as np
import pytest

from grounded_mortgage_assistant.config import ROOT
from grounded_mortgage_assistant.rag.public_evidence import SampleEvidence
from grounded_mortgage_assistant.rag.retrieval import FrozenRetriever, bundle_kind
from grounded_mortgage_assistant.rag.sample_bundle import (
    build_sample_bundle,
    validate_sample_bundle,
    views,
)

RECORDS = ROOT / "data/sample/records.jsonl"
WORD = re.compile(r"[a-z]+")


def _words(text):
    return set(WORD.findall(text.lower()))


def encode(texts):
    """Deterministic bag-of-words vectors, so tests need no model download."""
    vectors = np.zeros((len(texts), 64), dtype="float32")
    for row, text in enumerate(texts):
        for word in _words(text):
            vectors[row, int(hashlib.sha256(word.encode()).hexdigest(), 16) % 64] += 1
    return vectors / np.linalg.norm(vectors, axis=1, keepdims=True)


class FakeEmbedder:
    def encode(self, texts, **_):
        return encode(texts)


class FakeReranker:
    @staticmethod
    def tokenizer(query, text, **_):
        return {"input_ids": (query + " " + text).split()}

    @staticmethod
    def predict(pairs, batch_size):
        return [
            len(_words(query) & _words(text)) / max(len(_words(query)), 1)
            for query, text in pairs
        ]


@pytest.fixture
def bundle(tmp_path):
    out = tmp_path / "sample-bundle"
    build_sample_bundle(RECORDS, out, encode)
    return out


def test_built_bundle_validates_and_hashes_its_own_files(bundle):
    manifest = validate_sample_bundle(bundle)
    assert manifest["kind"] == "sample"
    assert len(manifest["record_ids"]) == 20
    assert set(manifest["files"]) == {
        "records.json",
        "projections.json",
        "base/index.faiss",
        "clean/index.faiss",
    }


def test_builder_never_replaces_an_existing_bundle(bundle):
    with pytest.raises(FileExistsError):
        build_sample_bundle(RECORDS, bundle, encode)


def test_a_changed_file_stops_validation(bundle):
    path = bundle / "projections.json"
    projections = json.loads(path.read_text())
    first = next(iter(projections))
    projections[first]["generation_text"] = "Changed."
    path.write_text(json.dumps(projections))
    with pytest.raises(RuntimeError, match="projections.json changed"):
        validate_sample_bundle(bundle)


def test_a_bundle_from_other_models_is_rejected(bundle):
    path = bundle / "manifest.json"
    manifest = json.loads(path.read_text())
    manifest["embedding_model"] = "other/model@main"
    path.write_text(json.dumps(manifest))
    with pytest.raises(RuntimeError, match="different models"):
        validate_sample_bundle(bundle)


def test_the_frozen_corpus_wins_over_a_sample_bundle(tmp_path, bundle):
    artifact = tmp_path / "artifact"
    assert bundle_kind(artifact, tmp_path / "missing") is None
    assert bundle_kind(artifact, bundle) == "sample"
    (artifact / "union").mkdir(parents=True)
    (artifact / "union/manifest.json").write_text("{}")
    assert bundle_kind(artifact, bundle) == "frozen"


def test_sample_retrieval_and_rendering_use_the_sample_records(bundle):
    retriever = FrozenRetriever(
        bundle, embedder=FakeEmbedder(), reranker=FakeReranker()
    )
    result = retriever.retrieve("What does amortization mean?")
    top = result.records[0]
    assert top["id"] == "sample-mortgage-06"
    assert len(result.records) <= 5

    evidence = SampleEvidence.from_bundle(bundle)
    assert (
        top["metadata"]["generation_text"]
        == views(evidence.records[top["id"]])["generation"]
    )
    rendered = evidence.render([top["id"]], result.records)
    assert rendered["answer"] == evidence.records[top["id"]]["final_answer"]
    (citation,) = rendered["citations"]
    assert citation["url"].startswith("https://example.com/")
    assert citation["pdf_url"] is None
