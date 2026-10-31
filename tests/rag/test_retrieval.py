import json
import os
import shutil

import pytest

from grounded_mortgage_assistant.rag.retrieval import (
    ARTIFACT,
    FrozenRetriever,
    validate_frozen_bundle,
)

pytestmark = pytest.mark.requires_corpus


def test_frozen_two_view_retrieval_packs_complete_generation_records():
    result = FrozenRetriever().retrieve(
        "Do I need to provide any security when applying for a personal loan?"
    )
    assert len(result.trace["views"]["clean"]) == 10
    assert len(result.trace["views"]["base"]) == 10
    assert result.trace["characters"] <= 12000
    assert len(result.trace["final_ids"]) <= 5
    assert result.trace["generation_texts"]


def test_union_metadata_tampering_is_rejected_from_a_temporary_copy(tmp_path):
    manifest = json.loads((ARTIFACT / "union/manifest.json").read_text())
    root = tmp_path / "repository"
    copied = root / ARTIFACT.relative_to(ARTIFACT.parents[3])
    (copied / "union").mkdir(parents=True)
    shutil.copy2(ARTIFACT / "union/manifest.json", copied / "union/manifest.json")
    for relative in manifest["files"]:
        target = copied / "union" / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        if relative == "chunks.json":
            shutil.copy2(ARTIFACT / "union" / relative, target)
        else:
            os.symlink(ARTIFACT / "union" / relative, target)
    with (copied / "union/chunks.json").open("a") as handle:
        handle.write("\n")
    for relative in manifest["inputs"]:
        target = root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        os.symlink(ARTIFACT.parents[3] / relative, target)
    with pytest.raises(RuntimeError, match="union/chunks.json"):
        validate_frozen_bundle(copied)
