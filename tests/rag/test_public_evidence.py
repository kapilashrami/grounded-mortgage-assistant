"""Public evidence rendering and citation invariants over the whole frozen bundle."""

import json
import re

import pytest

from grounded_mortgage_assistant.rag.public_evidence import PublicEvidence
from grounded_mortgage_assistant.rag.retrieval import ARTIFACT

pytestmark = pytest.mark.requires_corpus

PDF_URL = re.compile(
    r"^/api/sources/(?P<id>[^/]+)\.pdf\?sha256=(?P<sha>[0-9a-f]{64})#page=(?P<page>\d+)$"
)


def _servable(evidence, url):
    match = PDF_URL.match(url)
    assert match, url
    path = evidence.pdf(match["id"], match["sha"])
    assert path is not None and path.is_file(), url
    return int(match["page"])


def test_every_frozen_record_renders_its_reviewed_extract_in_retrieval_order(
    monkeypatch,
):
    evidence = PublicEvidence()
    # Text rendering covers local statements without public URLs as well; citation
    # validity is checked independently below.
    monkeypatch.setattr(
        evidence, "citation", lambda record_id: {"evidence_id": record_id}
    )
    projections = json.loads((ARTIFACT / "union/projections.json").read_text())
    rows = [
        {
            "id": record_id,
            "metadata": {"generation_text": projections[record_id]["generation_text"]},
        }
        for record_id in evidence.records
    ]
    for start in range(0, len(rows), 5):
        batch = rows[start : start + 5]
        # The model's selection order must not override the retriever's order.
        selected = [row["id"] for row in reversed(batch)]
        rendered = evidence.render(selected, batch)
        expected_ids = [row["id"] for row in batch]
        assert rendered["answer"] == "\n\n".join(
            evidence.records[i]["final_answer"] for i in expected_ids
        ), start
        assert [c["evidence_id"] for c in rendered["citations"]] == expected_ids, start
        assert rendered["related_articles"] == [] and rendered["actions"] == []


def test_every_frozen_citation_is_servable_or_rejected():
    evidence = PublicEvidence()
    cited = english = 0
    for record_id in evidence.records:
        try:
            citation = evidence.citation(record_id)
        except ValueError:
            continue
        cited += 1
        assert citation["evidence_id"] == record_id
        assert citation["url"].startswith("https://")
        page = _servable(evidence, citation["pdf_url"])
        assert citation["pages"][0] == page
        if citation["pdf_language"] == "en":
            english += 1
            assert len(citation["pages"]) == 1
            original_page = _servable(evidence, citation["original_pdf_url"])
            assert citation["original_pages"][0] == original_page
        else:
            assert citation["pdf_language"] == "sv"
            assert "original_pdf_url" not in citation
    assert cited and english == cited, (cited, english)


def test_pdf_requires_the_pinned_digest():
    evidence = PublicEvidence()
    source_id, source = next(iter(evidence.english_sources.items()))
    assert evidence.pdf(source_id, source["sha256"]) is not None
    assert evidence.pdf(source_id, "0" * 64) is None
    assert evidence.pdf("unknown-source", source["sha256"]) is None


def test_unretrieved_selection_cannot_render_public_evidence():
    evidence = PublicEvidence()
    record_id = next(iter(evidence.records))
    with pytest.raises(ValueError, match="selection"):
        evidence.render([record_id], [])


def test_records_without_a_source_url_are_not_citable():
    evidence = PublicEvidence()
    uncitable = [
        record_id
        for record_id in evidence.records
        if not evidence.is_citable(record_id)
    ]
    # 58 frozen records lack an https source URL; the rest render a citation.
    assert len(uncitable) == 58
    with pytest.raises(ValueError, match="source URL"):
        evidence.citation(uncitable[0])
    assert all(
        evidence.is_citable(record_id)
        for record_id in evidence.records
        if record_id not in uncitable
    )
