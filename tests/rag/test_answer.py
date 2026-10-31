import asyncio

from grounded_mortgage_assistant.llm.conversation import EvidenceSelection
from grounded_mortgage_assistant.rag.answer import (
    RELEVANCE_THRESHOLD,
    answer_public_question,
)
from grounded_mortgage_assistant.rag.public_questions import PublicQuestionScope

PARTNER_URL = "https://www.example.com/partners/acme/loan"
GENERAL_URL = "https://www.example.com/mortgage"


class FakeEvidence:
    def __init__(self, uncitable=()):
        self.rendered_with = None
        self.uncitable = set(uncitable)

    def is_citable(self, record_id):
        return record_id not in self.uncitable

    def render(self, selected_ids, records):
        self.rendered_with = (list(selected_ids), [row["id"] for row in records])
        return {"answer": "rendered", "citations": [], "related_articles": []}


def scope_for(*records):
    return PublicQuestionScope(
        {
            record_id: {"source_context": [{"url": url, "title": title}]}
            for record_id, url, title in records
        }
    )


def record(record_id, score):
    return {"id": record_id, "score": score, "metadata": {"generation_text": "x"}}


def run(question, *, records, selection=None, scope=None, evidence=None):
    calls = []

    async def retriever(query):
        calls.append(("retrieve", query))
        if isinstance(records, Exception):
            raise records
        return records

    async def selector(query, rows):
        calls.append(("select", query))
        return selection or EvidenceSelection(
            evidence_ids=[rows[0]["id"]], needs_clarification=False
        )

    result = asyncio.run(
        answer_public_question(
            question,
            retriever=retriever,
            selector=selector,
            evidence=evidence or FakeEvidence(),
            scope=scope or scope_for(("a", GENERAL_URL, "Mortgage")),
        )
    )
    return result, calls


def test_answered_path_renders_the_selected_scored_records():
    evidence = FakeEvidence()
    result, calls = run(
        "What is a mortgage loan?",
        records=[record("a", 0.9)],
        evidence=evidence,
    )
    assert result.status == "answered"
    assert result.top_record_id == "a"
    assert result.top_score == 0.9
    assert result.rendered == {
        "answer": "rendered",
        "citations": [],
        "related_articles": [],
    }
    assert evidence.rendered_with == (["a"], ["a"])
    assert calls == [
        ("retrieve", "What is a mortgage loan?"),
        ("select", "What is a mortgage loan?"),
    ]


def test_top_score_below_threshold_is_unsupported_and_never_selects():
    result, calls = run(
        "What is the weather in Paris today?",
        records=[record("a", RELEVANCE_THRESHOLD - 0.01)],
    )
    assert result.status == "unsupported"
    assert result.top_record_id == "a"
    assert result.rendered is None
    assert calls == [("retrieve", "What is the weather in Paris today?")]


def test_top_score_at_threshold_is_supported():
    result, _ = run(
        "What is a mortgage loan?", records=[record("a", RELEVANCE_THRESHOLD)]
    )
    assert result.status == "answered"


def test_missing_score_fails_closed_as_unsupported():
    unscored = {"id": "a", "metadata": {"generation_text": "x"}}
    result, calls = run("What is a mortgage loan?", records=[unscored])
    assert result.status == "unsupported"
    assert result.top_score is None
    assert calls == [("retrieve", "What is a mortgage loan?")]


def test_partner_record_is_removed_unless_the_question_names_the_partner():
    scope = scope_for(
        ("partner", PARTNER_URL, "Acme loan"), ("general", GENERAL_URL, "Mortgage")
    )
    result, _ = run(
        "What is a mortgage loan?",
        records=[record("partner", 0.95), record("general", 0.8)],
        scope=scope,
    )
    assert result.status == "answered"
    assert result.top_record_id == "general"
    assert [row["id"] for row in result.records] == ["general"]

    named, _ = run(
        "What does Acme charge for a loan?",
        records=[record("partner", 0.95), record("general", 0.8)],
        scope=scope,
    )
    assert named.top_record_id == "partner"


def test_selector_clarification_is_unavailable():
    result, _ = run(
        "What is a mortgage loan?",
        records=[record("a", 0.9)],
        selection=EvidenceSelection(evidence_ids=[], needs_clarification=True),
    )
    assert result.status == "unavailable"
    assert result.rendered is None


def test_retriever_exception_is_unavailable():
    result, calls = run("What is a mortgage loan?", records=RuntimeError("down"))
    assert result.status == "unavailable"
    assert result.top_record_id is None
    assert calls == [("retrieve", "What is a mortgage loan?")]


def test_uncitable_records_are_never_offered_to_the_selector():
    offered = []

    async def retriever(query):
        return [record("a", 0.9), record("broken", 0.8), record("b", 0.7)]

    async def selector(query, rows):
        offered.extend(row["id"] for row in rows)
        return EvidenceSelection(evidence_ids=["a", "b"], needs_clarification=False)

    evidence = FakeEvidence(uncitable={"broken"})
    result = asyncio.run(
        answer_public_question(
            "Does it cost anything to compare loans?",
            retriever=retriever,
            selector=selector,
            evidence=evidence,
            scope=scope_for(("a", GENERAL_URL, "Mortgage")),
        )
    )
    assert offered == ["a", "b"]
    assert result.status == "answered"
    assert evidence.rendered_with == (["a", "b"], ["a", "b"])


def test_only_uncitable_records_is_unsupported_without_selection():
    result, calls = run(
        "What is a mortgage loan?",
        records=[record("a", 0.9)],
        evidence=FakeEvidence(uncitable={"a"}),
    )
    assert result.status == "unsupported"
    assert calls == [("retrieve", "What is a mortgage loan?")]
