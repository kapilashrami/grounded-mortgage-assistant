"""Bounded frozen retrieval comparison. It records, never masks, divergence."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from grounded_mortgage_assistant.rag.retrieval import FrozenRetriever

ROOT = Path(__file__).resolve().parents[1]
# The audited cases and receipts are company-owned and private; set EVIDENCE_DIR to rerun.
EVIDENCE = Path(os.environ.get("EVIDENCE_DIR", ROOT / "evidence")).expanduser()
CONTROL = EVIDENCE / "evaluation/control/retrieval/english-balanced-10x2-rerank.json"
OUTPUT = EVIDENCE / "evaluation/integration-parity.json"


def expected(row: dict) -> dict:
    trace = row["trace"]
    steps = {item.get("backend"): item for item in trace.get("steps", [])}
    return {
        "final_ids": trace.get(
            "final_hit_ids", [hit["id"] for hit in row.get("hits", [])]
        ),
        "texts": [hit["text"] for hit in row.get("hits", [])],
        "clean": [
            item.get("id")
            for item in steps.get("dense_clean", {}).get("candidates", [])
        ],
        "base": [
            item.get("id")
            for item in steps.get("dense_legacy", {}).get("candidates", [])
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int)
    parser.add_argument(
        "--output", type=Path, default=OUTPUT, help="receipt path; keep earlier ones"
    )
    args = parser.parse_args()
    rows = json.loads(CONTROL.read_text())["rows"][: args.limit]
    retriever = FrozenRetriever()
    comparisons = []
    for row in rows:
        result = retriever.retrieve(row["query"])
        wanted = expected(row)
        got = result.trace
        actual = {
            "final_ids": got["final_ids"],
            "texts": got["generation_texts"],
            "clean": [item["id"] for item in got["views"]["clean"]],
            "base": [item["id"] for item in got["views"]["base"]],
        }
        comparisons.append(
            {
                "case_id": row["case_id"],
                "query": row["query"],
                "matches": {key: actual[key] == wanted[key] for key in wanted},
                "expected": wanted,
                "actual": actual,
                "trace": got,
            }
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(
            {
                "compared": len(comparisons),
                "all_match": all(all(item["matches"].values()) for item in comparisons),
                "rows": comparisons,
            },
            indent=2,
        )
    )
    print(args.output)
    print(
        json.dumps(
            {
                "compared": len(comparisons),
                "all_match": all(all(item["matches"].values()) for item in comparisons),
            }
        )
    )


if __name__ == "__main__":
    main()
