"""Calibrate the public-answer relevance gate on the audited English control cases.

Every English case is retrieved with the frozen retriever and its raw top
reranker score is recorded. The threshold rule is fixed here, not tuned by hand:
choose the highest candidate that keeps at least 97% of supported cases.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from statistics import median

from grounded_mortgage_assistant.rag.retrieval import FrozenRetriever

ROOT = Path(__file__).resolve().parents[1]
# The audited cases and receipts are private (Lendo-owned); set EVIDENCE_DIR to rerun.
EVIDENCE = Path(os.environ.get("EVIDENCE_DIR", ROOT / "evidence")).expanduser()
CASES = EVIDENCE / "evaluation/control/cases.audited.json"
OUTPUT = EVIDENCE / "evaluation/2026-10-31-relevance-gate/calibration.json"
KEEP_SUPPORTED_MINIMUM = 0.97
CANDIDATES = [round(step / 100, 2) for step in range(1, 96)]


def distribution(values: list[float]) -> dict[str, float]:
    ordered = sorted(values)

    def percentile(fraction: float) -> float:
        return ordered[min(len(ordered) - 1, int(fraction * len(ordered)))]

    return {
        "min": ordered[0],
        "p5": percentile(0.05),
        "p10": percentile(0.10),
        "median": median(ordered),
        "max": ordered[-1],
    }


def main() -> None:
    cases = json.loads(CASES.read_text())["cases"]
    english = [case for case in cases if case["language"] == "en"]
    retriever = FrozenRetriever()
    rows = []
    for case in english:
        result = retriever.retrieve(case["query"])
        top = result.trace["reranked"][0]
        rows.append(
            {
                "id": case["id"],
                "expected_status": case["expected_status"],
                "top_score": top["score"],
                "top_id": top["id"],
            }
        )
        print(case["id"], case["expected_status"], round(top["score"], 4), flush=True)

    by_status = {
        status: [row["top_score"] for row in rows if row["expected_status"] == status]
        for status in sorted({row["expected_status"] for row in rows})
    }
    supported = by_status["supported"]
    unsupported = by_status.get("unsupported", [])
    sweep = []
    for threshold in CANDIDATES:
        sweep.append(
            {
                "threshold": threshold,
                "supported_kept": sum(score >= threshold for score in supported),
                "unsupported_rejected": sum(score < threshold for score in unsupported),
            }
        )
    eligible = [
        item
        for item in sweep
        if item["supported_kept"] >= KEEP_SUPPORTED_MINIMUM * len(supported)
    ]
    chosen = max(eligible, key=lambda item: item["threshold"])
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(
        json.dumps(
            {
                "rule": "highest threshold keeping at least 97% of supported cases",
                "cases": rows,
                "distribution": {
                    status: distribution(scores) for status, scores in by_status.items()
                },
                "sweep": sweep,
                "chosen": chosen,
                "supported_total": len(supported),
                "unsupported_total": len(unsupported),
            },
            indent=2,
        )
    )
    print(OUTPUT)
    print(
        json.dumps(
            {
                "chosen": chosen,
                "distribution": {k: distribution(v) for k, v in by_status.items()},
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
