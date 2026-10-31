# Relevance gate

**When:** 31 October 2026. **Data:** [relevance-gate.json](../results/relevance-gate.json)

## Why it exists

Retrieval always returns something, even for questions the corpus cannot answer. Without a coverage check, the model is handed weakly related evidence and may answer anyway. Earlier, coverage was decided by a lexical product projector. It failed a basic company question (K01 in the browser suite) because that question never reached retrieval. The gate replaces it: if the best reranker score is below a threshold, the assistant says it cannot answer from the reviewed sources. No model call is needed.

## How the threshold was chosen

The top reranker score was recorded for each of 159 fixed cases: 146 supported, 9 unsupported, and 4 partial or source-specific cases that count in neither total. Thresholds from 0.01 to 0.95 were swept. The recorded rule is to take the highest threshold that keeps at least 97% of supported cases.

| Threshold | Supported kept | Unsupported rejected |
| --- | --- | --- |
| 0.01 | 146/146 | 4/9 |
| 0.10 | 144/146 | 7/9 |
| **0.23 (chosen)** | **142/146** | **7/9** |
| 0.30 | 139/146 | 7/9 |
| 0.50 | 134/146 | 8/9 |
| 0.73 | 118/146 | 9/9 |

At 0.23, four supported cases are wrongly rejected (DEV-consumer-loan-02, DEV-glossary-02, TEST-business-loan-02, TEST-mortgage-01). Two unsupported cases are wrongly kept (UNSUPPORTED-05 at 0.31, UNSUPPORTED-06 at 0.72). Between 0.04 and 0.31, a higher threshold drops supported cases without rejecting any more unsupported ones. Rejecting all nine unsupported cases would cost 28 supported ones.

## What it does not show

- There are only nine unsupported cases, so 7/9 is a rough estimate.
- The cases are exposed development and diagnostic cases, not a held-out set.
- The gate decides coverage, not answer quality or citation correctness.

The full sweep and per-case scores, keyed by case ID, are in the JSON.
