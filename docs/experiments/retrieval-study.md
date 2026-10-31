# Retrieval study: chunking, embeddings and index

**When:** 14-15 October 2026, in the earlier capstone phase. **Data:** [retrieval-study.json](../results/retrieval-study.json)

## Why it was run

An assistant that cites sources can only be as good as the evidence it retrieves. Before building the conversation, the project compared how to split the corpus into passages, which embedding model to use and which vector index to search.

## What was compared

- **Baseline:** sentence chunks (256 tokens, 32 overlap), E5-small embeddings, a flat index, dense retrieval.
- **Selected:** each reviewed question-and-answer record kept whole, BGE-M3 embeddings, an HNSW index (M=32), dense retrieval.

195 development configurations reached a terminal result. Four tied on development quality, and the one with the lowest p95 latency was frozen before the held-out questions were run.

## Results: complete labelled evidence in the top 5

| Split | Selected | Baseline |
| --- | --- | --- |
| Development | 52/53 | 47/53 |
| Held-out | 26/28 | 25/28 |
| Historical diagnostics | 45/57 | 45/57 |

On the held-out split, TEST-business-loan-03 was gained, nothing was lost, and TEST-business-loan-01 and TEST-kreddy-01 were missed by both. The paired 95% interval for the held-out gain runs from 0.0 to +11.11 percentage points. The selected configuration was slower (held-out p95 36.18 ms vs 5.50 ms), which is still small.

## What it does not show

- It measures retrieval, not answer quality.
- An agent wrote and source-checked the labels, and they are not exhaustive. An answer found in another valid source can count as a miss.
- The corpus snapshot (1,477 records) was earlier than the current runtime's (1,476).
- The held-out gain is one case, and its interval includes zero.
- Reranking and the current two-view retrieval came later and were not part of this study.

## What changed

The project adopted whole-record BGE-M3 dense retrieval. The result supports that direction modestly, not decisively. The runtime later added a second view and a local reranker, evaluated separately ([reranker comparison](reranker-comparison.md)).
