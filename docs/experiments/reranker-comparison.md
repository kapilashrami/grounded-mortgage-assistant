# Reranker comparison: local BGE vs Cohere

**When:** overnight run finishing on 30 October 2026. **Data:** [reranker-comparison.json](../results/reranker-comparison.json)

## Why it was run

The runtime reranks retrieved candidates with a local model (BAAI bge-reranker-v2-m3). A hosted reranker might pick better evidence. The question was whether Cohere Rerank v5 Pro would recover enough extra supported facts to justify a network dependency and a per-request cost.

## How it was kept fair

Both rerankers saw the same 159 fixed candidate sets. Candidate IDs and order, query and document strings, duplicate handling, five-record packing and the 12,000-character budget were held fixed. Only the reranker changed. 149 cases had supported-fact labels (217 required facts).

## Results

| | Local BGE | Cohere |
| --- | --- | --- |
| Supported cases with all facts recovered | 129/149 | 131/149 |
| Required facts recovered | 193/217 | 195/217 |
| Median / p95 latency | 219 / 796 ms | 295 / 361 ms |

The candidate ceiling, the best any reranker could do with these candidates, was 136/149 cases and 201/217 facts.

Cohere gained complete support on DEV-consumer-loan-01, SF-B05, SF-C06 and SF-M09, and lost it on SF-C02 and gap-DEMO-INCOME-004. At fact level it gained 6 case-fact pairs and lost 4. SF-M01 exchanged one recovered fact for another. Cohere cost about $0.012 for 159 requests (a buffered estimate, not an invoice).

## What it does not show

- These are support-recovery lower bounds on exposed controls, not answer accuracy. No answers were generated from Cohere's evidence.
- The latencies cover different things: warm local GPU prediction versus HTTP over the network, from one run each.
- Cohere's API did not return a model identity, so the model version is unverified.

## Decision

The local BGE reranker stays. Cohere's net gain was two cases, and it came with regressions, not a uniform improvement.
