# Timeline

Dates come from the evidence: report headings, recorded UTC timestamps, dated evidence folders and saved result files. Times on 31 October are the local save times of the result files and only show the order of work.

## Earlier capstone phase (October 2026)

| Date | Experiment | Result |
| --- | --- | --- |
| 14-15 Oct | Chunking, embedding and index study ([summary](retrieval-study.md)) | Whole Q&A + BGE-M3 + HNSW32 chosen; held-out 26/28 vs 25/28 |

The retrieval lab also holds comparisons in folders dated 13 to 25 October (representations, local acceleration, cross-language retrieval, query translation, two-view retrieval). They are not summarized here.

## 29 to 31 October 2026

| Date | Experiment | Result |
| --- | --- | --- |
| 29 Oct | LLM-as-a-judge: answer generation (17:53 to 17:56 UTC) and integrity summary (18:15 UTC) ([summary](llm-judge.md)) | Incomplete: 305 valid of 549 judge requests |
| 29 Oct | Framework integration parity (22:08 UTC) | 159/159 exact matches |
| Night of 29-30 Oct | Local BGE vs Cohere reranker comparison, finished before 02:55 UTC ([summary](reranker-comparison.md)) | 129/149 vs 131/149; local BGE kept |
| Night of 29-30 Oct | Hosted gpt-5.4-mini state diagnostic ([failures](failures-and-pivots.md)) | Stopped at a failed correction scenario |
| 30 Oct | Local Qwen interpretation gate on a frozen build | 0/3 complete gates |
| 30 Oct | Fixed-query retrieval parity receipt | 159/159, 0 mismatches |
| 30 Oct | Browser receipts for draft handoff and stale-edit protection | Repaired and rechecked |
| 31 Oct (about 03:12) | Post-cleanup demo journey, local Qwen | 16/16 in 3 consecutive runs |
| 31 Oct (about 04:53) | Relevance gate calibration ([summary](relevance-gate.md)) | 0.23: 142/146 kept, 7/9 rejected |
| 31 Oct (by 12:49) | OpenAI provider browser runs, before and after fixes | gpt-4.1: 16/16 and 19/20 in 3 of 3 runs |
| 31 Oct (13:23 to 13:44) | OpenAI model comparison ([summary](model-comparison.md)) | All 48/48 on the journey; 55, 54 and 57 of 60 questions |
| 31 Oct (13:51) | Re-run after the chat-scroll fix | 19/20 twice, 16/16, no timeouts |

Data: [`../results/`](../results/)
