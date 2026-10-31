# Experiments

Short, plain-English summaries of how the mortgage assistant was evaluated, what each study shows, and what it does not. Every number comes from the project's private evidence, which holds the Lendo-owned corpus, questions, answers and logs. That evidence is not published, so these pages give counts, rates, thresholds and case IDs with outcomes only. Machine-readable versions are in [`../results/`](../results/).

| Page | Question it answers | Data |
| --- | --- | --- |
| [Timeline](timeline.md) | When each experiment ran | - |
| [Retrieval study](retrieval-study.md) | Did whole-Q&A chunks with BGE-M3 retrieve better than sentence chunks with E5-small? | [retrieval-study.json](../results/retrieval-study.json) |
| [Reranker comparison](reranker-comparison.md) | Is Cohere Rerank worth replacing the local BGE reranker? | [reranker-comparison.json](../results/reranker-comparison.json) |
| [Relevance gate](relevance-gate.md) | How does the system decide, without a model call, that the corpus cannot answer? | [relevance-gate.json](../results/relevance-gate.json) |
| [Model comparison](model-comparison.md) | Which OpenAI model should the demo use? | [model-comparison.json](../results/model-comparison.json) |
| [LLM-as-a-judge](llm-judge.md) | How good were generated answers? (incomplete) | [llm-judge.json](../results/llm-judge.json) |
| [Failures and pivots](failures-and-pivots.md) | What broke, and what changed because of it | [qwen-gate.json](../results/qwen-gate.json), [browser-suites.json](../results/browser-suites.json) |

Two further result files support these pages:

- [retrieval-parity.json](../results/retrieval-parity.json): the rebuilt framework retrieval reproduced the saved frozen retrieval exactly for 159 of 159 fixed queries (final evidence IDs, packed texts and both candidate views). This checks implementation, not quality.
- [browser-suites.json](../results/browser-suites.json): every run of the 16-check demo journey and the 20-question suite through the real UI, per provider, by ID.

## How to read these results

The studies measure different boundaries and should not be added together. Most use exposed development cases, not an unseen acceptance set, and several have small denominators. Where a result is incomplete or a difference is within noise, the page says so. No general model-quality or final acceptance result is claimed.
