# Evaluation

Every number on this page comes from exposed development and diagnostic cases written during the project, not from an unseen acceptance set, and several denominators are small. Per-case outcomes, keyed by case ID, are in [results/](results/).

The corpus, questions, answers and logs belong to a Swedish loan comparison company and are not published, so this page gives counts, rates, thresholds and case IDs only. Case IDs containing `kreddy` are about Kreddy, the company's personal finance management (PFM) app. The studies measure different parts of the system, so their numbers should not be added together.

## Timeline

The work happened in three stages. The retrieval study came first, in an earlier capstone phase. The framework build followed, with the judge run, the reranker comparison and the first state and browser checks. Final testing ran the demo journey, calibrated the relevance gate and compared OpenAI models.

| Stage | Work | Result |
| --- | --- | --- |
| Earlier capstone phase | [Chunking, embedding and index study](#earlier-retrieval-study) | Whole Q&A + BGE-M3 + HNSW32 chosen; held-out 26/28 vs 25/28 |
| Framework build | [LLM-as-a-judge](#llm-as-a-judge) on an earlier hosted configuration | Stopped after 305 valid grades of 549 |
| | [Framework integration parity](#retrieval-parity) | 159/159 exact matches |
| | [Local BGE vs Cohere reranker](#reranker-comparison) | 129/149 vs 131/149; local BGE kept |
| | Hosted gpt-5.4-mini state diagnostic | Stopped at a failed correction scenario |
| | [Local Qwen interpretation gate](#failures-and-pivots) on a frozen build | 0/3 complete gates |
| | [Fixed-query retrieval parity receipt](#retrieval-parity) | 159/159, 0 mismatches |
| | Browser checks for draft handoff and stale-edit protection | Repaired and rechecked |
| Final testing | [Post-cleanup demo journey](#browser-suites), local Qwen | 16/16 in 3 consecutive runs |
| | [Relevance gate calibration](#relevance-gate) | 0.23: 142/146 kept, 7/9 rejected |
| | OpenAI provider browser runs, before and after fixes | gpt-4.1: 16/16 and 19/20 in 3 of 3 runs |
| | [OpenAI model comparison](#model-comparison) | All 48/48 on the journey; 55, 54 and 57 of 60 questions |
| | Re-run after the chat-scroll fix | 19/20 twice, 16/16, no timeouts |

The earlier retrieval lab also compared representations, local acceleration, cross-language retrieval, query translation and two-view retrieval. Those comparisons are not summarized here.

## Relevance gate

Data: [relevance-gate.json](results/relevance-gate.json)

Retrieval always returns something, even for questions the corpus cannot answer, and a model handed weakly related evidence may answer anyway. Coverage used to be decided by a lexical product projector, which failed a basic company question (K01 in the browser suite) because that question never reached retrieval. The gate replaced it. If the best reranker score is below a threshold, the assistant says it cannot answer from the reviewed sources, and no model is called.

To pick the threshold, the top reranker score was recorded for each of 159 fixed cases: 146 supported, 9 unsupported, and 4 partial or source-specific cases that count in neither total. Thresholds from 0.01 to 0.95 were swept, and the rule was to take the highest threshold that keeps at least 97% of supported cases.

| Threshold | Supported kept | Unsupported rejected |
| --- | --- | --- |
| 0.01 | 146/146 | 4/9 |
| 0.10 | 144/146 | 7/9 |
| 0.23 (chosen) | 142/146 | 7/9 |
| 0.30 | 139/146 | 7/9 |
| 0.50 | 134/146 | 8/9 |
| 0.73 | 118/146 | 9/9 |

At 0.23, four supported cases are wrongly rejected (DEV-consumer-loan-02, DEV-glossary-02, TEST-business-loan-02, TEST-mortgage-01) and two unsupported cases are wrongly kept (UNSUPPORTED-05 at 0.31, UNSUPPORTED-06 at 0.72). Between 0.04 and 0.31, raising the threshold drops supported cases without rejecting any more unsupported ones. Rejecting all nine unsupported cases would cost 28 supported ones.

With only nine unsupported cases, 7/9 is a rough estimate. The gate decides coverage and says nothing about answer quality or citation correctness. The full sweep and per-case scores are in the JSON.

## Reranker comparison

Data: [reranker-comparison.json](results/reranker-comparison.json)

The runtime reranks retrieved candidates with a local model, BAAI bge-reranker-v2-m3. The question was whether Cohere Rerank v5 Pro would recover enough extra supported facts to justify a network dependency and a per-request cost. Both rerankers saw the same 159 fixed candidate sets. Candidate IDs and order, query and document strings, duplicate handling, five-record packing and the 12,000-character budget were held fixed, so only the reranker changed. 149 cases had supported-fact labels, with 217 required facts between them.

| | Local BGE | Cohere |
| --- | --- | --- |
| Supported cases with all facts recovered | 129/149 | 131/149 |
| Required facts recovered | 193/217 | 195/217 |
| Median / p95 latency | 219 / 796 ms | 295 / 361 ms |

The candidate ceiling, the best any reranker could do with these candidates, was 136/149 cases and 201/217 facts. Cohere gained complete support on DEV-consumer-loan-01, SF-B05, SF-C06 and SF-M09, and lost it on SF-C02 and gap-DEMO-INCOME-004. At fact level it gained 6 case-fact pairs and lost 4, and SF-M01 swapped one recovered fact for another. The 159 Cohere requests cost about $0.012 (a buffered estimate, not an invoice).

These are support-recovery lower bounds on exposed controls. No answers were generated from Cohere's evidence. The latencies come from one run each and measure different things: warm local GPU prediction against HTTP over the network. Cohere's API did not return a model identity, so the model version is unverified.

The local BGE reranker stayed. Cohere's net gain was two cases, and it came with regressions.

## Earlier retrieval study

Data: [retrieval-study.json](results/retrieval-study.json)

Before building the conversation, the capstone compared ways to split the corpus into passages, embedding models and vector indexes. The baseline used sentence chunks (256 tokens, 32 overlap), E5-small embeddings and a flat index. The selected configuration kept each reviewed question-and-answer record whole, with BGE-M3 embeddings and an HNSW index (M=32). Both used dense retrieval. 195 development configurations reached a terminal result. Four tied on development quality, and the one with the lowest p95 latency was frozen before the held-out questions were run.

Complete labelled evidence in the top 5:

| Split | Selected | Baseline |
| --- | --- | --- |
| Development | 52/53 | 47/53 |
| Held-out | 26/28 | 25/28 |
| Historical diagnostics | 45/57 | 45/57 |

On the held-out split, TEST-business-loan-03 was gained, nothing was lost, and both configurations missed TEST-business-loan-01 and TEST-kreddy-01. The paired 95% interval for the held-out gain runs from 0.0 to +11.11 percentage points. The selected configuration was slower (held-out p95 36.18 ms vs 5.50 ms), which is still small.

The study measured retrieval only. An agent wrote and source-checked the labels, and they are not exhaustive, so an answer found in another valid source can count as a miss. The corpus snapshot had 1,477 records, one more than the current runtime's 1,476. The project adopted whole-record BGE-M3 dense retrieval, but the held-out gain is one case and its interval includes zero, so the study supports that choice only modestly. Reranking and the current two-view retrieval came later and were evaluated separately in the [reranker comparison](#reranker-comparison).

## Retrieval parity

Data: [retrieval-parity.json](results/retrieval-parity.json)

The framework build reimplemented retrieval, so it was checked against the saved frozen retrieval on 159 fixed queries. The check compares final evidence IDs, packed texts and the top-10 candidate IDs from each of the two dense views. The integration check matched 159 of 159, and the later fixed-query receipt found 0 mismatches. This isolates implementation changes and says nothing about retrieval quality. Both receipts predate the retriever returning scores, so a re-run is still open.

## Browser suites

Data: [browser-suites.json](results/browser-suites.json)

Scripted Playwright runs go through the real chat UI and engineering console. The 16-check demo journey covers a cited knowledge answer, the LTV card, refinance intent, the draft, the form handoff, a privacy rejection, the console and Start over. The 20-question suite has knowledge, LTV, journey, unsupported and privacy questions. The JSON lists every run per provider, with checks and questions by ID.

On the final build, OpenAI gpt-4.1 passed 16/16 on the journey and 19/20 on the questions in 3 of 3 runs. Local Qwen did the same in one run, and the post-cleanup local Qwen journey passed 16/16 in 3 consecutive runs. The one failing question in every run is U02, a known routing limit described under [Failures and pivots](#failures-and-pivots).

## Model comparison

Data: [model-comparison.json](results/model-comparison.json)

With OpenAI, the model only interprets turns and picks evidence IDs through tool calls. The server retrieves, calculates and renders. The comparison tested whether a cheaper model could do that narrow job. Each model ran the 16-check journey and the 20-question suite 3 times through the real UI, with the same backend and frontend. Latency, cost and evidence counts come from saved console traces.

| Model | Journey | Questions | Model misses | UI timeouts | Median / p90 turn | Cost per turn | Evidence per answer |
| --- | --- | --- | --- | --- | --- | --- | --- |
| gpt-4.1 | 48/48 | 55/60 | L01 | P01 | 1.19 / 1.63 s | $0.0032 | 2.10 |
| gpt-4.1-mini | 48/48 | 54/60 | L04 | K04, K05 | 1.34 / 1.64 s | $0.0006 | 2.16 |
| gpt-5.4-mini (effort none) | 48/48 | 57/60 | none | none | 1.20 / 1.65 s | $0.0014 | 1.83 |

U02 failed in every run for every model because of a routing rule. In each model miss, the proposal for an explicit LTV question failed server validation, so the user got a clarification request instead of a card and nothing incorrect was shown. The UI timeouts happened when the chat closed itself on scripted scroll events before the harness typed, with no model call involved. After the frontend fix, gpt-4.1 scored 19/20 twice and 16/16. Evidence per answer is the mean number of records selected per answered product question, which measures selection width and says nothing about correctness. gpt-5.4-mini with effort low could not run, because Chat Completions rejects function tools for this model with any effort other than none. The whole comparison cost about $0.31 at list prices.

The demo stays on gpt-4.1. gpt-5.4-mini looked slightly better and cheaper, but with 9 runs per model, one miss against none is within noise.

## LLM-as-a-judge

Data: [llm-judge.json](results/llm-judge.json)

Retrieval metrics show whether the right evidence was found. They do not show whether the answer built on it is complete, grounded and properly cited, and grading hundreds of answers by hand is slow. A calibrated LLM judge grades each answer on fixed dimensions (completeness, grounding, citation locality, calibration, language, source qualification, dialogue context) and gives an overall "fully accepted" label.

An earlier hosted configuration generated answers for 159 questions in each of four retrieval setups (arms), giving 549 unique answers that cover 636 arm-question evaluations. Before use, the `gpt-6-sol` judge agreed with reference labels on 28 of 30 acceptance decisions.

The grading stopped when the OpenAI spend limit was hit, after 305 valid grades out of 549 answers, so it cannot pick a winner yet. The quota rejected another 242 requests (HTTP 429), and 2 grades were internally inconsistent and count as invalid. A quota rejection is a missing grade, not a failed answer. Valid coverage per arm ranged from 23 to 157, and grading did not stop at random, so comparing acceptance rates across arms would be unfair. Only 10 questions were graded for every arm, and on those, 7 or 8 of 10 answers were fully accepted in each arm. All 10 are answerable, exposed development or diagnostic cases, which is far too few to separate the arms across 159 questions.

The grades are model judgments and should be treated as provisional. The judge model also generated the answers, and the reference labels were agent adjudications.

The remaining answers will be graded with the same judge. The 244 missing judgments (242 quota rejections plus 2 inconsistent) will use the same model, prompt and schema, so all 636 evaluations share one judge. Switching judges would mean calibrating a new one and regrading all 549 answers.

## Failures and pivots

Data: [qwen-gate.json](results/qwen-gate.json), [browser-suites.json](results/browser-suites.json)

Four problems changed the design:

- When Qwen chose its own tools, it sometimes skipped retrieval. The service now owns the retrieval path.
- Freeform synthesis failed source support. The model now picks evidence IDs, and the server renders reviewed extracts with validated citations. Answers are easier to check and less fluent.
- Facts were mixed with drafts. An amount correction could start a draft, and a later LTV question could miss the calculation. A hosted gpt-5.4-mini diagnostic failed the same way, which put the problem in the state contract rather than the provider. Facts and draft snapshots now have separate lifecycles, and drafting needs an explicit request.
- The local Qwen gate on a frozen build scored 0/3. Every run completed the positive journey and failed at the negated-amount step, which led to narrower proposal validation. The gate has not been rerun on the current build.

Browser testing found these defects. Each was reproduced in the real UI and fixed with a test:

- Start over failed behind a stale frontend proxy.
- An old draft could overwrite a newer form edit.
- No-increase phrasings lost their proof, so the draft was refused.
- A third-party mortgage raised an error instead of being marked out of scope.
- Qwen sometimes labelled the explicit LTV question an acknowledgement. The server now promotes it.
- The reversed LTV phrasing was rejected as personal information.
- K01 never reached retrieval. The [relevance gate](#relevance-gate) replaced the lexical projector.
- 58 of 1,476 records lacked a citable source URL, so K05 failed closed in 2 of 3 early OpenAI runs. Only citable records are offered now.
- OpenAI labelled the refinance intent a calculation in 2 of 6 runs. A turn without amounts or calculation wording can no longer start one.
- Stating that intent marked the unchanged LTV card stale. It now goes stale only when an amount changes.
- The chat minimized on scripted scrolls, causing harness timeouts. It now responds only to pointer or wheel input.

A final UI check rejected six synthetic PII cases with an empty trace question and answer and zero model tokens. It failed closed while the detector was paused and recovered on retry, and pressing Start over during a pending answer brought back none of the old facts.

Two limits are still open. Any digit keeps a question off the public path, so U02 gets a rephrase request instead of the not-found message. A bare LTV question without amounts retrieves instead of asking for them.
