# Lendo product chat and mortgage assistant

Technical documentation | 31 October 2026

## 1. Problem, scope and demonstrated journey

Customers reading about loans often need an explanation, a calculation and help moving the same details into an application form. A useful assistant must keep source conditions attached to its answers, calculate from the customer's actual inputs and preserve their control over the form.

This prototype answers product questions from reviewed English evidence and demonstrates an existing-mortgage refinance journey. It runs on local Qwen by default; the recorded demo uses OpenAI `gpt-4.1` through the same validated interface. The customer asks what a mortgage is, checks an illustrative loan-to-value calculation, supplies refinance details and explicitly requests an application draft. One **Start application** action opens the editable mortgage form with the supported fields filled. The observed journey uses a SEK 2,000,000 mortgage and SEK 2,500,000 property value, giving **80.00% LTV**. It does not submit an application or decide lending eligibility.

The knowledge corpus covers consumer loans, business loans, mortgages, insurance and Kreddy. The demonstrated handoff is Refinance without additional borrowing. A successful example does not establish general conversation reliability, and business impact remains an unmeasured hypothesis.

The project implements an independent FastAPI/LangGraph backend and local integration. It reuses the company's Gatsby interface and previously reviewed capstone source material. The proprietary frontend is kept outside this repository, with 15 saved page contexts for offline browsing. Routes without saved content can be unavailable. This is a local demonstration using synthetic financial values; submission, BankID, credit checks and analytics are blocked.

The [design one-pager](capstone-one-ish-pager.pdf), submitted on 25 October 2026, describes the original hosted/bilingual design and is kept unchanged. The sections below describe the implemented scope and distinguish historical measurements from current browser observations.

Evidence: [current mortgage and session-safety UI verification](experiments/failures-and-pivots.md), [English PDF opening and stale-form recheck](experiments/failures-and-pivots.md).

<!-- TODO: At final recording, record the exact backend/frontend revisions and any uncommitted changes beside the demo runtime; do not label an earlier receipt as verification of later bytes. -->

<div style="page-break-after: always;"></div>

## 2. Source preparation and retrieval

The runtime searches a frozen collection of **1,476 reviewed English records**. These records were prepared from captured public-source material, with original Swedish PDFs retained for provenance. Each record carries product scope, applicability, qualifications, source context and version information. The runtime validates artifact hashes and row order before using the saved indexes. It does not crawl websites or rebuild the corpus during a customer turn.

Search uses two representations of the same records. Pinned BGE-M3 embeddings encode the query; each FAISS HNSW index returns ten candidates. The service unions the candidates and runs the local `bge-reranker-v2-m3` cross-encoder against the cleaned search text. Equivalent evidence is removed, then at most five records are packed within 12,000 characters. The search representation and the richer evidence supplied for answering serve different purposes: concise matching text helps retrieval, while the answer context retains source qualifications.

Public questions go straight to retrieval. The reranker score of the best candidate decides whether the corpus covers the question; below **0.23** the assistant says it could not find the answer without asking the model. The threshold was [calibrated](results/relevance-gate.json) on the audited English cases as the highest value keeping at least 97% of supported cases: **142/146** supported kept, **7/9** unsupported rejected. Partner-specific records are dropped unless the question names the partner, so a broad question cannot adopt one lender's terms. This replaced an earlier lexical product projector, which mapped `What is a business loan?` to contact information and never retrieved `What is Lendo`.

The model selects IDs from the current evidence pack, which holds only records the server can cite. The server checks membership and renders the corresponding reviewed extracts. The displayed answer is extractive; unrestricted synthesis is not used on this path. A valid citation proves where an extract came from, while relevance and missing conditions still require review. The chat exposes source-website and PDF links; it does not display a separate qualifications panel. Source conditions remain part of the retrieval metadata and need review against the linked evidence.

Each answer links the source website and a local English PDF citation. The 16 generated English PDFs keep links to their Swedish originals, and their manifest binds record/page mappings to the frozen corpus hashes. A captured source is evidence for this demonstration, not a guarantee of today's lending policy.

Implementation: [frozen retrieval](../src/grounded_mortgage_assistant/rag/retrieval.py), [public question routing](../src/grounded_mortgage_assistant/rag/public_questions.py), [relevance gate](../src/grounded_mortgage_assistant/rag/answer.py), [extract and citation validation](../src/grounded_mortgage_assistant/rag/public_evidence.py). The [detailed architecture](architecture/detailed.md) gives the connected source map.

<div style="page-break-after: always;"></div>

## 3. Workflow, context and customer control

The [architecture visual](architecture/detailed.html) separates the browser, API, model, private session state and evidence/tools. FastAPI and Pydantic define typed contracts. LangGraph expresses the finite interpretation/action path. LangChain's OpenAI-compatible adapter calls **Qwen3.5-9B-MLX-4bit-938d891** through local oMLX on port 8011, or OpenAI **gpt-4.1** when `LLM_PROVIDER=openai`. The provider is fixed at startup with no fallback, and both pass the same schema, model-identity and budget checks. In hosted mode only the safe projection (masked amounts, opaque candidate handles) and public evidence text leave the machine, after the PII check. The backend and frontend run on ports 8099 and 8001.

Framework choice follows the workflow's needs: bounded actions, explicit state transitions and inspectable failures. Public questions can use the direct scoped retrieval path; personal mortgage turns use the finite graph. The model proposes interpretations or evidence IDs. Python validates those proposals and executes the calculation or draft operation. Framework adoption has not itself demonstrated better answer quality.

Current facts live in session RAM with revisions, status and exact source proofs. Amounts and original proof spans stay on the private server side. The interpreter receives approved symbols, opaque candidate handles and a safe state projection. It chooses subject, conditionality, requested action and fact/scope candidate IDs. A model-selected ID must still pass ownership, membership and source validation before changing memory.

The Qwen tokenizer counts the outgoing context, including tool schemas, against an 8,192-token input budget with 1,800 output tokens reserved; for OpenAI this is an estimate and the returned usage is recorded. Optional complete exchanges can be pruned; authoritative facts remain separate from conversation history. There is no cross-session learning or durable customer database. Request identity, replay checks, reset and expiry protect the session lifecycle.

Financial tools use `Decimal` and current proven inputs. Refinance LTV is existing loan divided by property value; New derives the loan from price minus down payment; Expand adds the requested borrowing to the existing loan. Results retain inputs, formula, units and rounding. These calculations do not establish individual eligibility.

A draft is a snapshot, separate from current facts, and requires an explicit request. For the demonstrated local Refinance path, **Start application** checks the current session, draft lifetime and form snapshot, then fills the supported mortgage fields. A form edit after the card appeared was reproduced as an overwrite bug and repaired; the browser recheck preserved the customer's changed property value. The form remains editable and its final submission stays disabled. Only the Refinance draft is supported; amounts and intent are typed in chat, without guided question buttons.

A separate loopback service integrates Guardrails DetectPII with local Presidio. It has an isolated dependency environment because its dependencies conflict with the backend SDK. Detection errors stop processing; recognized financial source spans are replaced only in a disposable detector input to avoid mistaking legitimate mortgage amounts for identifiers. Question clauses split at commas, so the amounts in `My existing mortgage is 2000000 and property is worth 2500000, what is my loan to value?` are recognized and masked before detection instead of being rejected as personal data. This adds a check, not comprehensive PII coverage.

The [31 October UI check](experiments/failures-and-pivots.md) rejected six synthetic PII cases with zero model tokens, failed closed while the detector was paused, and showed that Start over during a pending answer recovered no old facts.

<div style="page-break-after: always;"></div>

## 4. Comparisons and evaluation evidence

The project compared retrieval approaches before choosing the current design. Results below measure different boundaries and must be read separately.

| Study | Baseline | Alternative | What it supports |
| --- | --- | --- | --- |
| Earlier chunking and embedding study: correct evidence in the top 5, 28 held-out questions | Sentence chunks, E5-small, Flat dense: **25/28** | Whole Q&A, BGE-M3, HNSW32 dense: **26/28** | A one-case retrieval gain; paired uncertainty included zero |
| Fixed-candidate reranker comparison | Local BGE: **129/149** complete supported cases; **193/217** facts | Cohere: **131/149** cases; **195/217** facts | Small net support gain with four case gains and two losses |
| Framework integration parity | Saved frozen retrieval | Rebuilt retrieval: **159/159** exact matches | Candidate/final-evidence preservation for fixed queries |
| Coverage decision | Lexical product projector | Reranker gate at 0.23: **142/146** supported kept, **7/9** unsupported rejected | Retrieval decides coverage without a model call |
| [Model provider](results/browser-suites.json), same browser suites | Local Qwen: journey **16/16**, questions **19/20** (1 run) | OpenAI gpt-4.1: journey **16/16**, questions **19/20** in each of 3 runs | Equal on these suites; OpenAI exposed two defects that Qwen had hidden, fixed before these runs |
| [OpenAI model choice](experiments/model-comparison.md), 3 runs each | gpt-4.1: questions **55/60**, $0.0032 per turn | gpt-5.4-mini (effort none): **57/60**, $0.0014; gpt-4.1-mini: **54/60**, $0.0006 | All 48/48 on the journey at about 1.2 s per turn; differences are within noise, so the demo stays on gpt-4.1 |

The earlier chunking and embedding study used an earlier corpus and agent-authored, non-exhaustive labels; it supports the dense BGE direction, not an answer-quality score. The [reranker comparison](experiments/reranker-comparison.md) kept 159 candidate sets, packing and equivalence rules fixed; 149 had supported-case labels. Local BGE remains the runtime choice because Cohere's net gain was small and included regressions.

The [159-query parity receipt](results/retrieval-parity.json) isolates implementation changes from retrieval changes. The browser suites are exposed functional examples, not unseen acceptance results.

Final quality should be measured over complete journeys, counting every required answer, clarification, calculation and browser action. Errors and timeouts count as failures. Report source support, applicable citations, numeric correctness and forbidden actions separately. The original design targets included 80% journey success and supported-answer coverage, 85% evidence recall@5, and all critical cases passing. Those are targets; this document does not claim they were met by the current local runtime.

The console reports recorded usage and timing, identifies estimates/unknowns and shows $0 for local calls and a list-price estimate for OpenAI calls; local hardware cost is unmeasured.

LLM-as-a-judge was attempted on an earlier hosted configuration study: 549 generated answers across four retrieval arms (159 questions each), graded by a `gpt-6-sol` judge whose calibration agreed on 28/30 acceptance labels. Grading stopped at the project spend limit with **305** valid reviews, **242** quota rejections and **2** inconsistent grades. Valid coverage ranged from 23 to 157 per arm, so no winner is claimed; on the 10 cases graded for every arm, 7 or 8 of 10 answers were fully accepted in each. A follow-up will grade the remaining answers with the same judge and report complete results. [Evaluation report](experiments/llm-judge.md)

<div style="page-break-after: always;"></div>

## 5. Failure analysis and limits

Early tool-selection experiments allowed Qwen to skip retrieval, so the service took ownership of the required retrieval path. Freeform local synthesis also failed the source-support requirement. Selecting current evidence IDs and rendering reviewed extracts made the answer boundary easier to inspect, at the cost of less fluent and sometimes overlapping responses.

The earlier state design mixed conversation facts with application drafts. An amount correction could start draft preparation, and a subsequent LTV question could miss the calculation. A stopped hosted diagnostic using the returned `gpt-5.4-mini-2026-03-17` identity reproduced correction/intent failures. Changing the provider did not repair the state contract. The redesign separated current facts from draft snapshots and made draft preparation explicit. [Saved diagnostic evidence](experiments/reranker-comparison.md)

On 30 October, the frozen `c9ca601` development build completed positive memory steps in three sessions but failed the negated-amount control in each; strict validation rejected its proposals. That historical gate is **0/3**, with later controls unreached. It motivated narrower proposal validation and remains regression evidence. It has not been rerun on the current build, so neither failure nor success on that full gate is asserted for today's runtime. [Historical gate](results/qwen-gate.json)

Browser work exposed failures beyond inference. Start over initially failed because the running frontend proxy was stale; reloading the task-owned server restored the existing allowed reset route. Direct draft handoff initially overwrote a newer form edit; snapshot validation was repaired and checked in the browser. These failures explain why the visible customer path matters alongside module correctness. [Reset and handoff evidence](experiments/failures-and-pivots.md), [stale-edit recheck](experiments/failures-and-pivots.md)

Final testing and a recording rehearsal found eight more defects, each reproduced in the browser and fixed with a test. `without extra borrowing` lost its no-increase proof. `my friend mortgage` raised an error instead of marking third-party scope. Qwen at temperature 0 sometimes labelled the explicit LTV question an acknowledgement; the server now promotes it and records a Routing event. The reversed LTV phrasing was rejected as personal information, and `What is Lendo` never reached retrieval. Finally, 58 of 1,476 records lacked a citable source URL, so an answer selecting one failed closed; this surfaced only when OpenAI selected all five candidates. OpenAI also labelled the stated refinance intent as a calculation after an LTV (2 of 6 runs); a turn without amounts or calculation wording now cannot start one. Stating that intent also marked the unchanged LTV as stale; a calculation now goes stale only when an amount changes. A 20-question browser suite now covers these cases. Two routing limits remain: any digit keeps a question off the public path, so `Who won the football World Cup in 2022?` gets a rephrase request rather than the not-found message, and a bare `What is my loan to value?` retrieves instead of asking for the two amounts.

The engineering console shows projections, retrieval IDs, model identity, usage, cost and stages, including a safe clarification reason. It does not retain every candidate, reranker score or model payload needed for exact replay; debug content stays in RAM and is purged with the session.

This submission describes a working local demonstration and its measured boundaries. General model reliability, final unseen acceptance, mobile visual validation and business impact remain unestablished. The system has no customer-account access, live database connection or financial execution capability. The next quality work is a frozen whole-journey evaluation covering current routing limits and historical negation regressions, with source-based human scoring before relying on an automated judge.

<!-- TODO: Render and visually inspect a final 4-5 page export only when requested. These five Markdown sections set the intended page structure; actual pagination has not been verified. -->
