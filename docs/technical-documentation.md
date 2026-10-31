# Product chat and mortgage assistant

Technical documentation | October 2026

## 1. Problem, scope and demonstrated journey

This project was built for a Swedish loan comparison company. Mortgages, consumer loans, business loans and insurance come with complex rules and conditions, and a customer reading about a loan often needs an explanation, a calculation and help moving the same details into an application form. An AI assistant that can answer basic to complex questions can take this work off customer agents, so they can focus on handling applications and converting them. The assistant described here is meant to help customers through those rules. It has to keep source conditions attached to its answers, calculate from the customer's own inputs and leave the customer in control of the form.

The prototype answers product questions from reviewed English evidence and demonstrates an existing-mortgage refinance journey. It runs on local Qwen by default, and the recorded demo uses OpenAI `gpt-4.1` through the same validated interface. The customer asks what a mortgage is, checks an illustrative loan-to-value (LTV) calculation, gives refinance details and explicitly asks for an application draft. One Start application action opens the editable mortgage form with the supported fields filled in. With a SEK 2,000,000 mortgage and a SEK 2,500,000 property value, the observed LTV is 80.00%. The assistant does not submit an application or decide lending eligibility.

The knowledge corpus covers consumer loans, business loans, mortgages, insurance and Kreddy, the company's personal finance management (PFM) app. The only demonstrated handoff is Refinance without additional borrowing. One successful example does not show that conversations are reliable in general, and business impact is still an unmeasured hypothesis.

The project adds an independent FastAPI/LangGraph backend and a local integration. It reuses the company's Gatsby interface and capstone source material that had already been reviewed. The proprietary frontend stays outside this repository, with 15 saved page contexts for offline browsing, so routes without saved content may be unavailable. Everything runs locally on synthetic financial values, and submission, BankID, credit checks and analytics are blocked.

The [design one-pager](capstone-one-ish-pager.pdf) planned a hosted, bilingual assistant. The sections below describe what was implemented and keep historical measurements apart from current browser observations.

Evidence: [current mortgage and session-safety UI verification, English PDF opening and stale-form recheck](evaluation.md#failures-and-pivots).

<div style="page-break-after: always;"></div>

## 2. Source preparation and retrieval

The runtime searches a frozen collection of 1,476 reviewed English records, prepared from captured public-source material. The original Swedish PDFs are kept for provenance. Each record carries product scope, applicability, qualifications, source context and version information. Before using the saved indexes, the runtime validates artifact hashes and row order. It does not crawl websites or rebuild the corpus during a customer turn.

Search uses two representations of the same records. Pinned BGE-M3 embeddings encode the query, and each FAISS HNSW index returns ten candidates. The service unions the candidates and scores them with the local `bge-reranker-v2-m3` cross-encoder against the cleaned search text. Equivalent evidence is removed, and at most five records are packed within 12,000 characters. The concise search text helps matching, and the richer answer context keeps the source qualifications.

Public questions go straight to retrieval. The reranker score of the best candidate decides whether the corpus covers the question. Below 0.23, the assistant says it could not find the answer and does not call the model. The threshold was [calibrated](results/relevance-gate.json) on the audited English cases as the highest value that keeps at least 97% of supported cases. At that value, 142 of 146 supported cases were kept and 7 of 9 unsupported cases were rejected ([details](evaluation.md#relevance-gate)). Partner-specific records are dropped unless the question names the partner, so a broad question cannot pick up one lender's terms. The gate replaced an earlier lexical product projector, which mapped `What is a business loan?` to contact information and never retrieved the record describing the company itself.

The model selects IDs from the current evidence pack, which holds only records the server can cite. The server checks membership and renders the matching reviewed extracts, so the displayed answer is extractive. Unrestricted synthesis is not used on this path. A valid citation proves where an extract came from, but relevance and missing conditions still need review. The chat shows source-website and PDF links and has no separate qualifications panel. Source conditions stay in the retrieval metadata and should be checked against the linked evidence.

Each answer links the source website and a local English PDF citation. The 16 generated English PDFs link to their Swedish originals, and their manifest binds record and page mappings to the frozen corpus hashes. A captured source is evidence for this demonstration only and may not match today's lending policy.

Implementation: [frozen retrieval](../src/grounded_mortgage_assistant/rag/retrieval.py), [public question routing](../src/grounded_mortgage_assistant/rag/public_questions.py), [relevance gate](../src/grounded_mortgage_assistant/rag/answer.py), [extract and citation validation](../src/grounded_mortgage_assistant/rag/public_evidence.py). The [architecture document](architecture.md) maps how these parts connect.

<div style="page-break-after: always;"></div>

## 3. Workflow, context and customer control

The [architecture visual](architecture/detailed.html) separates the browser, the API, the model, private session state, and evidence and tools. An interactive version is live at `https://kapilashrami.github.io/grounded-mortgage-assistant/architecture/detailed.html`. FastAPI and Pydantic define typed contracts, and LangGraph expresses the finite path from interpretation to action. LangChain's OpenAI-compatible adapter calls Qwen3.5-9B-MLX-4bit-938d891 through local oMLX on port 8011, or OpenAI gpt-4.1 when `LLM_PROVIDER=openai`. The provider is fixed at startup with no fallback, and both providers pass the same schema, model-identity and budget checks. In hosted mode, only the safe projection (masked amounts and opaque candidate handles) and public evidence text leave the machine, and only after the PII check. The backend runs on port 8099 and the frontend on port 8001.

The frameworks fit what the workflow needs: bounded actions, explicit state transitions and failures that can be inspected. Public questions can take the direct scoped retrieval path, while personal mortgage turns go through the finite graph. The model proposes interpretations or evidence IDs, and Python validates them before it runs the calculation or draft operation. Adopting these frameworks has not by itself shown better answer quality.

Current facts live in session RAM with revisions, status and exact source proofs. Amounts and the original proof spans stay on the private server side. The interpreter receives approved symbols, opaque candidate handles and a safe state projection, and it chooses the subject, conditionality, requested action and fact or scope candidate IDs. A model-selected ID must still pass ownership, membership and source validation before memory changes.

The Qwen tokenizer counts the outgoing context, tool schemas included, against an 8,192-token input budget with 1,800 output tokens reserved. For OpenAI the count is an estimate, and the usage the API returns is recorded. Optional complete exchanges can be pruned, and authoritative facts are kept apart from conversation history. Nothing is learned across sessions, and there is no durable customer database. Request identity, replay checks, reset and expiry protect the session lifecycle.

Financial tools use `Decimal` and only current proven inputs. For Refinance, LTV is the existing loan divided by the property value. New derives the loan from price minus down payment, and Expand adds the requested borrowing to the existing loan. Each result keeps its inputs, formula, units and rounding. The calculations do not establish individual eligibility.

A draft is a snapshot kept apart from current facts, and it is only created on an explicit request. On the demonstrated local Refinance path, Start application checks the current session, the draft lifetime and the form snapshot, then fills the supported mortgage fields. A bug that overwrote form edits made after the card appeared was reproduced and fixed, and a browser recheck kept the customer's changed property value. The form stays editable and final submission stays disabled. Only the Refinance draft is supported, and amounts and intent are typed in chat, with no guided question buttons.

A separate loopback service runs Guardrails DetectPII with local Presidio. It has an isolated dependency environment because of conflicts with the backend SDK. Detection errors stop processing. Recognized financial source spans are replaced only in a throwaway copy of the detector input, so legitimate mortgage amounts are not mistaken for identifiers. Question clauses are split at commas, so in `My existing mortgage is 2000000 and property is worth 2500000, what is my loan to value?` the amounts are recognized and masked before detection, and the question is not rejected as personal data. This adds one check and does not give full PII coverage.

A [UI check](evaluation.md#failures-and-pivots) rejected six synthetic PII cases with zero model tokens and failed closed while the detector was paused. It also showed that pressing Start over during a pending answer brought back none of the old facts.

<div style="page-break-after: always;"></div>

## 4. Comparisons and evaluation evidence

The project compared retrieval approaches before choosing the current design. The results measure different boundaries and should be read separately.

| Study | Baseline | Alternative | What it supports |
| --- | --- | --- | --- |
| [Earlier chunking and embedding study](evaluation.md#earlier-retrieval-study): correct evidence in the top 5, 28 held-out questions | Sentence chunks, E5-small, Flat dense: 25/28 | Whole Q&A, BGE-M3, HNSW32 dense: 26/28 | A one-case retrieval gain; paired uncertainty included zero |
| [Fixed-candidate reranker comparison](evaluation.md#reranker-comparison) | Local BGE: 129/149 complete supported cases; 193/217 facts | Cohere: 131/149 cases; 195/217 facts | Small net support gain with four case gains and two losses |
| [Framework integration parity](evaluation.md#retrieval-parity) | Saved frozen retrieval | Rebuilt retrieval: 159/159 exact matches | Candidate and final evidence preserved for fixed queries |
| [Coverage decision](evaluation.md#relevance-gate) | Lexical product projector | Reranker gate at 0.23: 142/146 supported kept, 7/9 unsupported rejected | Retrieval decides coverage without a model call |
| [Model provider](results/browser-suites.json), same [browser suites](evaluation.md#browser-suites) | Local Qwen: journey 16/16, questions 19/20 (1 run) | OpenAI gpt-4.1: journey 16/16, questions 19/20 in each of 3 runs | Equal on these suites; OpenAI exposed two defects that Qwen had hidden, fixed before these runs |
| [OpenAI model choice](evaluation.md#model-comparison), 3 runs each | gpt-4.1: questions 55/60, $0.0032 per turn | gpt-5.4-mini (effort none): 57/60, $0.0014; gpt-4.1-mini: 54/60, $0.0006 | All 48/48 on the journey at about 1.2 s per turn; differences are within noise, so the demo stays on gpt-4.1 |

The earlier chunking and embedding study used an earlier corpus and non-exhaustive labels written by an agent. It supports the move to dense BGE retrieval and gives no answer-quality score. The [reranker comparison](evaluation.md#reranker-comparison) kept 159 candidate sets, packing and equivalence rules fixed, and 149 of those sets had supported-case labels. Local BGE remains the runtime choice because Cohere's net gain was small and came with regressions.

The [159-query parity receipt](results/retrieval-parity.json) separates implementation changes from retrieval changes. The browser suites are functional examples the system was developed against, so they are not unseen acceptance results.

Final quality should be measured over complete journeys, counting every required answer, clarification, calculation and browser action, with errors and timeouts counted as failures. Source support, applicable citations, numeric correctness and forbidden actions should be reported separately. The original design targets were 80% journey success and supported-answer coverage, 85% evidence recall@5, and all critical cases passing. These remain targets, and this document does not claim the current local runtime meets them.

The console reports recorded usage and timing, marks estimates and unknowns, and shows $0 for local calls and a list-price estimate for OpenAI calls. Local hardware cost is not measured.

LLM-as-a-judge was tried on an earlier hosted configuration study. It covered 549 generated answers across four retrieval arms (159 questions each), graded by a `gpt-6-sol` judge whose calibration agreed on 28 of 30 acceptance labels. Grading stopped at the project spend limit with 305 valid reviews, 242 quota rejections and 2 inconsistent grades. Valid coverage ranged from 23 to 157 reviews per arm, so no winner is claimed. On the 10 cases graded for every arm, 7 or 8 of 10 answers were fully accepted in each arm. A follow-up will grade the remaining answers with the same judge and report complete results ([evaluation report](evaluation.md#llm-as-a-judge)).

<div style="page-break-after: always;"></div>

## 5. Failure analysis and limits

In early tool-selection experiments, Qwen could skip retrieval, so the service took over the required retrieval step. Freeform local synthesis also failed the source-support requirement. Selecting current evidence IDs and rendering reviewed extracts made the answer boundary easier to inspect, at the cost of less fluent and sometimes overlapping answers.

The earlier state design mixed conversation facts with application drafts. An amount correction could start draft preparation, and a following LTV question could miss the calculation. A stopped hosted diagnostic using the returned `gpt-5.4-mini-2026-03-17` identity reproduced the correction and intent failures, which showed that changing the provider did not repair the state contract. The redesign separated current facts from draft snapshots and made draft preparation explicit ([saved diagnostic evidence](evaluation.md#reranker-comparison)).

A frozen `c9ca601` development build completed the positive memory steps in three sessions but failed the negated-amount control in each one, and strict validation rejected its proposals. That historical gate scored 0/3, and the later controls were never reached. It led to narrower proposal validation and is kept as regression evidence. It has not been rerun on the current build, so this document claims neither failure nor success on the full gate for today's runtime ([historical gate](results/qwen-gate.json)).

Browser work exposed failures outside inference. Start over first failed because the running frontend proxy was stale, and reloading the task-owned server restored the existing allowed reset route. Direct draft handoff first overwrote a newer form edit, and snapshot validation was repaired and checked in the browser. These failures are why the project tests the visible customer path as well as module correctness ([reset, handoff and stale-edit evidence](evaluation.md#failures-and-pivots)).

Final testing and a recording rehearsal found eight more defects. Each was reproduced in the browser and fixed with a test:

- `without extra borrowing` lost its no-increase proof.
- `my friend mortgage` raised an error instead of marking third-party scope.
- Qwen at temperature 0 sometimes labelled the explicit LTV question an acknowledgement. The server now promotes it and records a Routing event.
- The reversed LTV phrasing was rejected as personal information.
- The question asking what the company is never reached retrieval.
- 58 of the 1,476 records had no citable source URL, so an answer that selected one failed closed. This only surfaced when OpenAI selected all five candidates.
- OpenAI labelled the stated refinance intent as a calculation after an LTV in 2 of 6 runs. A turn without amounts or calculation wording can no longer start one.
- Stating that intent also marked the unchanged LTV as stale. A calculation now goes stale only when an amount changes.

A 20-question browser suite now covers these cases. Two routing limits remain. Any digit keeps a question off the public path, so `Who won the football World Cup in 2022?` gets a rephrase request instead of the not-found message. A bare `What is my loan to value?` also goes to retrieval when it should ask for the two amounts.

The engineering console shows projections, retrieval IDs, model identity, usage, cost and stages, including a safe clarification reason. It does not keep every candidate, reranker score or model payload, so exact replay is not possible. Debug content stays in RAM and is purged with the session.

This document describes a working local demonstration and the limits of what was measured. General model reliability, final unseen acceptance, mobile visual validation and business impact are not established. The system has no access to customer accounts, no live database connection and no ability to execute financial transactions. The next quality work is a frozen whole-journey evaluation that covers the current routing limits and the historical negation regressions, scored by humans against the sources before an automated judge is relied on.
