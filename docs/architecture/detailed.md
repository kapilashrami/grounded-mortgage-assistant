# Detailed architecture

The [interactive diagram](detailed.html) accompanies the [technical documentation](../technical-documentation.md). This map describes the runtime and the demonstrated Refinance handoff.

```mermaid
flowchart TD
    UI[Gatsby chat and form, port 8001] --> API[FastAPI memory service, port 8099]
    API --> PII[Local PII gate: Guardrails DetectPII and Presidio, port 8097]
    PII -->|allow or reject decision| API
    API -->|public question| RETRIEVE[Frozen BGE-M3 and FAISS: 10 + 10 candidates]
    API -->|personal mortgage turn| GRAPH[Finite LangGraph: interpret then act]
    GRAPH -->|safe symbols and opaque candidates| LLM[LLM provider fixed at startup: Qwen3.5-9B via oMLX :8011, or OpenAI gpt-4.1]
    LLM -->|proposals only| VALIDATE[Schema, model identity, ownership and exact proof validation]
    VALIDATE --> STATE[Session RAM: current facts, revisions and source proofs]
    STATE --> TOOLS[Decimal calculation or explicit draft preparation]
    GRAPH -->|knowledge action| RETRIEVE
    RETRIEVE --> RERANK[Local BGE rerank, equivalence removal, pack up to five]
    FILES[Frozen reviewed records and saved source PDFs] --> RETRIEVE
    RERANK --> GATE{Top reranker score >= 0.23?}
    GATE -->|no| NOTFOUND[Not found in the available Lendo information]
    GATE -->|yes, citable records only| SELECT[LLM selects current evidence IDs]
    SELECT --> RENDER[Validate and render extracts and English PDF citations]
    TOOLS --> RESPONSE[Accepted server-memory commit and response]
    RENDER --> RESPONSE
    RESPONSE --> UI
    UI -->|explicit Start application, current snapshot| FORM[Editable local Refinance form]
    API --> CONSOLE[In-memory engineering console: Flow, evidence, usage, outcomes]
```

## Ownership and boundaries

The API factory is `grounded_mortgage_assistant.app:create_app`. `LLM_PROVIDER` selects local Qwen (default) or OpenAI once at startup. Startup verifies the pinned local model and capacity, or that the OpenAI key can see the configured model, then loads the local tokenizer and frozen retriever. There is no fallback between providers, and both pass the same request, response and model-identity checks.

Public questions go straight to retrieval without personal interpretation; a reranker gate of 0.23, calibrated on the audited English cases, decides coverage before any model call. Personal turns pass through restricted-English ingress and a finite graph. The model proposes candidate selections; only validated proposals can update current facts. Original values and exact proof spans remain private in session RAM. Safe context and complete optional exchanges are token-budgeted independently of the evidence character budget.

The PII client calls a separate loopback detector and accepts only its boolean decision. Detector failure blocks new processing. A disposable detector string can mask recognized financial spans using existing source proofs; this does not change the customer's original request or financial state. The detector environment has separately locked dependencies and local language assets.

The [current UI receipt](../experiments/failures-and-pivots.md) records six rejected synthetic PII cases with empty trace question/answer and zero model tokens, detector-outage rejection/retry recovery, and reset during pending inference. Trace questions use corpus-owned public text or canonical symbols with amount handles replaced by `AMOUNT`. Flow-answer traces keep the fixed server question label, not submitted text. Structured-answer and Forget paths were source-reviewed; the chat-first UI does not expose those controls.

Retrieval reads hash-checked local indexes and projections. It does not crawl sources or contact a production data service. Records without a citable source URL (58 of 1,476) are removed before selection. The model selects packed evidence IDs, then the server verifies and renders the saved extracts. English citation PDFs are hash-bound derivatives of the reviewed corpus and retain original Swedish provenance. A source capture can be dated or limited in scope.

Financial tools use current proven inputs. Drafts have their own snapshot, revision and lifetime. The browser owns visible form edits. For the local Refinance path, **Start application** checks the current draft and form snapshot before filling the supported step-1 fields. A changed form must not be overwritten by an old card. The flow remains editable and cannot submit. Only the Refinance draft is supported; amounts and intent are typed in chat, with no guided question buttons.

The console is served by the same backend. It exposes safe execution projections and stopping reasons, with bounded in-memory retention and session lifecycle purging. Its trace does not contain every retrieval candidate, reranker score or exact model argument, so it cannot prove full pipeline replay.

## Source map

| Runtime responsibility | Implementation |
| --- | --- |
| API routes, lifecycle, public route and console | [app.py](../../src/grounded_mortgage_assistant/app.py), [contracts.py](../../src/grounded_mortgage_assistant/contracts.py), [console/](../../src/grounded_mortgage_assistant/console/static/inspect.js) |
| Guardrail: private ingress and local PII detector | [guardrails/privacy.py](../../src/grounded_mortgage_assistant/guardrails/privacy.py), [guardrails/pii.py](../../src/grounded_mortgage_assistant/guardrails/pii.py), [detector service](../../services/pii/app.py) |
| Workflow: finite interpretation/action graph | [workflow/graph.py](../../src/grounded_mortgage_assistant/workflow/graph.py) |
| LLM: provider selection, model identity, candidate contract and safe context | [llm/conversation.py](../../src/grounded_mortgage_assistant/llm/conversation.py), [llm/interpretation_candidates.py](../../src/grounded_mortgage_assistant/llm/interpretation_candidates.py), [llm/model_context.py](../../src/grounded_mortgage_assistant/llm/model_context.py) |
| RAG: public question routing, frozen retrieval, relevance gate, extract rendering and PDF provenance | [rag/public_questions.py](../../src/grounded_mortgage_assistant/rag/public_questions.py), [rag/retrieval.py](../../src/grounded_mortgage_assistant/rag/retrieval.py), [rag/answer.py](../../src/grounded_mortgage_assistant/rag/answer.py), [rag/public_evidence.py](../../src/grounded_mortgage_assistant/rag/public_evidence.py) |
| State: current facts, provenance, replay, reset and expiry | [state/memory.py](../../src/grounded_mortgage_assistant/state/memory.py), [state/proofs.py](../../src/grounded_mortgage_assistant/state/proofs.py), [state/runtime.py](../../src/grounded_mortgage_assistant/state/runtime.py) |
| Tools: LTV calculation and Refinance draft building | [tools/calculations.py](../../src/grounded_mortgage_assistant/tools/calculations.py), [tools/draft.py](../../src/grounded_mortgage_assistant/tools/draft.py), [tools/schemas.py](../../src/grounded_mortgage_assistant/tools/schemas.py) |
| Console trace events | [tracing.py](../../src/grounded_mortgage_assistant/tracing.py) |
| Customer chat and direct handoff | Proprietary frontend (not in this repository): `apps/landing/src/components/LAIfChat/LAIfChat.tsx` and `MortgageDraftContext.tsx` |

Source links describe the inspected implementation; the current UI receipt records source hashes for the verified safety changes. The diagrams deliberately carry no committed-repository verification badge because this local repository has no origin remote. Automated diagram checks validate the artifact and browser layout, not runtime behavior. The [README](../../README.md) links the results and the demo video.
