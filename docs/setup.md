# Setup

## Tests

You need Python 3.14. The tests run without the corpus and without a model: the API tests use the synthetic sample records, and the tests that check the private corpus bundle skip with a reason.

```sh
python3.14 -m venv .venv
.venv/bin/python -m pip install -r requirements.lock
.venv/bin/python -m pytest -q
.venv/bin/python -m ruff check .
```

## Running the backend

The backend needs three things: a frozen corpus bundle at `DATA_DIR`, an LLM, and the local PII detector, which has its own environment ([services/pii/README.md](../services/pii/README.md)). Embeddings (`BAAI/bge-m3`) and the reranker (`BAAI/bge-reranker-v2-m3`) load from the Hugging Face cache at pinned revisions, so download them once before running offline.

| Setting | What it does |
| --- | --- |
| `DATA_DIR` | Corpus folder. Defaults to `data/corpus`, which Git ignores. |
| `LOCAL_TOKENIZER_PATH` | Local Qwen tokenizer used to budget the model context. Required for real model calls. |
| `LLM_PROVIDER` | `local` (the default, an OpenAI-compatible endpoint at `127.0.0.1:8011`) or `openai`. |
| `OPENAI_API_KEY`, `OPENAI_MODEL` | Used with `openai`. The model defaults to `gpt-4.1`. |
| `OPENAI_REASONING_EFFORT` | GPT-5 models only. Chat Completions accepts tools only with `none`. |
| `CORPUS_NAME` | Name of the organisation shown in answers. Empty by default. |

```sh
DATA_DIR=/path/to/corpus LOCAL_TOKENIZER_PATH=/path/to/Qwen3.5-9B-MLX-4bit \
  PYTHONPATH=src HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  .venv/bin/python -m uvicorn grounded_mortgage_assistant.app:create_app --factory --host 127.0.0.1 --port 8099
```

Send turns to `POST /api/chat`, and open the engineering console at `http://127.0.0.1:8099/inspect`. The provider is fixed at startup and there is no fallback between providers. With `openai`, only the masked turn and public evidence text leave the machine, after the PII check.

## Retrieval without the private corpus

A fresh clone can't answer product questions yet, because the reference bundle belongs to the company. A loader that turns [data/sample/records.jsonl](../data/sample/records.jsonl) into a bundle is planned. Until it exists, you can build one by hand:

1. Download `BAAI/bge-m3` and `BAAI/bge-reranker-v2-m3`.
2. Embed each record's question and answer with BGE-M3 and build a FAISS HNSW index.
3. Write the records, vectors, index and a SHA-256 manifest in the layout that `validate_frozen_bundle` in [rag/retrieval.py](../src/grounded_mortgage_assistant/rag/retrieval.py) reads.
4. Replace the pinned hashes and record count in that function with the values from your manifest, and point `DATA_DIR` at the bundle.

The LTV, draft and privacy logic never uses retrieval, but the app still loads the bundle when it starts. Letting it start without one is part of the loader work.
