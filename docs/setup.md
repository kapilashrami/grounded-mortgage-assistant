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

The backend needs an LLM and the local PII detector, and product questions also need a retrieval bundle at `DATA_DIR`. The detector has its own environment ([services/pii/README.md](../services/pii/README.md)). Embeddings (`BAAI/bge-m3`) and the reranker (`BAAI/bge-reranker-v2-m3`) load from the Hugging Face cache at pinned revisions, so download them once before running offline.

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

The reference bundle belongs to the company, so a fresh clone has none. The app still starts without it: product questions get a fixed "ask about the available information" reply, and the LTV, draft and privacy paths work as usual.

To try retrieval on synthetic data, build a bundle from the 20 records in [data/sample/records.jsonl](../data/sample/records.jsonl):

```sh
PYTHONPATH=src .venv/bin/python scripts/build_sample_bundle.py
```

The script embeds each record twice with the pinned BGE-M3, a plain question-and-answer view and a labelled view, and builds a FAISS HNSW index for each. It writes them to `DATA_DIR/sample-bundle`, by default `data/corpus/sample-bundle`, which Git ignores. The models download from Hugging Face on the first run.

The bundle's manifest hashes its own files and names the models, so a changed file or a different model stops startup. Sample answers cite the records' `example.com` links and have no PDF. If the company's pinned bundle is also present, it always wins.
