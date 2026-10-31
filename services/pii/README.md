# Local PII detector

The backend sends every customer message here before a model sees it. The service accepts loopback traffic only and never writes request text to logs, traces or disk.

Email addresses are caught with a plain pattern check before the detector runs, so they don't depend on entity recognition. A `/check` response has only two fields: `pii_detected` and `category`, which is `email_address`, `personal_information` or `null`. No findings or request text come back.

## Running it

It needs its own environment, `.venv-pii`, with the packages from `requirements.lock`, the spaCy model `en_core_web_lg==3.8.0` and the NLTK `punkt` and `punkt_tab` data:

```bash
.venv-pii/bin/python -m nltk.downloader -d .venv-pii/nltk_data punkt punkt_tab
```

The service checks these files before it imports Guardrails, so it never downloads anything at runtime. Start it from the repository root with an absolute `NLTK_DATA` path:

```bash
env -i PATH="$PATH" PYTHONNOUSERSITE=1 NLTK_DATA="$(pwd)/.venv-pii/nltk_data" OTEL_SDK_DISABLED=true LITELLM_LOCAL_MODEL_COST_MAP=true .venv-pii/bin/python -m uvicorn services.pii.app:app --host 127.0.0.1 --port 8097 --workers 1 --no-access-log
```

It refuses to start if a user Guardrails configuration exists. The main backend refuses to answer when this service is down.
