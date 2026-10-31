# Local PII gate

This service only accepts loopback traffic and never writes request text to logs, traces,
or disk. Launch it from the repository root in a clean environment after installing the
locked dependencies, `en_core_web_lg==3.8.0`, and the `punkt` and `punkt_tab` tokenizer
datasets into `.venv-pii`:

It blocks conventional email-address syntax directly before the local detector runs, so
email coverage does not depend on language-model entity recognition.

Successful `/check` responses contain only `pii_detected` and `category`. The category
is `email_address` or `personal_information` when PII is detected, and `null` otherwise.
No detector findings or request text are returned.

```bash
.venv-pii/bin/python -m nltk.downloader -d .venv-pii/nltk_data punkt punkt_tab
```

The service checks those local assets before importing Guardrails, so it does not attempt
a runtime download. Launch it with an explicit absolute `NLTK_DATA` directory:

```bash
env -i PATH="$PATH" PYTHONNOUSERSITE=1 NLTK_DATA="$(pwd)/.venv-pii/nltk_data" OTEL_SDK_DISABLED=true LITELLM_LOCAL_MODEL_COST_MAP=true .venv-pii/bin/python -m uvicorn services.pii.app:app --host 127.0.0.1 --port 8097 --workers 1 --no-access-log
```

Startup fails closed when a user Guardrails configuration exists, before the SDK is imported.
The main candidate backend also fails closed when this local service is unavailable.
