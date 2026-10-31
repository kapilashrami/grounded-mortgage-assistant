"""Runtime settings, read from the environment only. Nothing defaults to a home directory."""

from __future__ import annotations

import os
from collections.abc import Mapping
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

# Hugging Face models, loaded by repository ID at a pinned revision.
EMBEDDING_MODEL = ("BAAI/bge-m3", "5617a9f61b028005a4858fdac845db406aefb181")
RERANKER_MODEL = ("BAAI/bge-reranker-v2-m3", "953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e")


def data_dir(env: Mapping[str, str] = os.environ) -> Path:
    """Root of the frozen corpus; manifest paths inside it are relative to this folder."""
    return Path(env.get("DATA_DIR", ROOT / "data" / "corpus")).expanduser().resolve()


def tokenizer_path(env: Mapping[str, str] = os.environ) -> Path:
    """Local Qwen tokenizer used for exact context budgeting. Required for real model calls."""
    value = env.get("LOCAL_TOKENIZER_PATH", "").strip()
    if not value:
        raise RuntimeError("LOCAL_TOKENIZER_PATH is required to call a model")
    return Path(value).expanduser()


def corpus_name(env: Mapping[str, str] = os.environ) -> str:
    """Name of the organisation whose content the corpus holds, used in user-facing text."""
    return env.get("CORPUS_NAME", "").strip()


def not_found_answer(name: str) -> str:
    if name:
        return (
            f"I could not find that in the available {name} information. "
            f"Please ask about {name}'s loans, insurance, company or services."
        )
    return (
        "I could not find that in the available information. "
        "Please ask about loans, insurance, the company or its services."
    )


def unavailable_answer(name: str) -> str:
    return f"Please ask a question about the available {name + ' ' if name else ''}information."
