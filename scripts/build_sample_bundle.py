"""Build the retrieval bundle for the synthetic sample records.

Usage: python scripts/build_sample_bundle.py [--records PATH] [--out PATH]

Writes DATA_DIR/sample-bundle by default. Needs the pinned BGE-M3 model, which
is downloaded from Hugging Face on first use. The backend picks the bundle up
at startup when no frozen corpus is present.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from grounded_mortgage_assistant.config import EMBEDDING_MODEL, ROOT
from grounded_mortgage_assistant.rag.retrieval import SAMPLE_BUNDLE
from grounded_mortgage_assistant.rag.sample_bundle import build_sample_bundle


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--records", type=Path, default=ROOT / "data/sample/records.jsonl"
    )
    parser.add_argument("--out", type=Path, default=SAMPLE_BUNDLE)
    args = parser.parse_args()

    import faiss
    import torch
    from sentence_transformers import SentenceTransformer

    # torch and FAISS each bring OpenMP; on macOS both multithreaded can crash.
    torch.set_num_threads(1)
    faiss.omp_set_num_threads(1)
    model = SentenceTransformer(
        EMBEDDING_MODEL[0], revision=EMBEDDING_MODEL[1], device="cpu"
    )
    model.max_seq_length = 8192

    def encode(texts: list[str]):
        return model.encode(texts, normalize_embeddings=True, convert_to_numpy=True)

    manifest = build_sample_bundle(args.records, args.out, encode)
    print(json.dumps({"out": str(args.out), "records": len(manifest["record_ids"])}))


if __name__ == "__main__":
    main()
