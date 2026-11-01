# Data

The runtime corpus is not in this repository. The assistant was built on 1,476 reviewed English records taken from the company's public website, plus search indexes and citation PDFs. The company owns that content, so it stays private.

- `sample/records.jsonl` has 20 generic mortgage records written for this repository, with `example.com` sources ([sample/README.md](sample/README.md)). The tests use them through a fixture.
- `DATA_DIR` points the backend at a corpus folder. The default is `data/corpus`, which Git ignores. At startup the backend checks the bundle's SHA-256 manifests and stops if a file is missing or has changed.
- `scripts/build_sample_bundle.py` turns the sample into a bundle the backend can search, and the app starts without any bundle. [docs/setup.md](../docs/setup.md) has the details.
