# Data

The runtime corpus is not part of this repository. The assistant was built on 1,476 reviewed English records prepared from Lendo's public website, with search indexes and citation PDFs. That content is Lendo-owned and cannot be published.

- `sample/records.jsonl` holds 20 generic English mortgage records written for this repository, with `example.com` sources (see [sample/README.md](sample/README.md)). The test suite uses them through a fixture.
- `DATA_DIR` points the backend at a corpus folder (default `data/corpus`, ignored by Git). Startup checks the bundle's SHA-256 manifests and stops if anything is missing or changed.
- A loader that turns the sample into a bundle the backend accepts is planned; the README lists the manual steps until then.
