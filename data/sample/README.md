# Synthetic sample records

`records.jsonl` has 20 generic English mortgage records written for this repository, so retrieval can run without the private corpus. They explain general concepts rather than any lender's terms, and their source URLs point to `example.com` on purpose.

Each line has `id`, `product`, `question`, `answer`, `qualifications` (conditions that limit the answer) and `source` (`title`, `url`). `scripts/build_sample_bundle.py` turns them into a retrieval bundle the backend can search; see [docs/setup.md](../../docs/setup.md).
