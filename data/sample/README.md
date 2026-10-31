# Synthetic sample records

`records.jsonl` holds 20 generic English mortgage records written for this repository, so the retrieval path can be run without the private Lendo corpus. They are country-neutral explanations, not any lender's terms, and their source URLs point to `example.com` on purpose.

Each line has `id`, `product`, `question`, `answer`, `qualifications` (conditions that limit the answer) and `source` (`title`, `url`). A bundle builder that embeds these records into the frozen BGE-M3/FAISS format is planned for the public repository; until then the files are data only.
