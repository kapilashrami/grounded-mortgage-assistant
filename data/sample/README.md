# Synthetic sample records

`records.jsonl` has 20 generic English mortgage records written for this repository, so retrieval can run without the private corpus. They explain general concepts rather than any lender's terms, and their source URLs point to `example.com` on purpose.

Each line has `id`, `product`, `question`, `answer`, `qualifications` (conditions that limit the answer) and `source` (`title`, `url`). Nothing embeds them into the BGE-M3/FAISS bundle format yet, so for now they are plain data.
