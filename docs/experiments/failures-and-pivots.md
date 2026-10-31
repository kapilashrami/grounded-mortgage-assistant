# Failures and pivots

Data: [qwen-gate.json](../results/qwen-gate.json), [browser-suites.json](../results/browser-suites.json).

## Design pivots

- **Tool selection skipped retrieval.** When Qwen chose its own tools, it sometimes did not retrieve. The service now owns the retrieval path.
- **Freeform synthesis failed source support.** The model now picks evidence IDs, and the server renders reviewed extracts with validated citations. Answers are easier to check but less fluent.
- **Facts mixed with drafts.** An amount correction could start a draft, and a later LTV question could miss the calculation. A hosted gpt-5.4-mini diagnostic failed the same way, so the problem was the state contract, not the provider. Facts and draft snapshots now have separate lifecycles, and drafting needs an explicit request.
- **Qwen gate, 30 October: 0/3.** A frozen build completed the positive journey in every run but failed the negated-amount step. This led to narrower proposal validation. The gate has not been rerun on the current build.

## Defects found in browser testing

Each was reproduced in the real UI and fixed with a test:

- Start over failed behind a stale frontend proxy.
- An old draft could overwrite a newer form edit.
- No-increase phrasings lost their proof, so the draft was refused.
- A third-party mortgage raised an error instead of being marked out of scope.
- Qwen sometimes labelled the explicit LTV question an acknowledgement. The server now promotes it.
- The reversed LTV phrasing was rejected as personal information.
- K01 never reached retrieval. The [relevance gate](relevance-gate.md) replaced the lexical projector.
- 58 of 1,476 records lacked a citable source URL, so K05 failed closed in 2 of 3 early OpenAI runs. Only citable records are offered now.
- OpenAI labelled the refinance intent a calculation in 2 of 6 runs. A turn without amounts or calculation wording now cannot start one.
- Stating that intent marked the unchanged LTV card stale. It now goes stale only when an amount changes.
- The chat minimized on scripted scrolls, causing harness timeouts. It now responds only to pointer or wheel input.

## Still open

Any digit keeps a question off the public path, so U02 gets a rephrase request instead of the not-found message. A bare LTV question without amounts retrieves instead of asking for them.
