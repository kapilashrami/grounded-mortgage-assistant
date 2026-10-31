# OpenAI model comparison

**When:** 31 October 2026. **Data:** [model-comparison.json](../results/model-comparison.json)

## Why it was run

With OpenAI, the model only interprets turns and picks evidence IDs through tool calls; the server retrieves, calculates and renders. A cheaper model might suffice for that narrow job.

## Setup

Same backend, frontend and suites for each model: the 16-check journey and the 20-question suite, 3 runs each, through the real UI. Latency, cost and evidence counts come from saved console traces.

## Results

| Model | Journey | Questions | Model misses | UI timeouts | Median / p90 turn | Cost per turn | Evidence per answer |
| --- | --- | --- | --- | --- | --- | --- | --- |
| gpt-4.1 | 48/48 | 55/60 | L01 | P01 | 1.19 / 1.63 s | $0.0032 | 2.10 |
| gpt-4.1-mini | 48/48 | 54/60 | L04 | K04, K05 | 1.34 / 1.64 s | $0.0006 | 2.16 |
| gpt-5.4-mini (effort none) | 48/48 | 57/60 | none | none | 1.20 / 1.65 s | $0.0014 | 1.83 |

- **U02** failed in every run for every model. It is a routing rule, not model behaviour.
- **Model misses:** the proposal for an explicit LTV question failed server validation, so the user got a clarification request instead of a card. Nothing incorrect was shown.
- **UI timeouts:** the chat closed itself on scripted scroll events before the harness typed; no model call was involved. After the frontend fix, gpt-4.1 scored 19/20 twice and 16/16.
- **Evidence per answer:** mean records selected per answered product question. This is selection width, not correctness.
- **gpt-5.4-mini with effort low** could not run. Chat Completions rejects function tools with any effort other than none for this model.

The whole comparison cost about $0.31 at list prices.

## What it does not show

With 9 runs per model, a difference of one miss versus none is not statistically meaningful. The suites are exposed functional examples, not unseen acceptance tests.

## Decision

The demo stays on gpt-4.1. gpt-5.4-mini looked slightly better and cheaper, but the differences are within noise.
