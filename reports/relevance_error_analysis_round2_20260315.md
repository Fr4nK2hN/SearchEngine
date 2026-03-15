# Relevance Error Analysis Round 2 - 2026-03-15

Dataset: `/Users/frank/Develop/SearchEngine/data/ltr_training_data_feedback_combined.json`
Evaluation target: online `/search` results scored against feedback labels with `nDCG@10`

## Implemented Guardrail

For non-prefix queries with at least 3 content terms, low stopword ratio, and an original `easy -> baseline` route, runtime now upgrades the selected mode to `ltr`:

- route label remains `easy`
- selected mode becomes `ltr`
- route guardrail tag: `topical_easy_ltr`

## Result Summary

| Strategy | nDCG@10 | Avg Latency (ms) | P95 Latency (ms) |
| --- | ---: | ---: | ---: |
| Baseline | 0.1719 | 4.2 | 7.6 |
| Adaptive after hard-mode fix (`hard -> cross_encoder`) | 0.1816 | - | - |
| Adaptive after topical guardrail | 0.1962 | 113.0 | 228.7 |

## Improvement

- Relative to baseline: `+0.0243` nDCG@10
- Relative to previous adaptive step: `+0.0146` nDCG@10

## Representative Queries Fixed By Guardrail

| Query | Previous Adaptive Route | Current Adaptive Route | Note |
| --- | --- | --- | --- |
| `machine learning algorithms` | `easy -> baseline` | `easy -> ltr` | `ltr` beats baseline on this topical 3-word query |
| `home gardening vegetables` | `easy -> baseline` | `easy -> ltr` | same pattern |

## Current Routing Distribution

| Route | Query Count |
| --- | ---: |
| `easy -> ltr` with `topical_easy_ltr` | 22 |
| `hard -> cross_encoder` | 17 |
| `easy -> baseline` | 3 |

## Remaining Gaps

The largest remaining misses are now concentrated in three categories:

1. Incomplete / prefix-like queries where baseline is still better, e.g. `who coined the`
2. Named-entity queries where `ltr` or `hybrid` can outperform current hard `cross_encoder`, e.g. `james madison`
3. Queries with noisy supervision where all modes still score `0.0`

## Code References

- `/Users/frank/Develop/SearchEngine/app.py`
- `/Users/frank/Develop/SearchEngine/static/script.js`
