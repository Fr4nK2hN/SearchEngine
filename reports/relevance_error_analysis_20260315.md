# Relevance Error Analysis - 2026-03-15

Dataset: `/Users/frank/Develop/SearchEngine/data/ltr_training_data_feedback_combined.json`
Evaluation target: online `/search` results scored against feedback labels with `nDCG@10`

## Main Finding

The largest adaptive relevance loss was caused by the `hard -> hybrid` path. On this live evaluation slice, switching the hard path to `cross_encoder` improved adaptive quality more than any lightweight lexical rerank or router guardrail tested.

## Before/After Summary

| Strategy | nDCG@10 |
| --- | ---: |
| Baseline | 0.1719 |
| Cross-Encoder | 0.1764 |
| Adaptive (previous: easy->baseline, hard->hybrid) | 0.1618 |
| Adaptive (current: easy->baseline, hard->cross_encoder) | 0.1816 |

## Routing Distribution

| Route | Query Count |
| --- | ---: |
| `easy -> baseline` | 25 |
| `hard -> cross_encoder` | 17 |

## Highest Remaining Gaps

| Query | Adaptive | Best Single Mode | Best nDCG@10 | Gap |
| --- | ---: | --- | ---: | ---: |
| `who coined the` | 0.2575 | `baseline` | 0.6677 | 0.4102 |
| `james madison` | 0.3863 | `ltr` | 0.7725 | 0.3862 |
| `home gardening vegetables` | 0.0000 | `ltr` | 0.3010 | 0.3010 |
| `machine learning algorithms` | 0.0000 | `ltr` | 0.2801 | 0.2801 |
| `investment strategies` | 0.0000 | `cross_encoder` | 0.2641 | 0.2641 |
| `day subutex` | 0.0000 | `hybrid` | 0.1934 | 0.1934 |

## Error Pattern Notes

- Prefix-like or underspecified queries still sometimes get routed to hard when baseline is better, e.g. `who coined the`.
- Some topical 3-word queries are still routed to easy baseline even though `ltr` or `hybrid` performs better, e.g. `machine learning algorithms`.
- A subset of zero-score queries have noisy or weak supervision in the feedback dataset itself; for those queries all single modes remain at `0.0`.

## Implemented Optimization

Runtime default changed from:
- `ADAPTIVE_HARD_MODE=hybrid`

to:
- `ADAPTIVE_HARD_MODE=cross_encoder`

Code references:
- `/Users/frank/Develop/SearchEngine/app.py`
- `/Users/frank/Develop/SearchEngine/docker-compose.yml`
- `/Users/frank/Develop/SearchEngine/ranking/query_router.py`

## Next Candidate Improvements

1. Add a prefix-query guardrail so trailing-stopword / incomplete queries fall back to `baseline`.
2. Add a topical multi-term guardrail so certain 3+ content-term queries can bypass `baseline` and go to `ltr`.
3. Improve or clean noisy feedback labels for queries where all modes score `0.0`.
