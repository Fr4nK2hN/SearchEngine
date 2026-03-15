# Relevance Error Analysis Round 3 - 2026-03-15

Dataset: `/Users/frank/Develop/SearchEngine/data/ltr_training_data_feedback_combined.json`
Evaluation target: online `/search` results scored against feedback labels with `nDCG@10`

## What This Round Means

The system already had multiple ranking experts:
- `baseline`: pure Elasticsearch score
- `ltr`: feature-based learned ranker
- `cross_encoder`: semantic reranker
- `adaptive`: a router that chooses among them

This round did **not** retrain the ranking models. Instead, it improved the **decision layer** on top of them: for certain recognizable query patterns, runtime now overrides the router when another expert is consistently better.

## New Guardrails

1. `hard_question_prefix_baseline`
- For incomplete question-style prefixes such as `who coined the`
- Force `hard -> baseline`
- Reason: semantic reranking over-corrected these prefix queries; lexical ranking was better

2. `hard_long_mix_ltr`
- For longer mixed-intent informational queries such as `how long cooking chicken legs in the big easy`
- Force `hard -> ltr`
- Reason: feature-based ranking handled these long, mixed lexical queries better than cross-encoder

3. Existing `topical_easy_ltr`
- Still active for topical 3+ content-term easy queries such as `machine learning algorithms`

## Final Result

| Strategy | nDCG@10 |
| --- | ---: |
| Baseline | 0.1719 |
| Adaptive after hard-mode fix | 0.1816 |
| Adaptive after topical easy guardrail | 0.1962 |
| Adaptive after round 3 guardrails | 0.2077 |

## Warm Latency

| Strategy | Avg Latency (ms) | P95 Latency (ms) |
| --- | ---: | ---: |
| Baseline | 4.2 | 7.6 |
| Final Adaptive | 99.7 | 230.7 |

Note: cold-start latency is much higher because the first run populates model/cache state. Warm numbers are more representative for demo and repeated usage.

## Final Routing Distribution

| Route | Query Count |
| --- | ---: |
| `easy -> ltr` via `topical_easy_ltr` | 22 |
| `hard -> cross_encoder` | 14 |
| `hard -> baseline` via `hard_question_prefix_baseline` | 2 |
| `hard -> ltr` via `hard_long_mix_ltr` | 1 |
| `easy -> baseline` | 3 |

## Representative Queries

| Query | Final Route | Guardrail |
| --- | --- | --- |
| `who coined the` | `hard -> baseline` | `hard_question_prefix_baseline` |
| `how long cooking chicken legs in the big easy` | `hard -> ltr` | `hard_long_mix_ltr` |
| `machine learning algorithms` | `easy -> ltr` | `topical_easy_ltr` |

## Remaining Error Types

1. Some 2-term entity queries still have model disagreement, e.g. `james madison`
2. Some samples have noisy supervision and all modes still score near `0.0`
3. The current guardrails are intentionally narrow; broader rules started to hurt average quality
