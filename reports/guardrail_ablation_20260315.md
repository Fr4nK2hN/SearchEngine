# Guardrail Ablation (2026-03-15)

- Dataset: `data/ltr_training_data_feedback_combined.json`
- Queries: `42`

| Config | nDCG@10 | MRR@10 | Recall@10 | Avg Latency (ms) | P95 Latency (ms) | Hard Rate | Guardrail Hits |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| router_only | 0.1816 | 0.2000 | 0.2143 | 104.2 | 267.7 | 40.5% | 0.0% |
| all_guardrails | 0.2077 | 0.2171 | 0.2619 | 112.0 | 258.5 | 40.5% | 59.5% |
| minus_topical_easy_ltr | 0.1930 | 0.2103 | 0.2262 | 87.5 | 257.6 | 40.5% | 7.1% |
| minus_hard_question_prefix_baseline | 0.1979 | 0.2086 | 0.2500 | 123.7 | 260.9 | 40.5% | 54.8% |
| minus_hard_long_mix_ltr | 0.2060 | 0.2153 | 0.2619 | 118.5 | 269.7 | 40.5% | 57.1% |

## Key Deltas

- `all_guardrails` vs `router_only`: nDCG@10 `+0.0261`
- `all_guardrails` vs `minus_topical_easy_ltr`: nDCG@10 `+0.0146`
- `all_guardrails` vs `minus_hard_question_prefix_baseline`: nDCG@10 `+0.0098`
- `all_guardrails` vs `minus_hard_long_mix_ltr`: nDCG@10 `+0.0017`

## Guardrail Trigger Counts (all_guardrails)

- `topical_easy_ltr`: 22
- `hard_question_prefix_baseline`: 2
- `hard_long_mix_ltr`: 1
