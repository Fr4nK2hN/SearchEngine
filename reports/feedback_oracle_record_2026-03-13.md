# Feedback Oracle Experiment Record

- Generated at (UTC): `2026-03-13T13:58:24.149073+00:00`
- Primary result: `models/oracle_feedback_combined_pretrained/oracle_results.json`
- Secondary result: `models/oracle_feedback_combined_pretrained_noprefix/oracle_results.json`
- Dataset report: `data/ltr_training_data_feedback_combined.json.report.json`

## Feedback Dataset

- Samples: `42`
- Search count: `45`
- Searches with confirmed click: `44`
- Label distribution: `{'0': 348, '2': 14, '3': 36, '4': 22}`
- Dropped: `{'click_not_in_candidates': 2, 'not_enough_clicks': 1}`

## Primary (Combined Feedback)

- Data used: `42`
- Best single expert: `Hybrid`
- Oracle nDCG gain: `0.1187` (15.83%)
- Selection count: `{'Baseline': 19, 'LTR': 11, 'Hybrid': 4, 'Cross-Encoder': 8}`
- Query profile: `prefix_like_ratio=30.95%`, `first_doc_is_max_ratio=47.62%`

| Expert | nDCG@10 | MRR@10 | Avg Latency (ms) |
| --- | --- | --- | --- |
| Baseline | 0.7342 | 0.6863 | 0.00 |
| LTR | 0.7387 | 0.6992 | 258.46 |
| Cross-Encoder | 0.6984 | 0.6332 | 71.68 |
| Hybrid | 0.7499 | 0.7040 | 91.45 |

Warnings:
- Prefix-like query ratio is 31.0%; evaluation may not reflect real search intent.

## Secondary (No Prefix Filter)

- Data used: `29`
- Best single expert: `Baseline`
- Oracle nDCG gain: `0.1423` (19.44%)
- Selection count: `{'Baseline': 14, 'LTR': 8, 'Cross-Encoder': 6, 'Hybrid': 1}`
- Query profile: `prefix_like_ratio=0.00%`, `first_doc_is_max_ratio=44.83%`

| Expert | nDCG@10 | MRR@10 | Avg Latency (ms) |
| --- | --- | --- | --- |
| Baseline | 0.7317 | 0.6908 | 0.00 |
| LTR | 0.7309 | 0.6908 | 248.74 |
| Cross-Encoder | 0.6566 | 0.5636 | 71.18 |
| Hybrid | 0.7224 | 0.6661 | 91.27 |

Warnings:
- Query count is only 29; oracle variance is likely high.

## Conclusion

Feedback-driven oracle shows clear upper bound over best single expert, supporting adaptive routing as a viable direction.
