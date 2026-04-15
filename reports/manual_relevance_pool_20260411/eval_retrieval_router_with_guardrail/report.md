# Manual Judgment Evaluation Report

- Generated at: `2026-04-11T16:55:26.286294+00:00`
- Judgments: `reports/manual_relevance_pool_20260411/judgment_pool.csv`
- Label column: `manual_label`
- Base URL: `http://127.0.0.1:5000`
- Top-K: `10`
- Labeled rows: `741`
- Unlabeled rows ignored in ground truth: `0`
- Queries with labels: `42`
- Skipped incomplete queries: `0`

| Mode | NDCG@10 | MRR@10 | Recall@10 | Avg Response (ms) | P95 (ms) | Avg Unjudged Returned |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `baseline` | 0.8589 | 0.9238 | 0.5944 | 20.04 | 28.65 | 0.00 |
| `ltr` | 0.7988 | 0.9363 | 0.5779 | 105.57 | 139.47 | 0.00 |
| `adaptive` | 0.8866 | 0.9786 | 0.5944 | 106.35 | 304.23 | 0.00 |
| `hybrid` | 0.8159 | 0.9439 | 0.5779 | 315.62 | 367.99 | 0.00 |
| `cross_encoder` | 0.9146 | 0.9792 | 0.6050 | 1438.26 | 1919.91 | 0.00 |

## Notes

- Labels should be manually reviewed; weak historical labels should not be treated as final ground truth.
- `Avg Unjudged Returned` should be close to 0 if the pooled judgment set covers all evaluated modes well.
- Returned documents outside the judgment pool are treated as 0 relevance, so high unjudged coverage still biases the result downward.
