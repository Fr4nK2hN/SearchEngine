# Manual Judgment Evaluation Report

- Generated at: `2026-04-11T16:59:13.260258+00:00`
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
| `baseline` | 0.8589 | 0.9238 | 0.5944 | 22.32 | 32.75 | 0.00 |
| `ltr` | 0.7988 | 0.9363 | 0.5779 | 37.28 | 57.28 | 0.00 |
| `adaptive` | 0.8866 | 0.9786 | 0.5944 | 42.10 | 111.52 | 0.00 |
| `hybrid` | 0.8159 | 0.9439 | 0.5779 | 109.08 | 129.76 | 0.00 |
| `cross_encoder` | 0.9146 | 0.9792 | 0.6050 | 444.48 | 549.92 | 0.00 |

## Notes

- Labels should be manually reviewed; weak historical labels should not be treated as final ground truth.
- `Avg Unjudged Returned` should be close to 0 if the pooled judgment set covers all evaluated modes well.
- Returned documents outside the judgment pool are treated as 0 relevance, so high unjudged coverage still biases the result downward.
