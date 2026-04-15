# Manual Judgment Evaluation Report

- Generated at: `2026-04-11T19:38:50.915338+00:00`
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
| `baseline` | 0.8589 | 0.9238 | 0.5944 | 16.34 | 22.45 | 0.00 |
| `ltr` | 0.8994 | 1.0000 | 0.5944 | 37.27 | 45.05 | 0.00 |
| `adaptive` | 0.8866 | 0.9786 | 0.5944 | 39.88 | 106.89 | 0.00 |
| `hybrid` | 0.7308 | 0.9294 | 0.4346 | 113.57 | 131.91 | 4.21 |
| `cross_encoder` | 0.8857 | 0.9786 | 0.5944 | 106.37 | 120.44 | 0.00 |

## Notes

- Labels should be manually reviewed; weak historical labels should not be treated as final ground truth.
- `Avg Unjudged Returned` should be close to 0 if the pooled judgment set covers all evaluated modes well.
- Returned documents outside the judgment pool are treated as 0 relevance, so high unjudged coverage still biases the result downward.
