# Manual Judgment Evaluation Report

- Generated at: `2026-04-11T20:03:40.494646+00:00`
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
| `baseline` | 0.8589 | 0.9238 | 0.5944 | 15.70 | 22.53 | 0.00 |
| `ltr` | 0.8994 | 1.0000 | 0.5944 | 23.24 | 29.71 | 0.00 |
| `adaptive` | 0.8993 | 1.0000 | 0.5944 | 15.36 | 22.87 | 0.00 |
| `hybrid` | 0.7308 | 0.9294 | 0.4346 | 92.64 | 108.03 | 4.21 |
| `cross_encoder` | 0.8857 | 0.9786 | 0.5944 | 84.77 | 100.79 | 0.00 |

## Notes

- Labels should be manually reviewed; weak historical labels should not be treated as final ground truth.
- `Avg Unjudged Returned` should be close to 0 if the pooled judgment set covers all evaluated modes well.
- Returned documents outside the judgment pool are treated as 0 relevance, so high unjudged coverage still biases the result downward.
