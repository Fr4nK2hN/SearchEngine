# Manual Judgment Evaluation Report

- Generated at: `2026-04-11T19:34:02.130048+00:00`
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
| `baseline` | 0.8589 | 0.9238 | 0.5944 | 18.94 | 31.61 | 0.00 |
| `ltr` | 0.7141 | 0.9095 | 0.4346 | 30.15 | 46.42 | 4.21 |
| `adaptive` | 0.8866 | 0.9786 | 0.5944 | 35.58 | 88.90 | 0.00 |
| `hybrid` | 0.7308 | 0.9294 | 0.4346 | 103.19 | 126.02 | 4.21 |
| `cross_encoder` | 0.8857 | 0.9786 | 0.5944 | 89.41 | 109.24 | 0.00 |

## Notes

- Labels should be manually reviewed; weak historical labels should not be treated as final ground truth.
- `Avg Unjudged Returned` should be close to 0 if the pooled judgment set covers all evaluated modes well.
- Returned documents outside the judgment pool are treated as 0 relevance, so high unjudged coverage still biases the result downward.
