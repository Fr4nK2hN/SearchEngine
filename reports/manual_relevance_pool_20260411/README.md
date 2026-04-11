# Manual Relevance Judgment Pool

- Generated at: `2026-04-11T15:53:13.728139+00:00`
- Query source: `data/ltr_training_data_feedback_combined.json`
- Search API: `http://127.0.0.1:5000`
- Modes: `baseline,ltr,adaptive,hybrid,cross_encoder`
- Top-N per mode: `10`
- Query count: `42`
- Candidate rows: `741`
- Average pooled candidates per query: `17.64`
- Rows with previous weak labels: `34`

## How to Label

Fill `manual_label` for every row. Use a small graded scale:

| Label | Meaning |
| ---: | --- |
| 0 | Irrelevant or misleading |
| 1 | Marginally related, only weak keyword overlap |
| 2 | Partially relevant, useful but incomplete |
| 3 | Highly relevant and directly answers the query |

Do not copy `previous_label` blindly. It is only a weak reference from the earlier dataset and should be corrected when it is noisy.

After labeling, run:

```bash
docker exec app python tools/analysis/evaluate_manual_judgments.py \
  --judgments reports/manual_relevance_pool_20260411/judgment_pool.csv \
  --out-dir reports/manual_relevance_pool_20260411/eval_after_labeling
```

The evaluator skips incomplete queries by default. If you label a smaller subset, label all rows for those selected queries instead of labeling scattered rows.

## Why This Solves the Low NDCG Problem

- The pool merges Top-N results from multiple rankers, so the judged set covers more documents than the old 10-doc candidate list.
- Manual labels increase reliable positive examples and reduce weak-label noise.
- End-to-end NDCG becomes more meaningful because returned Top-10 documents are likely to exist in the judged pool.
