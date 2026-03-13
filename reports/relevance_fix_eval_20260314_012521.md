# Relevance Fix Evaluation (20260314_012521)

Dataset: `/app/data/ltr_training_data_feedback_combined.json`
Queries: 42

| Strategy | nDCG@10 | MRR@10 | Recall@10 | Avg Latency (ms) | P95 Latency (ms) |
|---|---:|---:|---:|---:|---:|
| baseline | 0.7380 | 0.6861 | 1.0000 | 8.7 | 14.8 |
| ltr | 0.6499 | 0.6109 | 0.8571 | 359.4 | 786.6 |
| hybrid | 0.6731 | 0.6392 | 0.8571 | 116.6 | 142.4 |
| adaptive (current easy->baseline) | 0.7388 | 0.6940 | 0.9881 | 95.2 | 270.6 |
| adaptive old-sim (easy->ltr) | 0.6532 | 0.6094 | 0.8571 | 73.9 | 133.1 |

## Key deltas
- adaptive(current) vs adaptive(old-sim):
  - nDCG@10: +0.0856
  - MRR@10: +0.0846
  - Recall@10: +0.1310
  - Avg latency: +21.3 ms
