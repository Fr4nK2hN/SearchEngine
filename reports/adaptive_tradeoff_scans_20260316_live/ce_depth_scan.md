# CE Depth Scan (2026-03-15)

- Dataset: `data/ltr_training_data_feedback_combined.json`
- Queries: `42`

| Config | CE Top-N | nDCG@10 | MRR@10 | Recall@10 | Avg Latency (ms) | P95 Latency (ms) | Hard Rate | CE Queries |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| current_policy | policy | 0.1784 | 0.1923 | 0.2143 | 43.3 | 174.1 | 21.4% | 6 |
| ce_top_5 | 5 | 0.1784 | 0.1923 | 0.2143 | 27.6 | 65.9 | 21.4% | 6 |
| ce_top_10 | 10 | 0.1784 | 0.1923 | 0.2143 | 32.1 | 97.1 | 21.4% | 6 |
| ce_top_20 | 20 | 0.1784 | 0.1923 | 0.2143 | 45.2 | 177.2 | 21.4% | 6 |
| ce_top_30 | 30 | 0.1784 | 0.1923 | 0.2143 | 54.7 | 255.7 | 21.4% | 6 |
