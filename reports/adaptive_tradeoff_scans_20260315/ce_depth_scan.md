# CE Depth Scan (2026-03-15)

- Dataset: `data/ltr_training_data_feedback_combined.json`
- Queries: `42`

| Config | CE Top-N | nDCG@10 | MRR@10 | Recall@10 | Avg Latency (ms) | P95 Latency (ms) | Hard Rate | CE Queries |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| current_policy | policy | 0.2077 | 0.2171 | 0.2619 | 113.1 | 266.7 | 40.5% | 14 |
| ce_top_5 | 5 | 0.2077 | 0.2171 | 0.2619 | 49.1 | 66.7 | 40.5% | 14 |
| ce_top_10 | 10 | 0.2077 | 0.2171 | 0.2619 | 58.9 | 92.6 | 40.5% | 14 |
| ce_top_20 | 20 | 0.2119 | 0.2195 | 0.2738 | 86.4 | 172.4 | 40.5% | 14 |
| ce_top_30 | 30 | 0.2077 | 0.2171 | 0.2619 | 117.9 | 279.4 | 40.5% | 14 |
