# Adaptive Threshold Sweep Warm (20260314_035853)

Dataset: `/Users/frank/Develop/SearchEngine/data/ltr_training_data_feedback_combined.json`
Queries: 42

| Hard Threshold | nDCG@10 | MRR@10 | Recall@10 | Avg Latency (ms) | P95 Latency (ms) | Easy | Hard |
|---|---:|---:|---:|---:|---:|---:|---:|
| 0.35 | 0.7497 | 0.7020 | 0.9881 | 306.6 | 1075.4 | 23 | 19 |
| 0.45 | 0.7388 | 0.6940 | 0.9881 | 311.7 | 1166.8 | 25 | 17 |
| 0.55 | 0.7388 | 0.6940 | 0.9881 | 306.7 | 1195.8 | 25 | 17 |
| 0.65 | 0.7407 | 0.6861 | 1.0000 | 187.8 | 1227.9 | 36 | 6 |
| 0.75 | 0.7349 | 0.6790 | 1.0000 | 169.4 | 1224.8 | 37 | 5 |

Baseline reference:
`nDCG@10=0.7380, MRR@10=0.6861, Recall@10=1.0000, AvgLatency=9.4ms`
