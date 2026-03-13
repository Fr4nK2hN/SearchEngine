# Adaptive Hard Top-K Cap Sweep (20260314_033538)

Dataset: `/Users/frank/Develop/SearchEngine/data/ltr_training_data_feedback_combined.json`
Queries: 42

| Hard Top-K Cap | nDCG@10 | MRR@10 | Recall@10 | Avg Latency (ms) | P95 Latency (ms) | Hard Queries | Avg Hard Top-K |
|---|---:|---:|---:|---:|---:|---:|---:|
| none | 0.7388 | 0.6940 | 0.9881 | 331.1 | 1218.0 | 17 | 30.0 |
| 10 | 0.7322 | 0.6940 | 0.9762 | 284.3 | 988.8 | 17 | 10.0 |
| 15 | 0.7388 | 0.6940 | 0.9881 | 293.3 | 1026.8 | 17 | 15.0 |
| 20 | 0.7388 | 0.6940 | 0.9881 | 323.0 | 1102.0 | 17 | 20.0 |

Baseline reference:
`nDCG@10=0.7380, MRR@10=0.6861, Recall@10=1.0000, AvgLatency=8.8ms`
