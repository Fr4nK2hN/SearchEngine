# Adaptive Hard Top-K Cap Sweep Warm (20260314_034018)

Dataset: `/Users/frank/Develop/SearchEngine/data/ltr_training_data_feedback_combined.json`
Queries: 42
Warmup queries: 5

| Hard Top-K Cap | nDCG@10 | MRR@10 | Recall@10 | Avg Latency (ms) | P95 Latency (ms) | Hard Queries | Avg Hard Top-K |
|---|---:|---:|---:|---:|---:|---:|---:|
| none | 0.7388 | 0.6940 | 0.9881 | 272.6 | 1022.6 | 17 | 30.0 |
| 10 | 0.7322 | 0.6940 | 0.9762 | 263.9 | 987.9 | 17 | 10.0 |
| 15 | 0.7388 | 0.6940 | 0.9881 | 288.6 | 1024.0 | 17 | 15.0 |
| 20 | 0.7388 | 0.6940 | 0.9881 | 278.4 | 1130.8 | 17 | 20.0 |

Baseline reference:
`nDCG@10=0.7380, MRR@10=0.6861, Recall@10=1.0000, AvgLatency=8.9ms, P95=15.7ms`
