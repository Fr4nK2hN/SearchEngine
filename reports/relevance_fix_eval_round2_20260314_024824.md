# Relevance Fix Evaluation Round2 (20260314_024824)

## Added comparisons

| Strategy | nDCG@10 | MRR@10 | Recall@10 | Avg Latency (ms) | P95 Latency (ms) |
|---|---:|---:|---:|---:|---:|
| adaptive (current easy->baseline, hard->hybrid top20/30) | 0.7388 | 0.6940 | 0.9881 | 95.2 | 270.6 |
| adaptive old-sim (easy->ltr, hard->hybrid) | 0.6532 | 0.6094 | 0.8571 | 73.9 | 133.1 |
| cross_encoder | 0.6950 | 0.6405 | 0.9643 | 263.9 | 385.6 |
| adaptive sim (easy->baseline, hard->hybrid top10) | 0.7322 | 0.6940 | 0.9762 | 53.7 | 137.4 |

## Interpretation
- 当前 `adaptive` 在质量指标上最佳（nDCG@10 / MRR@10）。
- `easy->ltr` 的旧行为在该评测集上明显退化，保持 `easy->baseline` 是正确方向。
- `hard->hybrid top10` 延迟明显更好，但 nDCG@10 与 Recall@10 有小幅下降，可作为“演示低延迟模式”备选。
- `cross_encoder` 单模型不是最优点，不建议替代当前 hard 专家。
