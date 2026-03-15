| Expert | nDCG@10 | MRR@10 | Recall@50 | Avg Latency (ms) | P95 Latency (ms) |
| --- | --- | --- | --- | --- | --- |
| Baseline | 0.7342 | 0.6863 | 1.0000 | 0.00 | 0.00 |
| LTR | 0.7661 | 0.7614 | 1.0000 | 228.06 | 473.93 |
| Cross-Encoder | 0.6984 | 0.6332 | 1.0000 | 76.67 | 89.16 |
| Hybrid | 0.7624 | 0.7179 | 1.0000 | 92.58 | 105.17 |
| **Oracle (upper bound)** | **0.8703** | **0.8492** | **1.0000** | **75.97** | **232.87** |

- Best single expert: `LTR`
- Oracle nDCG@10 gain: `0.1042` (13.60%)
- LTR source: `pretrained`
- Data profile: queries=42 prefix_like_ratio=31.0% first_doc_is_max_ratio=47.6%

### Warnings
- Prefix-like query ratio is 31.0%; evaluation may not reflect real search intent.
