# Combined Candidate Check (2026-03-15)

| Config | Threshold | Avg CE Top-N | Hard Rate | nDCG@10 | MRR@10 | Recall@10 | Avg Latency (ms) | P95 Latency (ms) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| current | 0.4500 | 30.0 | 40.5% | 0.2077 | 0.2171 | 0.2619 | 107.2 | 245.6 |
| candidate | 0.6062 | 20.0 | 31.0% | 0.2077 | 0.2171 | 0.2619 | 68.4 | 156.8 |

- Quality is unchanged on this evaluation slice.
- Candidate reduces average latency by 38.8 ms and P95 latency by 88.8 ms.
