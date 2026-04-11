# Latest Search Quality and Latency Evaluation

- Generated at: `2026-04-11T15:42:47.031181+00:00`
- Dataset: `data/ltr_training_data_feedback_combined.json`
- Base URL: `http://127.0.0.1:5000`
- Top-K: `10`
- Query count: `42`

| Mode | NDCG@10 | Avg Response (ms) | P50 (ms) | P95 (ms) | Avg Results |
| --- | ---: | ---: | ---: | ---: | ---: |
| `baseline` | 0.1719 | 93.18 | 85.34 | 183.53 | 10.00 |
| `ltr` | 0.1236 | 53.56 | 51.31 | 76.04 | 10.00 |
| `adaptive` | 0.1903 | 53.26 | 36.94 | 108.94 | 10.00 |
| `hybrid` | 0.1343 | 121.61 | 118.67 | 149.06 | 10.00 |
| `cross_encoder` | 0.1764 | 457.28 | 455.49 | 657.94 | 10.00 |

## Notes

- NDCG@10 is computed against the feedback-labeled dataset by matching returned document IDs.
- Response time is measured as client-side HTTP elapsed time for the current `/search` API.
- The chart is generated as an SVG for direct use in slides.
