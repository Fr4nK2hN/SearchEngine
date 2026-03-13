# Algorithm Main Table & Ablation (Multi-Seed, Final Round)

- Date: `2026-03-13` (Asia/Shanghai)
- Output dir: `models/adaptive_ablation_eval_v2_quantile_round3`
- Script: `tools/run_adaptive_ablation_eval.py`

## Experiment Setup

- Data: `data/ltr_training_data_v2_quantile.json`
- Clean sample count: `75`
- Seed list: `0,1,2,3,4`
- Per-seed sampled queries: `60`
- Router config (from model):
  - `easy_mode=ltr`
  - `hard_mode=hybrid`
  - `hard_threshold=0.45`
  - `hard_top_k=30`
- Adaptive top-k policy (final): `0.08:30,0.10:20,1.00:30`

## Main Results Table

| Method | nDCG@10 (mean±std) | MRR@10 (mean±std) | Recall@50 (mean±std) | Avg Latency ms (mean±std) | P95 Latency ms (mean±std) | Hard Rate (mean±std) |
| --- | --- | --- | --- | --- | --- | --- |
| Baseline | 0.7546±0.0036 | 1.0000±0.0000 | 1.0000±0.0000 | 0.00±0.00 | 0.00±0.00 | - |
| LTR | 0.7914±0.0044 | 1.0000±0.0000 | 1.0000±0.0000 | 266.54±10.37 | 573.71±24.07 | - |
| Cross-Encoder | 0.9124±0.0041 | 1.0000±0.0000 | 1.0000±0.0000 | 230.59±1.29 | 279.36±1.92 | - |
| Hybrid | 0.8127±0.0037 | 1.0000±0.0000 | 1.0000±0.0000 | 343.40±10.23 | 646.05±21.83 | - |
| Adaptive | 0.8576±0.0024 | 1.0000±0.0000 | 1.0000±0.0000 | 369.77±12.94 | 713.19±28.09 | 45.33%±2.45% |

## Ablation Table

| Method | nDCG@10 (mean±std) | MRR@10 (mean±std) | Recall@50 (mean±std) | Avg Latency ms (mean±std) | P95 Latency ms (mean±std) | Hard Rate (mean±std) |
| --- | --- | --- | --- | --- | --- | --- |
| Fixed-LTR (No Routing) | 0.7914±0.0044 | 1.0000±0.0000 | 1.0000±0.0000 | 266.54±10.37 | 573.71±24.07 | - |
| Fixed-Hybrid@30 (No Routing) | 0.9018±0.0027 | 1.0000±0.0000 | 1.0000±0.0000 | 497.13±10.00 | 790.86±25.12 | - |
| Router + Fixed Top-K | 0.8576±0.0024 | 1.0000±0.0000 | 1.0000±0.0000 | 369.77±12.94 | 713.19±28.09 | 45.33%±2.45% |
| Router + Adaptive Top-K | 0.8576±0.0024 | 1.0000±0.0000 | 1.0000±0.0000 | 368.75±13.27 | 713.19±28.09 | 45.33%±2.45% |

## Observations

- Quality-wise, `Cross-Encoder` and `Fixed-Hybrid@30` are strongest on this dataset.
- Current `Adaptive` (router + fixed top-k) sits between LTR and full hard reranking in both quality and latency.
- Final adaptive top-k policy achieves **no nDCG drop** against fixed top-k in this multi-seed run, with a small latency gain:
  - `ΔnDCG@10 = 0.0000`
  - `ΔAvgLatency = -1.02 ms` (adaptive vs fixed top-k)

## Important Data Characteristic

On `data/ltr_training_data_v2_quantile.json`:

- `first_doc_is_max_ratio = 97.33%`

This strongly compresses MRR/Recall differences and makes nDCG the primary discriminative metric.

## Repro Command

```bash
docker compose exec -T app python tools/run_adaptive_ablation_eval.py \
  --data data/ltr_training_data_v2_quantile.json \
  --router-model models/query_router.pkl \
  --ltr-model models/ltr_model.pkl \
  --output-dir models/adaptive_ablation_eval_v2_quantile_round3 \
  --seeds 0,1,2,3,4 \
  --max-queries 60 \
  --adaptive-topk 0.08:30,0.10:20,1.00:30
```
