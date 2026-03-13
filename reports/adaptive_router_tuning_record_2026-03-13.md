# Adaptive Router Tuning Record

- Generated on: `2026-03-13` (Asia/Shanghai)
- Tuning output dir: `models/adaptive_tuning_v2_quantile`
- Grid search script: `tools/tune_adaptive_router.py`
- Deployment model: `models/query_router.pkl`

## Inputs

- Data: `data/ltr_training_data_v2_quantile.json`
- Router model: `models/query_router.pkl`
- LTR model: `models/ltr_model.pkl`
- Sample method: `random`
- Max queries: `60`
- Threshold grid: `0.35, 0.45, 0.50, 0.55, 0.65`
- Hard top-k grid: `10, 20, 30`

## Recommendation

Unconstrained best (max nDCG):

- `threshold=0.35`
- `hard_top_k=30`
- `nDCG@10=0.9422`
- `avg latency=489.43ms`
- `hard rate=95.00%`

Constrained recommendation (`max-hard-rate=0.60`):

- `threshold=0.45`
- `hard_top_k=30`
- `nDCG@10=0.8826`
- `MRR@10=1.0000`
- `Recall@50=1.0000`
- `avg latency=362.51ms`
- `p95 latency=720.43ms`
- `hard rate=45.00%`

## Top 5 Configs (by nDCG then latency)

| Threshold | Hard Top-K | nDCG@10 | Avg Latency (ms) | P95 (ms) | Hard Rate |
| --- | --- | --- | --- | --- | --- |
| 0.35 | 30 | 0.9422 | 489.43 | 797.57 | 95.00% |
| 0.45 | 30 | 0.8826 | 362.51 | 720.43 | 45.00% |
| 0.50 | 30 | 0.8601 | 345.04 | 704.42 | 36.67% |
| 0.55 | 30 | 0.8553 | 334.67 | 704.42 | 31.67% |
| 0.65 | 30 | 0.8257 | 302.54 | 640.27 | 16.67% |

## Applied Changes

- Re-exported router with recommended parameters:
  - `hard_threshold=0.45`
  - `hard_top_k=30`
  - model file: `models/query_router.pkl`
  - training summary: `models/router_baseline_feedback_combined.json`
  - label policy: `expert_pool`
  - export mode: `fit_on_full_data=true`

## Notes

- This tuning used offline replay-like evaluation, not online live traffic A/B.
- `threshold=0.35` has the highest nDCG but routes 95% queries to hard path; not ideal for adaptive behavior.
- Current deployment uses the constrained recommendation (`threshold=0.45`) to preserve routing diversity and latency.

## Round 2 (Adaptive Top-K Policy)

- Added runtime `hard_topk_policy` support in router model/application.
- Selected policy: `0.08:30,0.10:20,1.00:30`
  - Means for hard queries:
    - `delta = hard_prob - threshold <= 0.08`: use top-30
    - `0.08 < delta <= 0.10`: use top-20
    - `delta > 0.10`: use top-30
- Offline multi-seed ablation (see `models/adaptive_ablation_eval_v2_quantile_round3`):
  - `Router + Fixed Top-K`: nDCG@10 `0.8576`, avg latency `369.77ms`
  - `Router + Adaptive Top-K`: nDCG@10 `0.8576`, avg latency `368.75ms`
  - This round achieved non-inferior quality with a small latency reduction.
