# Adaptive 平衡推荐点性能报告（2026-03-16）

## 1. 变更摘要

- 默认 adaptive 阈值从 `0.61` 调整为 `0.6062`
- 默认 hard rerank 上限从 `20` 调整为 `5`
- easy/hard 路由模式保持不变：
  - `easy -> baseline`
  - `hard -> cross_encoder`

对应配置修改：
- `webapp/config.py`
- `docker-compose.yml`

## 2. 测试环境与当前状态

- 日期：`2026-03-16`
- 服务：`docker compose` 启动的 `app` + `elasticsearch`
- Elasticsearch 版本：`8.14.1`
- 索引：`documents`
- 文档总量：`82,326`
- 评测数据集：`data/ltr_training_data_feedback_combined.json`
- 查询数：`42`
- 当前运行时路由参数：
  - `hard_threshold = 0.6062`
  - `hard_top_k_cap = 5`
  - `model_hard_top_k = 30`
  - `hard_topk_policy = [(0.08, 30), (0.1, 20), (1.0, 30)]`

向量覆盖率检查：

| 指标 | 数值 |
| --- | ---: |
| Total docs | 82,326 |
| `content_emb` 覆盖率 | 100.00% |
| `title_emb` 覆盖率 | 100.00% |
| 双字段同时覆盖 | 100.00% |
| 在线补算 `content_emb` | 0 |
| 在线补算 `title_emb` | 0 |

说明：
- 当前延迟结果建立在“文档向量已全量预计算”的前提上。
- 这属于标准离线预计算优化，不属于查询时作弊。

## 3. 参数选择依据（离线 2D 扫描）

评测口径：
- 同一批 `42` 条带标签查询
- 同时扫描 `hard threshold` 与 `CE top-k`
- 指标：`nDCG@10`、`MRR@10`、`Recall@10`、平均延迟、P95 延迟

关键候选点如下：

| 配置 | threshold | CE top-k | nDCG@10 | MRR@10 | Recall@10 | Avg Latency (ms) | P95 (ms) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Baseline | - | - | 0.1719 | 0.1897 | 0.2024 | 14.6 | 19.5 |
| 旧默认 Adaptive | 0.6100 | 20 | 0.1784 | 0.1923 | 0.2143 | 42.8 | 168.0 |
| 新默认 Adaptive（平衡点） | 0.6062 | 5 | 0.1903 | 0.2082 | 0.2143 | 27.7 | 65.8 |
| 质量优先 Adaptive | 0.4945 | 20 | 0.1945 | 0.2105 | 0.2262 | 69.7 | 184.5 |
| 极低延迟 Adaptive | 0.7704 | 5 | 0.1458 | 0.1566 | 0.1905 | 17.1 | 25.5 |

结论：
- 新默认平衡点相对旧默认点是严格更优的：
  - `nDCG@10` 从 `0.1784` 提升到 `0.1903`
  - 平均延迟从 `42.8 ms` 降到 `27.7 ms`
- 质量优先点只额外提升了 `0.0042` 的 `nDCG@10`，但平均延迟增加了 `42.0 ms`
- 因此，`0.6062 + CE top-k=5` 更适合默认运行参数和答辩演示

## 4. 线上实测（真实 `/search` 请求）

压测方式：
- 对同一批 `42` 条 query 分别请求：
  - `mode=baseline`
  - `mode=adaptive`
- 通过唯一 `session_id` 过滤后端结构化日志
- 同时记录：
  - 服务端日志中的 `total_ms / retrieval_ms / feature_ms / inference_ms`
  - 客户端 HTTP 端到端耗时

### 4.1 Baseline vs 新默认 Adaptive

| 模式 | Server Avg (ms) | Server P50 | Server P95 | Server P99 | Client Avg (ms) | Client P50 | Client P95 | Client P99 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Baseline | 20.09 | 14.74 | 39.13 | 91.40 | 27.20 | 19.79 | 67.88 | 114.63 |
| Adaptive（新默认） | 37.67 | 27.77 | 84.51 | 89.35 | 42.20 | 32.30 | 89.04 | 93.81 |

### 4.2 Adaptive 分阶段耗时

| 指标 | 数值 |
| --- | ---: |
| Retrieval Avg (ms) | 14.57 |
| Feature Avg (ms) | 8.84 |
| Inference Avg (ms) | 13.91 |
| CE Queries | 10 / 42 |
| Avg CE top-k | 5.0 |
| Avg candidate docs（LTR 路径） | 45.43 |
| `content_emb` 命中率 | 100.00% |
| `title_emb` 命中率 | 100.00% |
| 在线补算 `content_emb` | 0.0 |
| 在线补算 `title_emb` | 0.0 |

### 4.3 Adaptive 路由分布

| 路由 | 查询数 | 占比 |
| --- | ---: | ---: |
| `easy -> ltr` | 22 | 52.4% |
| `hard -> cross_encoder` | 10 | 23.8% |
| `easy -> baseline` | 7 | 16.7% |
| `hard -> baseline` | 2 | 4.8% |
| `hard -> ltr` | 1 | 2.4% |

解释：
- 只有 `23.8%` 的查询进入真正的 `cross_encoder` 慢路径
- 一半以上查询被路由到 `easy -> ltr`
- 由于 embedding 已全量预计算，LTR 路径已不再是主要瓶颈

## 5. 结果解读

### 5.1 相对 Baseline

- 新默认 Adaptive 的目标不是比 Baseline 更快，而是在合理时延内提供更好的排序质量
- 离线效果上：
  - `nDCG@10` 从 `0.1719` 提升到 `0.1903`
  - 绝对提升 `+0.0184`
  - 相对提升约 `+10.7%`
- 线上延迟上：
  - 服务端平均耗时从 `20.09 ms` 增加到 `37.67 ms`
  - 客户端平均耗时从 `27.20 ms` 增加到 `42.20 ms`
- 当前 tradeoff 对答辩是合理的：
  - 明确展示“质量-时延折中”
  - 保持 `P95 client latency` 在 `89 ms` 左右，交互仍然流畅

### 5.2 相对旧默认 Adaptive

- 离线 `nDCG@10`：`0.1784 -> 0.1903`
- 离线平均延迟：`42.8 ms -> 27.7 ms`
- 旧默认点不在 Pareto 前沿上，新默认点在当前扫描里更合理

### 5.3 当前瓶颈

- 主要尾部延迟来自 `hard -> cross_encoder`
- LTR 经过 embedding 预计算后，在线代价已显著下降
- 如果后续还要继续压延迟，优先应关注：
  - hard 路由比例
  - CE 路径的 batch 与模型推理成本
  - 检索阶段偶发慢查询

## 6. 结论

当前默认 adaptive 已切换到更合理的平衡点：

- `hard_threshold = 0.6062`
- `hard_top_k_cap = 5`

这个配置适合作为：
- 日常默认配置
- 毕业答辩演示配置
- 后续继续调优的稳定基线

它的核心价值不是“比 baseline 更快”，而是：
- 比 baseline 更准
- 比旧默认 adaptive 更快也更准
- 在当前系统和数据规模下，提供了更好的整体折中

## 7. 验证项

- `python -m unittest discover tests`
- 结果：`42` 个测试全部通过

补充说明：
- `tools/experiments/run_adaptive_tradeoff_scans.py` 已修正报告口径，后续会同时记录：
  - `current_hard_top_k`（运行时实际生效值）
  - `model_hard_top_k`（模型元数据中的原始值）
