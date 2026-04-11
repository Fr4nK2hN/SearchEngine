# 系统性能分析（2026-04-07）

## 1. 分析对象与口径

本次分析基于当前正在运行的系统实例：

- `app` 容器在线
- `elasticsearch` 容器在线
- 当前索引：`documents`

本次分析分为三部分：

1. 当前运行环境与索引规模
2. 当前实时 benchmark
3. 近阶段真实日志对照

说明：

- 历史 `logs/events.log` 混有 2026-03 的旧数据，不能直接代表当前系统状态
- 因此本次结论以 `2026-04-07` 新增实测请求为主

## 2. 当前运行环境

### 2.1 容器资源占用

采样时间：`2026-04-07`

| 服务 | CPU | 内存 |
| :--- | :--- | :--- |
| `app` | `0.30%` | `785.7 MiB / 7.653 GiB` |
| `elasticsearch` | `1.07%` | `4.42 GiB / 7.653 GiB` |

结论：

- 当前主要内存消耗在 Elasticsearch，不在 Flask / 排序服务
- `app` 侧内存占用处于可接受范围

### 2.2 索引规模

索引统计：

- 文档数：`82,326`
- 删除文档数：`19,326`
- 主分片存储大小：约 `1.3 GiB`
- 索引健康状态：`yellow`

结论：

- 当前性能结果是在 `8.2 万` 级文档规模下取得的
- `yellow` 主要是单节点副本未分配，不直接影响当前查询速度，但说明高可用性未完善
- 删除文档较多，后续可以关注 segment merge / force merge 对存储效率的影响

### 2.3 embedding 覆盖率

覆盖率分析结果：

- `content_emb`: `82,326 / 82,326`，`100.00%`
- `title_emb`: `82,326 / 82,326`，`100.00%`
- 两者同时存在：`100.00%`

结论：

- 文档向量已经完成全量预计算
- 当前线上查询不需要再临时补算 embedding

## 3. 当前运行配置

从 `/model_info` 读取到的当前关键配置：

- `adaptive easy_mode = baseline`
- `adaptive hard_mode = cross_encoder`
- `hard_threshold = 0.6062`
- `hard_top_k_cap = 5`

说明：

- Router 模型文件里仍保留 `hard_top_k = 30`
- 但运行时真正生效的是 `cap = 5`
- 也就是说，当前 hard 查询只会对前 `5` 个候选执行重排序

## 4. 实时 Benchmark

### 4.1 测试方法

测试时间：`2026-04-07`

固定查询集共 `8` 条：

- `python tutorial`
- `how to use java`
- `data intensive computing`
- `job condition`
- `semantic search`
- `machine learning basics`
- `Donald Trump`
- `what is data mining`

对每条查询分别测试以下 5 种策略：

- `baseline`
- `adaptive`
- `ltr`
- `cross_encoder`
- `hybrid`

总请求数：`40`

统计口径：

- 服务端指标来自 `logs/events.log` 中对应请求的 `Search successful` 日志
- 采用 `(mode, query)` 去重，剔除 warm-up 请求

### 4.2 服务端延迟结果

| 策略 | 样本数 | 平均总耗时 (ms) | P50 (ms) | P95 (ms) | P99 (ms) | 平均检索 (ms) | 平均特征 (ms) | 平均推理 (ms) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `baseline` | 8 | `19.21` | `15.94` | `33.88` | `33.88` | `19.21` | `0.00` | `0.00` |
| `ltr` | 8 | `23.20` | `19.18` | `38.62` | `38.62` | `12.81` | `9.62` | `0.73` |
| `adaptive` | 8 | `38.55` | `25.28` | `63.01` | `63.01` | `14.17` | `5.87` | `18.26` |
| `hybrid` | 8 | `95.01` | `83.14` | `147.39` | `147.39` | `13.84` | `2.00` | `79.11` |
| `cross_encoder` | 8 | `327.93` | `340.58` | `665.43` | `665.43` | `15.92` | `0.00` | `312.01` |

### 4.3 关键观察

#### 1. `baseline` 仍然是最快路径

- 平均仅 `19.21 ms`
- 所有开销都集中在 Elasticsearch 检索阶段

#### 2. `ltr` 已经非常接近 `baseline`

- 平均 `23.20 ms`
- 相比 `baseline` 只增加约 `4 ms`
- 当前 LTR 特征提取平均仅 `9.62 ms`

这说明前面做的 embedding 预计算已经明显生效。

#### 3. `adaptive` 是当前最合理的折中

- 平均 `38.55 ms`
- 明显快于 `hybrid` 和 `cross_encoder`
- 又比纯 `baseline` 保留了更强的重排序能力

#### 4. `cross_encoder` 仍然是当前最重的路径

- 平均 `327.93 ms`
- P95 达到 `665.43 ms`
- 主要瓶颈几乎全部来自推理阶段

#### 5. `hybrid` 明显比纯 `cross_encoder` 更轻，但仍然不算快

- 平均 `95.01 ms`
- 主要开销仍在推理阶段

## 5. Adaptive 路由分析

本轮 `adaptive` 的 8 条查询路由结果：

- `hard -> cross_encoder`: `3`
- `easy -> baseline`: `3`
- `easy -> ltr`: `2`

比例为：

- hard 路径：`37.5%`
- easy baseline：`37.5%`
- easy ltr：`25.0%`

结论：

- 当前 adaptive 并不是“一律走重模型”
- 大约 `62.5%` 的查询都被压在轻路径上
- 这也是当前 adaptive 延迟能够控制在 `38.55 ms` 的主要原因

## 6. 近阶段真实日志对照

筛选 `2026-04-01` 之后的 `Search successful` 日志，共 `46` 条：

| 实际策略 | 样本数 | 平均总耗时 (ms) | P95 (ms) |
| :--- | :--- | :--- | :--- |
| `Adaptive (easy -> baseline)` | 4 | `51.98` | `161.48` |
| `Adaptive (easy -> ltr)` | 3 | `79.38` | `152.42` |
| `Adaptive (hard -> cross_encoder)` | 6 | `80.95` | `132.98` |
| `LTR` | 9 | `25.28` | `41.86` |
| `Baseline` | 8 | `19.21` | `33.88` |
| `Cross-Encoder` | 8 | `327.93` | `665.43` |
| `Hybrid (top-10)` | 8 | `95.01` | `147.39` |

说明：

- 这组真实日志样本仍然较少
- 但整体趋势与实时 benchmark 一致
- 说明当前结论并不是只在测试请求中成立

## 7. 当前瓶颈判断

### 7.1 已经解决的瓶颈

- 文档 embedding 在线补算
- LTR 特征提取的高延迟问题

证据：

- embedding 命中率 `100%`
- `LTR` 平均仅 `23.20 ms`

### 7.2 当前主要瓶颈

当前最主要的性能瓶颈是：

- `Cross-Encoder` 推理时间

证据：

- `cross_encoder` 平均推理 `312.01 ms`
- `hybrid` 平均推理 `79.11 ms`
- adaptive 一旦走 hard 路径，总耗时就明显上升

### 7.3 次级瓶颈

- Elasticsearch 是系统最大内存占用方
- 当前索引有 `19,326` 删除文档，说明存在一定段合并与空间浪费风险

## 8. 结论

当前系统已经达到一个比较清晰的性能分层：

- `baseline`：最快，约 `19 ms`
- `ltr`：接近 baseline，约 `23 ms`
- `adaptive`：质量与性能折中，约 `39 ms`
- `hybrid`：明显更重，约 `95 ms`
- `cross_encoder`：最慢，约 `328 ms`

对当前项目而言，可以给出如下判断：

1. 系统当前已经具备“可演示、可答辩”的响应性能
2. `LTR` 经过优化后已不再是主要瓶颈
3. `adaptive` 是当前最合理的默认在线策略
4. 如果后续继续优化，重点应放在 hard 路径的 Cross-Encoder 推理成本，而不是继续压 LTR

## 9. 后续优化建议

### 优先级 1

继续压缩 `hard -> cross_encoder` 的成本：

- 降低 hard 查询比例
- 限制 hard 路径 top-k
- 研究更轻量的 cross-encoder 模型

### 优先级 2

完善日志分析口径：

- 给性能报表增加时间窗口过滤
- 避免历史旧日志混入当前结果

### 优先级 3

优化 Elasticsearch 维护状态：

- 关注删除文档带来的段膨胀
- 视情况进行 force merge / 重建索引

