# P0 修复 + 算法优化实施方案

## 概述

本次改动分两部分：修复 3 个高优先级 Bug，以及新增 **「伪相关反馈 (Pseudo-Relevance Feedback)」** 查询扩展算法。PRF 是信息检索领域的经典高级算法，体现了对 **两遍检索** 和 **Rocchio 模型** 的深入理解，非常适合在答辩中展示算法复杂度。

---

## Proposed Changes

### Part 1: P0 Bug Fixes

---

#### [MODIFY] [feature_extractor.py](file:///d:/Develop/SearchEngine/ranking/feature_extractor.py)

**修复 IDF 固定值问题**：新增 `build_idf_from_corpus()` 类方法，从 [data/msmarco_100k_processed.json](file:///d:/Develop/SearchEngine/data/msmarco_100k_processed.json) 读取全部文档，统计每个词的文档频率 (DF)，计算 `IDF = log(N / (DF + 1))`，结果持久化到 `models/idf_dict.json`。[__init__](file:///d:/Develop/SearchEngine/ranking/feature_extractor.py#28-33) 中自动加载该文件。

- 新增 `build_idf_from_corpus(data_path, output_path)` 静态方法
- 修改 [__init__](file:///d:/Develop/SearchEngine/ranking/feature_extractor.py#28-33) 的 `idf_cache` 初始化逻辑：尝试从 `models/idf_dict.json` 加载
- 修改 [_get_idf()](file:///d:/Develop/SearchEngine/ranking/feature_extractor.py#346-356) 使用加载的真实 IDF，未登录词降级为 [log(N)](file:///d:/Develop/SearchEngine/static/script.js#113-136) 而非固定 3.0

---

#### [MODIFY] [app.py](file:///d:/Develop/SearchEngine/app.py)

**修复 Cross-Encoder 分数融合**：对 [cross_encoder_rerank()](file:///d:/Develop/SearchEngine/app.py#325-341) 和 [hybrid_rerank()](file:///d:/Develop/SearchEngine/app.py#343-368) 中的分数混合实现 min-max 归一化，先将 ES 分数和 CE 分数各自归一化到 [0, 1]，再进行加权融合。

- 新增 `_normalize_scores(scores)` 辅助函数
- 修改 [cross_encoder_rerank()](file:///d:/Develop/SearchEngine/app.py#325-341) 和 [hybrid_rerank()](file:///d:/Develop/SearchEngine/app.py#343-368)
- 同时删除 [create_index_and_bulk_index()](file:///d:/Develop/SearchEngine/init.py#11-107) 冗余函数（索引创建统一由 [init.py](file:///d:/Develop/SearchEngine/init.py) 负责）
- `__main__` 入口不再调用索引创建，改为直接启动 Flask

---

#### [MODIFY] [init.py](file:///d:/Develop/SearchEngine/init.py)

添加幂等守卫：如果索引已存在且文档数量正确，则跳过重建，避免 Docker 重启时反复删除重建。

---

### Part 2: 伪相关反馈 (PRF) 查询扩展

> [!IMPORTANT]
> PRF 是信息检索领域的经典算法，核心思想是：**假设首次检索返回的前 K 个结果是相关的，从中提取高质量词项来扩展原始查询，然后用扩展后的查询进行第二次检索以提升召回和相关性。** 这体现了 Rocchio 反馈模型和两遍检索管线的设计思想。

---

#### [NEW] [query_expander.py](file:///d:/Develop/SearchEngine/ranking/query_expander.py)

新建 PRF 查询扩展模块，包含 `QueryExpander` 类：

```
class QueryExpander:
    """基于伪相关反馈 (PRF) 的查询扩展

    算法流程：
    1. 取首次检索的 Top-K 文档作为伪相关文档
    2. 对伪相关文档使用 TF-IDF 提取高权重词
    3. 用 Rocchio 公式计算扩展词权重:
       Q' = α·Q + β·(1/K)·Σ(D_i) - γ·(1/M)·Σ(D_j)
       其中正反馈取 Top-K，负反馈取 Bottom-M
    4. 选取权重最高的 N 个新词加入查询
    5. 用扩展查询进行第二轮检索
    """
```

关键方法：
- `expand_query(query, top_docs, bottom_docs=None, alpha=1.0, beta=0.75, gamma=0.15, num_expand_terms=5)` — Rocchio 公式扩展
- `_extract_doc_tfidf(documents)` — 从文档集合提取 TF-IDF 向量
- `_select_expansion_terms(query_terms, candidate_terms, num_terms)` — 选择区分度最高的扩展词

---

#### [MODIFY] [app.py](file:///d:/Develop/SearchEngine/app.py)

将 PRF 集成到搜索 API：

- `/search` 路由新增 `expand` 参数（`true/false`）
- 当开启时，在 Stage 1 (ES 召回) 后、Stage 2 (重排) 前插入 PRF 阶段：
  1. 用原始查询做首次检索（已有）
  2. 取 Top-5 结果进行 PRF 扩展
  3. 用扩展后的查询重新检索
  4. 合并去重后进入排序阶段
- 在日志中记录扩展词和 PRF 耗时

---

#### [MODIFY] [index.html](file:///d:/Develop/SearchEngine/templates/index.html)

在排序模式旁新增 "Query Expansion" 开关（checkbox）。

---

#### [MODIFY] [script.js](file:///d:/Develop/SearchEngine/static/script.js)

将 Query Expansion 开关的状态传递到 API 请求中。

---

### Part 3: IDF 预计算脚本

#### [NEW] [build_idf.py](file:///d:/Develop/SearchEngine/tools/build_idf.py)

独立脚本，从预处理数据构建 IDF 字典并保存到 `models/idf_dict.json`。可以独立运行，也被 [FeatureExtractor](file:///d:/Develop/SearchEngine/ranking/feature_extractor.py#22-432) 内部调用。

---

## Verification Plan

### Automated Tests

#### [NEW] [test_algorithms.py](file:///d:/Develop/SearchEngine/tests/test_algorithms.py)

使用 `pytest` 编写单元测试覆盖：

1. **IDF 计算** — 验证简单语料下 `build_idf_from_corpus` 输出合理（高频词 IDF < 低频词 IDF）
2. **分数归一化** — 验证 `_normalize_scores` 将不同量纲映射到 [0, 1]
3. **PRF 查询扩展** — 构造简单文档集，验证扩展词不含原始查询词、数量正确

运行命令：
```bash
cd d:\Develop\SearchEngine
pip install pytest
python -m pytest tests/test_algorithms.py -v
```

### Manual Verification

由用户在 Docker 环境启动后验证搜索界面功能正常（需要 Elasticsearch 运行）。
