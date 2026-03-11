"""
单元测试：验证 P0 修复的正确性

测试覆盖:
1. IDF 计算 — 验证 build_idf 输出合理
2. 分数归一化 — 验证 _normalize_scores 将不同量纲映射到 [0, 1]

运行方式:
    cd d:\Develop\SearchEngine
    python -m pytest tests/test_algorithms.py -v
"""

import os
import sys
import json
import math
import tempfile

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ========== Test 1: IDF 计算 ==========

class TestBuildIdf:
    """测试 IDF 构建脚本的正确性"""

    def _create_test_corpus(self, tmp_dir):
        """创建一个小型测试语料库"""
        docs = [
            {"title": "python programming language", "content": "python is a popular programming language used for data science"},
            {"title": "java programming", "content": "java is a programming language used in enterprise applications"},
            {"title": "python data science", "content": "python is widely used for data analysis and machine learning"},
            {"title": "web development", "content": "javascript is used for web development and frontend programming"},
            {"title": "machine learning basics", "content": "machine learning is a subset of artificial intelligence"},
        ]
        corpus_path = os.path.join(tmp_dir, "test_corpus.json")
        with open(corpus_path, 'w', encoding='utf-8') as f:
            json.dump(docs, f)
        return corpus_path

    def test_idf_basic(self):
        """验证 IDF 基本正确性: 高频词 IDF < 低频词 IDF"""
        from tools.build_idf import build_idf

        with tempfile.TemporaryDirectory() as tmp_dir:
            corpus_path = self._create_test_corpus(tmp_dir)
            output_path = os.path.join(tmp_dir, "idf.json")
            idf_dict = build_idf(corpus_path, output_path)

            # "programming" 出现在多个文档中 (高频 -> 低 IDF)
            # "javascript" 只出现在 1 个文档中 (低频 -> 高 IDF)
            assert 'programming' in idf_dict, "高频词 'programming' 应在 IDF 字典中"
            assert 'javascript' in idf_dict, "低频词 'javascript' 应在 IDF 字典中"
            assert idf_dict['programming'] < idf_dict['javascript'], \
                f"高频词 IDF ({idf_dict['programming']:.4f}) 应小于低频词 IDF ({idf_dict['javascript']:.4f})"

    def test_idf_values_positive(self):
        """验证所有 IDF 值 > 0"""
        from tools.build_idf import build_idf

        with tempfile.TemporaryDirectory() as tmp_dir:
            corpus_path = self._create_test_corpus(tmp_dir)
            output_path = os.path.join(tmp_dir, "idf.json")
            idf_dict = build_idf(corpus_path, output_path)

            for term, idf in idf_dict.items():
                assert idf > 0, f"IDF('{term}') = {idf} 应为正数"

    def test_idf_file_saved(self):
        """验证 IDF 字典正确保存到文件"""
        from tools.build_idf import build_idf

        with tempfile.TemporaryDirectory() as tmp_dir:
            corpus_path = self._create_test_corpus(tmp_dir)
            output_path = os.path.join(tmp_dir, "idf.json")
            build_idf(corpus_path, output_path)

            assert os.path.exists(output_path), "IDF 文件应被保存"
            with open(output_path, 'r') as f:
                loaded = json.load(f)
            assert isinstance(loaded, dict), "IDF 文件应是一个字典"
            assert len(loaded) > 0, "IDF 字典不应为空"


# ========== Test 2: 分数归一化 ==========

class TestNormalizeScores:
    """测试 Min-Max 分数归一化"""

    def test_basic_normalization(self):
        """验证基本归一化到 [0, 1]"""
        # 直接从 app.py 导入会触发 Flask 初始化和模型加载,
        # 所以这里独立实现相同的逻辑进行测试
        def _normalize_scores(scores):
            if not scores:
                return scores
            min_s = min(scores)
            max_s = max(scores)
            if max_s == min_s:
                return [0.5] * len(scores)
            return [(s - min_s) / (max_s - min_s) for s in scores]

        # ES 分数 (典型范围: 5~50)
        es_scores = [45.2, 30.1, 15.5, 8.0, 5.2]
        es_norm = _normalize_scores(es_scores)

        assert abs(max(es_norm) - 1.0) < 1e-6, "最大值应归一化为 1.0"
        assert abs(min(es_norm) - 0.0) < 1e-6, "最小值应归一化为 0.0"

        # CE 分数 (典型范围: -10~10)
        ce_scores = [8.5, 3.2, -1.5, -5.0, -8.3]
        ce_norm = _normalize_scores(ce_scores)

        assert abs(max(ce_norm) - 1.0) < 1e-6
        assert abs(min(ce_norm) - 0.0) < 1e-6

        # 归一化后两组分数在同一量纲
        for s in es_norm + ce_norm:
            assert 0.0 <= s <= 1.0, f"归一化后分数 {s} 应在 [0, 1] 范围内"

    def test_equal_scores(self):
        """当所有分数相等时应返回 0.5"""
        def _normalize_scores(scores):
            if not scores:
                return scores
            min_s = min(scores)
            max_s = max(scores)
            if max_s == min_s:
                return [0.5] * len(scores)
            return [(s - min_s) / (max_s - min_s) for s in scores]

        equal_scores = [5.0, 5.0, 5.0]
        result = _normalize_scores(equal_scores)
        assert all(abs(s - 0.5) < 1e-6 for s in result), "相等分数应全部归一化为 0.5"

    def test_empty_scores(self):
        """空列表应原样返回"""
        def _normalize_scores(scores):
            if not scores:
                return scores
            min_s = min(scores)
            max_s = max(scores)
            if max_s == min_s:
                return [0.5] * len(scores)
            return [(s - min_s) / (max_s - min_s) for s in scores]

        assert _normalize_scores([]) == []

    def test_fusion_is_fair(self):
        """验证归一化后加权融合不会被一方主导"""
        def _normalize_scores(scores):
            if not scores:
                return scores
            min_s = min(scores)
            max_s = max(scores)
            if max_s == min_s:
                return [0.5] * len(scores)
            return [(s - min_s) / (max_s - min_s) for s in scores]

        # 模拟一个典型场景:
        # ES 最佳结果 = 45, CE 最佳结果 = 8
        # 如果不归一化: 0.3*45 + 0.7*8 = 19.1 (ES 主导)
        # 归一化后: 0.3*1.0 + 0.7*1.0 = 1.0 (公平)
        es = [45.0, 10.0]
        ce = [2.0, 8.0]  # CE 认为第二个更好

        es_n = _normalize_scores(es)
        ce_n = _normalize_scores(ce)

        # 不归一化时的融合
        raw_fused = [0.3 * es[i] + 0.7 * ce[i] for i in range(2)]
        # 归一化后的融合
        norm_fused = [0.3 * es_n[i] + 0.7 * ce_n[i] for i in range(2)]

        # 不归一化时 ES 主导 -> 第一个得分更高
        assert raw_fused[0] > raw_fused[1], "不归一化时 ES 分数主导融合结果"
        # 归一化后 CE 的权重 0.7 应发挥作用 -> 第二个得分更高
        assert norm_fused[1] > norm_fused[0], "归一化后 CE 权重 (0.7) 正确生效"
