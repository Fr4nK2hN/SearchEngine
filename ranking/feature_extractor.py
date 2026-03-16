import json
import math
import os
import re
from collections import Counter, OrderedDict
from difflib import SequenceMatcher

import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

try:
    from nltk.corpus import stopwords as nltk_stopwords
except Exception:  # pragma: no cover - optional dependency
    nltk_stopwords = None


TOKEN_PATTERN = re.compile(r"[a-z0-9]+")


def load_stop_words():
    """Load stop words without triggering runtime downloads."""
    if nltk_stopwords is not None:
        try:
            return set(nltk_stopwords.words("english"))
        except LookupError:
            pass
    return set(ENGLISH_STOP_WORDS)


class FeatureExtractor:
    """
    高级特征提取器
    包含文本匹配、语义相似度、统计特征等多个维度
    """

    DEFAULT_IDF_PATH = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "models",
        "idf_dict.json",
    )

    def __init__(self, idf_path=None):
        self.semantic_model = SentenceTransformer("all-MiniLM-L6-v2")
        self.stop_words = load_stop_words()
        try:
            self._emb_cache_limit = max(0, int(os.getenv("FEATURE_EMBED_CACHE_SIZE", "2048")))
        except ValueError:
            self._emb_cache_limit = 2048
        try:
            self._emb_batch_size = max(1, int(os.getenv("FEATURE_EMBED_BATCH_SIZE", "32")))
        except ValueError:
            self._emb_batch_size = 32
        self._emb_cache = OrderedDict()
        try:
            self._text_cache_limit = max(0, int(os.getenv("FEATURE_TEXT_CACHE_SIZE", "4096")))
        except ValueError:
            self._text_cache_limit = 4096
        self._text_cache = OrderedDict()

        idf_file = idf_path or self.DEFAULT_IDF_PATH
        if os.path.exists(idf_file):
            with open(idf_file, "r", encoding="utf-8") as handle:
                self.idf_cache = json.load(handle)
            self._default_idf = max(self.idf_cache.values()) if self.idf_cache else 3.0
            print(f"✓ IDF 字典已加载: {len(self.idf_cache)} 个词项 (默认 IDF={self._default_idf:.2f})")
        else:
            self.idf_cache = {}
            self._default_idf = 3.0
            print(f"⚠ IDF 字典未找到 ({idf_file})，使用默认 IDF={self._default_idf}")

    def extract_all_features(self, query, document, es_score=None, query_emb=None):
        """
        提取所有特征，返回特征字典

        Args:
            query: 查询字符串
            document: 文档字典，包含 title, content 等字段
            es_score: Elasticsearch 原始分数（可选）

        Returns:
            dict: 特征名称到特征值的映射
        """
        features = {}

        doc_content = document.get("content", "")
        doc_title = document.get("title", "")
        query_analysis = self._analyze_text(query)
        content_analysis = self._analyze_text(doc_content)
        title_analysis = self._analyze_text(doc_title)
        content_head_analysis = self._analyze_text(content_analysis["lower"][:100])

        query_lower = query_analysis["lower"]
        query_terms = query_analysis["tokens"]
        query_term_set = query_analysis["token_set"]

        features.update(
            self._extract_literal_features(
                query_lower,
                query_terms,
                query_term_set,
                content_analysis,
                title_analysis,
            )
        )

        features.update(
            self._extract_position_features(
                query_terms,
                query_term_set,
                content_analysis,
                content_head_analysis,
                title_analysis,
            )
        )

        if query_emb is None:
            query_emb = self._encode_text(query)
        content_emb = None
        title_emb = None
        if isinstance(document.get("content_emb"), (list, tuple, np.ndarray)):
            content_emb = np.asarray(document.get("content_emb"), dtype=float)
        if isinstance(document.get("title_emb"), (list, tuple, np.ndarray)):
            title_emb = np.asarray(document.get("title_emb"), dtype=float)
        features.update(
            self._extract_semantic_features(
                query,
                doc_content,
                doc_title,
                query_emb=query_emb,
                content_emb=content_emb,
                title_emb=title_emb,
            )
        )

        features.update(
            self._extract_statistical_features(
                query_terms,
                content_analysis,
                title_analysis,
            )
        )

        features.update(
            self._extract_term_importance_features(
                query_terms,
                content_analysis,
            )
        )

        features.update(
            self._extract_bm25_features(
                query_terms,
                content_analysis,
                title_analysis,
            )
        )

        if es_score is not None:
            features["es_original_score"] = float(es_score)
            features["es_log_score"] = math.log(es_score + 1)

        return features

    def _tokenize(self, text):
        """分词并去除停用词。"""
        return self._tokenize_lower((text or "").lower())

    def _tokenize_lower(self, lower_text):
        if not lower_text:
            return []
        return [
            token
            for token in TOKEN_PATTERN.findall(lower_text)
            if token not in self.stop_words
        ]

    def _remember_cache_value(self, cache, cache_limit, key, value):
        if cache_limit <= 0:
            return value
        cache[key] = value
        cache.move_to_end(key)
        while len(cache) > cache_limit:
            cache.popitem(last=False)
        return value

    def _analyze_text(self, text):
        cache_key = text or ""
        cached = self._text_cache.get(cache_key)
        if cached is not None:
            self._text_cache.move_to_end(cache_key)
            return cached

        lower_text = cache_key.lower()
        tokens = self._tokenize_lower(lower_text)
        first_sentence_lower = lower_text.split(".", 1)[0] if lower_text else ""
        first_sentence_tokens = (
            tokens if first_sentence_lower == lower_text else self._tokenize_lower(first_sentence_lower)
        )
        analysis = {
            "lower": lower_text,
            "tokens": tokens,
            "token_set": set(tokens),
            "term_freqs": Counter(tokens),
            "word_count": len(cache_key.split()),
            "first_sentence_token_set": set(first_sentence_tokens),
        }
        return self._remember_cache_value(
            self._text_cache,
            self._text_cache_limit,
            cache_key,
            analysis,
        )

    def _cosine_similarity(self, left, right):
        left_arr = np.asarray(left, dtype=float)
        right_arr = np.asarray(right, dtype=float)
        denominator = np.linalg.norm(left_arr) * np.linalg.norm(right_arr)
        if denominator <= 0.0:
            return 0.0
        return float(np.dot(left_arr, right_arr) / denominator)

    # ==================== 字面匹配特征 ====================

    def _extract_literal_features(
        self,
        query_lower,
        query_terms,
        query_term_set,
        content_analysis,
        title_analysis,
    ):
        """字面匹配相关特征"""
        features = {}

        doc_content_lower = content_analysis["lower"]
        doc_title_lower = title_analysis["lower"]

        features["exact_match_content"] = 1.0 if query_lower in doc_content_lower else 0.0
        features["exact_match_title"] = 1.0 if query_lower in doc_title_lower else 0.0

        if query_term_set:
            covered = query_term_set & content_analysis["token_set"]
            features["term_coverage"] = len(covered) / len(query_term_set)
            features["covered_term_count"] = len(covered)
        else:
            features["term_coverage"] = 0.0
            features["covered_term_count"] = 0

        features["ordered_term_match"] = self._ordered_term_match(query_terms, doc_content_lower)
        features["fuzzy_match_score"] = self._fuzzy_match_score(query_lower, doc_content_lower[:500])

        return features

    def _ordered_term_match(self, query_terms, doc_content):
        """检查查询词是否按顺序出现在文档中"""
        if not query_terms:
            return 0.0

        last_pos = -1
        matched = 0

        for term in query_terms:
            pos = doc_content.find(term, last_pos + 1)
            if pos > last_pos:
                matched += 1
                last_pos = pos

        return matched / len(query_terms)

    def _fuzzy_match_score(self, query, doc_excerpt):
        """使用 Levenshtein 距离计算近似匹配分数"""
        return SequenceMatcher(None, query, doc_excerpt).ratio()

    # ==================== 位置特征 ====================

    def _extract_position_features(
        self,
        query_terms,
        query_term_set,
        content_analysis,
        content_head_analysis,
        title_analysis,
    ):
        """查询词在文档中的位置特征"""
        features = {}

        doc_content_lower = content_analysis["lower"]

        if query_term_set:
            title_match = query_term_set & title_analysis["token_set"]
            features["title_match_ratio"] = len(title_match) / len(query_term_set)
        else:
            features["title_match_ratio"] = 0.0

        if query_term_set:
            first_match = query_term_set & content_analysis["first_sentence_token_set"]
            features["first_sentence_match"] = len(first_match) / len(query_term_set)
        else:
            features["first_sentence_match"] = 0.0

        positions = []
        for term in query_terms:
            pos = doc_content_lower.find(term)
            if pos >= 0:
                positions.append(pos)

        if positions:
            doc_length = len(doc_content_lower)
            avg_position = sum(positions) / len(positions)
            features["avg_term_position"] = avg_position / doc_length if doc_length > 0 else 1.0
            features["min_term_position"] = min(positions) / doc_length if doc_length > 0 else 1.0
        else:
            features["avg_term_position"] = 1.0
            features["min_term_position"] = 1.0

        features["term_density_first_100"] = self._term_density(
            query_term_set,
            content_head_analysis["tokens"],
        )
        features["term_density_full"] = self._term_density(
            query_term_set,
            content_analysis["tokens"],
        )

        return features

    def _term_density(self, query_term_set, text_terms):
        """计算查询词在文本中的密度"""
        if not text_terms:
            return 0.0

        matches = sum(1 for term in text_terms if term in query_term_set)
        return matches / len(text_terms)

    # ==================== 语义特征 ====================

    def _extract_semantic_features(
        self,
        query,
        doc_content,
        doc_title,
        query_emb=None,
        content_emb=None,
        title_emb=None,
    ):
        """基于语义嵌入的特征"""
        features = {}

        try:
            if query_emb is None:
                query_emb = self._encode_text(query)
            if content_emb is None:
                content_emb = self._encode_text(doc_content[:512])

            features["semantic_sim_content"] = self._cosine_similarity(query_emb, content_emb)

            if doc_title:
                if title_emb is None:
                    title_emb = self._encode_text(doc_title)
                features["semantic_sim_title"] = self._cosine_similarity(query_emb, title_emb)
                features["title_content_consistency"] = self._cosine_similarity(title_emb, content_emb)
            else:
                features["semantic_sim_title"] = 0.0
                features["title_content_consistency"] = 0.0

        except Exception as exc:
            print(f"Error in semantic features: {exc}")
            features["semantic_sim_content"] = 0.0
            features["semantic_sim_title"] = 0.0
            features["title_content_consistency"] = 0.0

        return features

    def _encode_text(self, text):
        key = text or ""
        if key in self._emb_cache:
            self._emb_cache.move_to_end(key)
            return self._emb_cache[key]
        emb = self.semantic_model.encode(key)
        return self._remember_cache_value(
            self._emb_cache,
            self._emb_cache_limit,
            key,
            emb,
        )

    def _encode_many_texts(self, texts):
        results = [None] * len(texts)
        missing_indices_by_key = {}

        for idx, text in enumerate(texts):
            key = text or ""
            cached = self._emb_cache.get(key)
            if cached is not None:
                self._emb_cache.move_to_end(key)
                results[idx] = cached
                continue
            missing_indices_by_key.setdefault(key, []).append(idx)

        if missing_indices_by_key:
            missing_keys = list(missing_indices_by_key.keys())
            encoded = self.semantic_model.encode(
                missing_keys,
                batch_size=self._emb_batch_size,
                show_progress_bar=False,
            )
            for key, emb in zip(missing_keys, encoded):
                stored = self._remember_cache_value(
                    self._emb_cache,
                    self._emb_cache_limit,
                    key,
                    np.asarray(emb, dtype=float),
                )
                for idx in missing_indices_by_key[key]:
                    results[idx] = stored

        return results

    def hydrate_document_embeddings(self, documents):
        content_indices = []
        content_texts = []
        title_indices = []
        title_texts = []
        precomputed_content = 0
        precomputed_title = 0

        for idx, document in enumerate(documents):
            if isinstance(document.get("content_emb"), (list, tuple, np.ndarray)):
                precomputed_content += 1
            else:
                content_indices.append(idx)
                content_texts.append((document.get("content") or "")[:512])
            if document.get("title"):
                if isinstance(document.get("title_emb"), (list, tuple, np.ndarray)):
                    precomputed_title += 1
                else:
                    title_indices.append(idx)
                    title_texts.append(document.get("title") or "")

        if content_texts:
            for idx, emb in zip(content_indices, self._encode_many_texts(content_texts)):
                documents[idx]["content_emb"] = emb

        if title_texts:
            for idx, emb in zip(title_indices, self._encode_many_texts(title_texts)):
                documents[idx]["title_emb"] = emb

        return {
            "candidate_docs": len(documents),
            "precomputed_content_emb_count": precomputed_content,
            "precomputed_title_emb_count": precomputed_title,
            "encoded_content_emb_count": len(content_indices),
            "encoded_title_emb_count": len(title_indices),
        }

    # ==================== 统计特征 ====================

    def _extract_statistical_features(self, query_terms, content_analysis, title_analysis):
        """文档和查询的统计特征"""
        features = {}

        features["query_length"] = len(query_terms)
        features["doc_length"] = content_analysis["word_count"]
        features["title_length"] = title_analysis["word_count"]

        if features["doc_length"] > 0:
            features["query_doc_length_ratio"] = features["query_length"] / features["doc_length"]
        else:
            features["query_doc_length_ratio"] = 0.0

        doc_terms = content_analysis["tokens"]
        if doc_terms:
            features["doc_vocab_diversity"] = len(content_analysis["token_set"]) / len(doc_terms)
        else:
            features["doc_vocab_diversity"] = 0.0

        term_freqs = content_analysis["term_freqs"]
        query_term_freqs = [term_freqs.get(term, 0) for term in query_terms]

        if query_term_freqs:
            features["avg_query_term_freq"] = sum(query_term_freqs) / len(query_term_freqs)
            features["max_query_term_freq"] = max(query_term_freqs)
        else:
            features["avg_query_term_freq"] = 0.0
            features["max_query_term_freq"] = 0.0

        return features

    # ==================== 查询词重要性特征 ====================

    def _extract_term_importance_features(self, query_terms, content_analysis):
        """基于 TF-IDF 的查询词重要性"""
        features = {}

        doc_terms = content_analysis["tokens"]
        term_freqs = content_analysis["term_freqs"]

        tfidf_scores = []
        for term in query_terms:
            tf = term_freqs.get(term, 0) / len(doc_terms) if doc_terms else 0
            idf = self._get_idf(term)
            tfidf_scores.append(tf * idf)

        if tfidf_scores:
            features["avg_tfidf_score"] = sum(tfidf_scores) / len(tfidf_scores)
            features["max_tfidf_score"] = max(tfidf_scores)
            features["sum_tfidf_score"] = sum(tfidf_scores)
        else:
            features["avg_tfidf_score"] = 0.0
            features["max_tfidf_score"] = 0.0
            features["sum_tfidf_score"] = 0.0

        return features

    def _get_idf(self, term):
        """获取词的 IDF 值。"""
        if term in self.idf_cache:
            return float(self.idf_cache[term])
        return self._default_idf

    # ==================== BM25 特征 ====================

    def _extract_bm25_features(self, query_terms, content_analysis, title_analysis):
        """BM25 算法相关特征"""
        features = {}

        k1 = 1.5
        b = 0.75
        avgdl = 500

        doc_terms = content_analysis["tokens"]
        doc_length = len(doc_terms)
        term_freqs = content_analysis["term_freqs"]

        bm25_scores = []
        for term in query_terms:
            tf = term_freqs.get(term, 0)
            idf = self._get_idf(term)

            numerator = tf * (k1 + 1)
            denominator = tf + k1 * (1 - b + b * (doc_length / avgdl))

            bm25_score = idf * (numerator / denominator) if denominator > 0 else 0
            bm25_scores.append(bm25_score)

        if bm25_scores:
            features["bm25_score"] = sum(bm25_scores)
            features["bm25_max"] = max(bm25_scores)
        else:
            features["bm25_score"] = 0.0
            features["bm25_max"] = 0.0

        title_term_freqs = title_analysis["term_freqs"]
        title_bm25 = 0.0

        for term in query_terms:
            if term in title_term_freqs:
                title_bm25 += self._get_idf(term) * 2.0

        features["bm25_title"] = title_bm25

        return features

    def get_feature_names(self):
        """返回所有特征的名称列表（用于模型训练）"""
        return [
            "exact_match_content",
            "exact_match_title",
            "term_coverage",
            "covered_term_count",
            "ordered_term_match",
            "fuzzy_match_score",
            "title_match_ratio",
            "first_sentence_match",
            "avg_term_position",
            "min_term_position",
            "term_density_first_100",
            "term_density_full",
            "semantic_sim_content",
            "semantic_sim_title",
            "title_content_consistency",
            "query_length",
            "doc_length",
            "title_length",
            "query_doc_length_ratio",
            "doc_vocab_diversity",
            "avg_query_term_freq",
            "max_query_term_freq",
            "avg_tfidf_score",
            "max_tfidf_score",
            "sum_tfidf_score",
            "bm25_score",
            "bm25_max",
            "bm25_title",
            "es_original_score",
            "es_log_score",
        ]
