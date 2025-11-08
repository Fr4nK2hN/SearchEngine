import re
import math
import numpy as np
from collections import Counter
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# 确保下载必要的 NLTK 数据
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

class FeatureExtractor:
    """
    高级特征提取器
    包含文本匹配、语义相似度、统计特征等多个维度
    """
    
    def __init__(self):
        self.semantic_model = SentenceTransformer('all-MiniLM-L6-v2')
        self.stop_words = set(stopwords.words('english'))
        self.idf_cache = {}
        
    def extract_all_features(self, query, document, es_score=None):
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
        
        # 预处理
        query_lower = query.lower()
        query_terms = self._tokenize(query)
        doc_content = document.get('content', '')
        doc_title = document.get('title', '')
        
        # ========== 1. 字面匹配特征 ==========
        features.update(self._extract_literal_features(
            query, query_lower, query_terms, doc_content, doc_title
        ))
        
        # ========== 2. 位置特征 ==========
        features.update(self._extract_position_features(
            query_terms, doc_content, doc_title
        ))
        
        # ========== 3. 语义特征 ==========
        features.update(self._extract_semantic_features(
            query, doc_content, doc_title
        ))
        
        # ========== 4. 统计特征 ==========
        features.update(self._extract_statistical_features(
            query_terms, doc_content, doc_title
        ))
        
        # ========== 5. 查询词重要性特征 ==========
        features.update(self._extract_term_importance_features(
            query_terms, doc_content
        ))
        
        # ========== 6. BM25 相关特征 ==========
        features.update(self._extract_bm25_features(
            query_terms, doc_content, doc_title
        ))
        
        # ========== 7. 原始分数特征 ==========
        if es_score is not None:
            features['es_original_score'] = float(es_score)
            features['es_log_score'] = math.log(es_score + 1)
        
        return features
    
    def _tokenize(self, text):
        """分词并去除停用词"""
        tokens = word_tokenize(text.lower())
        return [t for t in tokens if t.isalnum() and t not in self.stop_words]
    
    # ==================== 字面匹配特征 ====================
    
    def _extract_literal_features(self, query, query_lower, query_terms, 
                                   doc_content, doc_title):
        """字面匹配相关特征"""
        features = {}
        
        doc_content_lower = doc_content.lower()
        doc_title_lower = doc_title.lower()
        
        # 1. 完全匹配
        features['exact_match_content'] = 1.0 if query_lower in doc_content_lower else 0.0
        features['exact_match_title'] = 1.0 if query_lower in doc_title_lower else 0.0
        
        # 2. 查询词覆盖率
        doc_terms = set(self._tokenize(doc_content))
        query_term_set = set(query_terms)
        
        if query_term_set:
            covered = query_term_set & doc_terms
            features['term_coverage'] = len(covered) / len(query_term_set)
            features['covered_term_count'] = len(covered)
        else:
            features['term_coverage'] = 0.0
            features['covered_term_count'] = 0
        
        # 3. 子序列匹配（有序词匹配）
        features['ordered_term_match'] = self._ordered_term_match(
            query_terms, doc_content_lower
        )
        
        # 4. 近似匹配（编辑距离）
        features['fuzzy_match_score'] = self._fuzzy_match_score(
            query_lower, doc_content_lower[:500]  # 只检查前500字符
        )
        
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
        from difflib import SequenceMatcher
        return SequenceMatcher(None, query, doc_excerpt).ratio()
    
    # ==================== 位置特征 ====================
    
    def _extract_position_features(self, query_terms, doc_content, doc_title):
        """查询词在文档中的位置特征"""
        features = {}
        
        doc_content_lower = doc_content.lower()
        doc_title_lower = doc_title.lower()
        
        # 1. 标题匹配度
        title_terms = set(self._tokenize(doc_title))
        query_term_set = set(query_terms)
        
        if query_term_set:
            title_match = query_term_set & title_terms
            features['title_match_ratio'] = len(title_match) / len(query_term_set)
        else:
            features['title_match_ratio'] = 0.0
        
        # 2. 首句匹配
        first_sentence = doc_content_lower.split('.')[0] if doc_content else ''
        first_sentence_terms = set(self._tokenize(first_sentence))
        
        if query_term_set:
            first_match = query_term_set & first_sentence_terms
            features['first_sentence_match'] = len(first_match) / len(query_term_set)
        else:
            features['first_sentence_match'] = 0.0
        
        # 3. 查询词首次出现位置（归一化）
        positions = []
        for term in query_terms:
            pos = doc_content_lower.find(term)
            if pos >= 0:
                positions.append(pos)
        
        if positions:
            doc_length = len(doc_content)
            avg_position = sum(positions) / len(positions)
            features['avg_term_position'] = avg_position / doc_length if doc_length > 0 else 1.0
            features['min_term_position'] = min(positions) / doc_length if doc_length > 0 else 1.0
        else:
            features['avg_term_position'] = 1.0
            features['min_term_position'] = 1.0
        
        # 4. 查询词密度（在文档不同区域）
        features['term_density_first_100'] = self._term_density(
            query_terms, doc_content_lower[:100]
        )
        features['term_density_full'] = self._term_density(
            query_terms, doc_content_lower
        )
        
        return features
    
    def _term_density(self, query_terms, text):
        """计算查询词在文本中的密度"""
        if not text:
            return 0.0
        
        text_terms = self._tokenize(text)
        if not text_terms:
            return 0.0
        
        matches = sum(1 for term in text_terms if term in query_terms)
        return matches / len(text_terms)
    
    # ==================== 语义特征 ====================
    
    def _extract_semantic_features(self, query, doc_content, doc_title):
        """基于语义嵌入的特征"""
        features = {}
        
        try:
            # 1. 查询-内容语义相似度
            query_emb = self.semantic_model.encode(query)
            content_emb = self.semantic_model.encode(doc_content[:512])  # 限制长度
            
            features['semantic_sim_content'] = float(
                cosine_similarity([query_emb], [content_emb])[0][0]
            )
            
            # 2. 查询-标题语义相似度
            if doc_title:
                title_emb = self.semantic_model.encode(doc_title)
                features['semantic_sim_title'] = float(
                    cosine_similarity([query_emb], [title_emb])[0][0]
                )
            else:
                features['semantic_sim_title'] = 0.0
            
            # 3. 标题-内容一致性
            if doc_title:
                features['title_content_consistency'] = float(
                    cosine_similarity([title_emb], [content_emb])[0][0]
                )
            else:
                features['title_content_consistency'] = 0.0
                
        except Exception as e:
            print(f"Error in semantic features: {e}")
            features['semantic_sim_content'] = 0.0
            features['semantic_sim_title'] = 0.0
            features['title_content_consistency'] = 0.0
        
        return features
    
    # ==================== 统计特征 ====================
    
    def _extract_statistical_features(self, query_terms, doc_content, doc_title):
        """文档和查询的统计特征"""
        features = {}
        
        # 1. 长度特征
        features['query_length'] = len(query_terms)
        features['doc_length'] = len(doc_content.split())
        features['title_length'] = len(doc_title.split())
        
        # 2. 长度比率
        if features['doc_length'] > 0:
            features['query_doc_length_ratio'] = features['query_length'] / features['doc_length']
        else:
            features['query_doc_length_ratio'] = 0.0
        
        # 3. 词汇多样性
        doc_terms = self._tokenize(doc_content)
        if doc_terms:
            unique_terms = set(doc_terms)
            features['doc_vocab_diversity'] = len(unique_terms) / len(doc_terms)
        else:
            features['doc_vocab_diversity'] = 0.0
        
        # 4. 查询词频率统计
        term_freqs = Counter(doc_terms)
        query_term_freqs = [term_freqs.get(term, 0) for term in query_terms]
        
        if query_term_freqs:
            features['avg_query_term_freq'] = sum(query_term_freqs) / len(query_term_freqs)
            features['max_query_term_freq'] = max(query_term_freqs)
        else:
            features['avg_query_term_freq'] = 0.0
            features['max_query_term_freq'] = 0.0
        
        return features
    
    # ==================== 查询词重要性特征 ====================
    
    def _extract_term_importance_features(self, query_terms, doc_content):
        """基于 TF-IDF 的查询词重要性"""
        features = {}
        
        # 简化的 IDF 计算（假设文档集合）
        # 在实际应用中，应该预先计算整个语料库的 IDF
        doc_terms = self._tokenize(doc_content)
        term_freqs = Counter(doc_terms)
        
        # 计算查询词的 TF-IDF 得分
        tfidf_scores = []
        for term in query_terms:
            tf = term_freqs.get(term, 0) / len(doc_terms) if doc_terms else 0
            # 简化的 IDF（实际应该从语料库统计）
            idf = self._get_idf(term)
            tfidf_scores.append(tf * idf)
        
        if tfidf_scores:
            features['avg_tfidf_score'] = sum(tfidf_scores) / len(tfidf_scores)
            features['max_tfidf_score'] = max(tfidf_scores)
            features['sum_tfidf_score'] = sum(tfidf_scores)
        else:
            features['avg_tfidf_score'] = 0.0
            features['max_tfidf_score'] = 0.0
            features['sum_tfidf_score'] = 0.0
        
        return features
    
    def _get_idf(self, term):
        """获取词的 IDF 值（简化版本）"""
        # 在实际应用中，应该从预计算的 IDF 字典中获取
        if term in self.idf_cache:
            return self.idf_cache[term]
        
        # 默认 IDF 值
        default_idf = 3.0
        self.idf_cache[term] = default_idf
        return default_idf
    
    # ==================== BM25 特征 ====================
    
    def _extract_bm25_features(self, query_terms, doc_content, doc_title):
        """BM25 算法相关特征"""
        features = {}
        
        # BM25 参数
        k1 = 1.5
        b = 0.75
        avgdl = 500  # 假设平均文档长度
        
        doc_terms = self._tokenize(doc_content)
        doc_length = len(doc_terms)
        term_freqs = Counter(doc_terms)
        
        bm25_scores = []
        for term in query_terms:
            tf = term_freqs.get(term, 0)
            idf = self._get_idf(term)
            
            # BM25 公式
            numerator = tf * (k1 + 1)
            denominator = tf + k1 * (1 - b + b * (doc_length / avgdl))
            
            bm25_score = idf * (numerator / denominator) if denominator > 0 else 0
            bm25_scores.append(bm25_score)
        
        if bm25_scores:
            features['bm25_score'] = sum(bm25_scores)
            features['bm25_max'] = max(bm25_scores)
        else:
            features['bm25_score'] = 0.0
            features['bm25_max'] = 0.0
        
        # 标题的 BM25 分数
        title_terms = self._tokenize(doc_title)
        title_term_freqs = Counter(title_terms)
        title_bm25 = 0.0
        
        for term in query_terms:
            if term in title_term_freqs:
                title_bm25 += self._get_idf(term) * 2.0  # 标题权重更高
        
        features['bm25_title'] = title_bm25
        
        return features
    
    def get_feature_names(self):
        """返回所有特征的名称列表（用于模型训练）"""
        # 这个列表需要与 extract_all_features 返回的特征保持一致
        return [
            # 字面匹配
            'exact_match_content', 'exact_match_title', 'term_coverage',
            'covered_term_count', 'ordered_term_match', 'fuzzy_match_score',
            
            # 位置特征
            'title_match_ratio', 'first_sentence_match', 'avg_term_position',
            'min_term_position', 'term_density_first_100', 'term_density_full',
            
            # 语义特征
            'semantic_sim_content', 'semantic_sim_title', 'title_content_consistency',
            
            # 统计特征
            'query_length', 'doc_length', 'title_length', 'query_doc_length_ratio',
            'doc_vocab_diversity', 'avg_query_term_freq', 'max_query_term_freq',
            
            # 查询词重要性
            'avg_tfidf_score', 'max_tfidf_score', 'sum_tfidf_score',
            
            # BM25
            'bm25_score', 'bm25_max', 'bm25_title',
            
            # 原始分数
            'es_original_score', 'es_log_score'
        ]