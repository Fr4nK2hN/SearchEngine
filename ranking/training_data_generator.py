import json
import os
import re
from elasticsearch import Elasticsearch
from sentence_transformers import CrossEncoder

from retrieval import DEFAULT_RECALL_RELAX_THRESHOLD, search_documents_with_fallback


class TrainingDataGenerator:
    """
    生成 Learning to Rank 训练数据
    
    策略：
    1. 使用 Cross-Encoder 作为"教师模型"生成伪标签
    2. 结合启发式规则确保标签质量
    3. 生成多样化的查询-文档对
    """
    
    def __init__(self, es_client, cross_encoder_model):
        self.es = es_client
        self.cross_encoder = cross_encoder_model
        try:
            self.recall_relax_threshold = max(
                1, int(os.getenv("RECALL_RELAX_THRESHOLD", str(DEFAULT_RECALL_RELAX_THRESHOLD)))
            )
        except ValueError:
            self.recall_relax_threshold = DEFAULT_RECALL_RELAX_THRESHOLD

        self._stopwords = {
            "a", "an", "and", "are", "as", "at", "be", "by", "for", "from",
            "how", "in", "is", "it", "of", "on", "or", "that", "the", "to",
            "was", "were", "what", "when", "where", "which", "who", "why",
            "will", "with", "you", "your",
        }
        self._query_token_pattern = re.compile(r"[a-z0-9]+")

    def _normalize_query(self, query):
        return " ".join((query or "").strip().split())

    def _query_tokens(self, query):
        text = (query or "").lower()
        return self._query_token_pattern.findall(text)

    def _stopword_ratio(self, tokens):
        if not tokens:
            return 1.0
        stop_count = sum(1 for t in tokens if t in self._stopwords)
        return stop_count / len(tokens)

    def _is_high_quality_query(self, query, min_terms=3, min_chars=8, max_stopword_ratio=0.85):
        normalized = self._normalize_query(query)
        tokens = self._query_tokens(normalized)
        if len(normalized) < min_chars:
            return False
        if len(tokens) < min_terms:
            return False
        if len(set(tokens)) <= 1:
            return False
        if self._stopword_ratio(tokens) > max_stopword_ratio:
            return False
        return True
        
    def generate_training_queries(
        self,
        num_queries=100,
        min_terms=3,
        min_chars=8,
        max_stopword_ratio=0.85,
        use_prefix_queries=False,
    ):
        """
        从文档集合中生成训练查询
        
        策略：
        1. 优先使用完整标题和 related_queries，减少残缺前缀查询
        2. 基于长度、词数、停用词比例过滤低质量查询
        3. 可选地补充标题前缀查询（默认关闭）
        """
        queries = []
        fallback_queries = []
        
        # 从 Elasticsearch 获取候选文档
        response = self.es.search(
            index="documents",
            body={
                "query": {"match_all": {}},
                "size": max(100, num_queries * 4)
            }
        )
        
        documents = response['hits']['hits']

        for doc in documents:
            source = doc['_source']

            candidates = []

            title = self._normalize_query(source.get('title', ''))
            if title:
                candidates.append(title)
                if use_prefix_queries:
                    title_words = title.split()
                    if len(title_words) >= 4:
                        candidates.append(' '.join(title_words[:4]))

            related = source.get('related_queries', []) or []
            for q in related[:3]:
                normalized = self._normalize_query(q)
                if normalized:
                    candidates.append(normalized)

            keywords = source.get('keywords', []) or []
            if len(keywords) >= 3:
                keyword_query = self._normalize_query(" ".join(keywords[:4]))
                if keyword_query:
                    candidates.append(keyword_query)

            for candidate in candidates:
                fallback_queries.append(candidate)
                if self._is_high_quality_query(
                    candidate,
                    min_terms=min_terms,
                    min_chars=min_chars,
                    max_stopword_ratio=max_stopword_ratio,
                ):
                    queries.append(candidate)

            if len(queries) >= num_queries:
                break

        # 去重且保持顺序
        def dedupe_keep_order(items):
            seen = set()
            out = []
            for item in items:
                if item in seen:
                    continue
                seen.add(item)
                out.append(item)
            return out

        queries = dedupe_keep_order(queries)
        if len(queries) < num_queries:
            for candidate in dedupe_keep_order(fallback_queries):
                if candidate not in queries:
                    queries.append(candidate)
                if len(queries) >= num_queries:
                    break

        return queries[:num_queries]
    
    def generate_training_data(
        self,
        queries,
        docs_per_query=50,
        apply_heuristics=True,
        drop_all_zero_queries=True,
        min_relevant_docs=1,
    ):
        """
        为每个查询生成训练数据
        
        Returns:
            List[Dict]: 训练数据，每个元素包含：
                - query: 查询字符串
                - documents: 文档列表
                - relevance_labels: 相关性标签 (0-4)
        """
        training_data = []
        
        for query in queries:
            query = self._normalize_query(query)
            if not query:
                continue
            print(f"Processing query: {query}")
            
            # 从 Elasticsearch 检索文档
            es_results = self._search_documents(query, size=docs_per_query)
            
            if len(es_results) < 5:
                continue  # 跳过结果太少的查询
            
            # 使用 Cross-Encoder 生成伪标签
            documents = []
            pairs = []
            for result in es_results:
                doc = result['_source']
                content = doc.get('content', '')
                pairs.append([query, content])
                documents.append({
                    'id': result['_id'],
                    'title': doc.get('title', ''),
                    'content': content,
                    'related_queries': doc.get('related_queries', []),
                    'es_score': result['_score']
                })

            # Cross-Encoder 批量打分
            relevance_scores = [float(s) for s in self.cross_encoder.predict(pairs)]
            
            # 将连续分数转换为离散标签 (0-4)
            relevance_labels = self._score_to_label(relevance_scores)
            
            if apply_heuristics:
                relevance_labels = self._apply_heuristic_rules(
                    query, documents, relevance_labels
                )

            relevant_count = sum(1 for x in relevance_labels if x > 0)
            if drop_all_zero_queries and relevant_count == 0:
                continue
            if relevant_count < max(0, int(min_relevant_docs)):
                continue
            
            training_data.append({
                'query': query,
                'documents': documents,
                'relevance_labels': relevance_labels
            })
        
        return training_data
    
    def _search_documents(self, query, size=50):
        """使用 Elasticsearch 检索文档"""
        try:
            results, _ = search_documents_with_fallback(
                self.es,
                query,
                size=size,
                hl=False,
                relax_threshold=self.recall_relax_threshold,
                index_name="documents",
            )
            return results
        except Exception as e:
            print(f"Search error for query '{query}': {e}")
            return []
    
    def _score_to_label(self, scores):
        """
        将连续的相关性分数转换为离散标签
        
        分数分布：
        - 4 (Perfect): score > 0.7
        - 3 (Excellent): 0.5 < score <= 0.7
        - 2 (Good): 0.3 < score <= 0.5
        - 1 (Fair): 0.1 < score <= 0.3
        - 0 (Bad): score <= 0.1
        """
        labels = []
        for score in scores:
            if score > 0.7:
                label = 4
            elif score > 0.5:
                label = 3
            elif score > 0.3:
                label = 2
            elif score > 0.1:
                label = 1
            else:
                label = 0
            labels.append(label)
        
        return labels
    
    def _apply_heuristic_rules(self, query, documents, labels):
        """
        应用启发式规则调整标签质量
        
        规则：
        1. 如果标题完全匹配查询，标签至少为 3
        2. 如果文档太短（<50词），降低标签
        3. 如果查询词覆盖率很低，降低标签
        """
        adjusted_labels = []
        query_lower = query.lower()
        query_terms = set(query_lower.split())
        
        for i, (doc, label) in enumerate(zip(documents, labels)):
            adjusted_label = label
            
            # 规则 1: 标题匹配
            title = doc.get('title', '').lower()
            if query_lower in title or title in query_lower:
                adjusted_label = max(adjusted_label, 3)
            
            # 规则 2: 文档长度
            content = doc.get('content', '')
            word_count = len(content.split())
            if word_count < 50:
                adjusted_label = max(0, adjusted_label - 1)
            
            # 规则 3: 查询词覆盖率
            doc_terms = set(content.lower().split())
            coverage = len(query_terms & doc_terms) / len(query_terms) if query_terms else 0
            
            if coverage < 0.3:
                adjusted_label = min(adjusted_label, 1)
            elif coverage > 0.8:
                adjusted_label = max(adjusted_label, 2)
            
            adjusted_labels.append(adjusted_label)
        
        return adjusted_labels
    
    def save_training_data(self, training_data, filepath):
        """保存训练数据到 JSON 文件"""
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(training_data, f, indent=2, ensure_ascii=False)
        
        print(f"Training data saved to {filepath}")
        print(f"Total queries: {len(training_data)}")
        
        # 统计标签分布
        all_labels = []
        for item in training_data:
            all_labels.extend(item['relevance_labels'])
        
        from collections import Counter
        label_dist = Counter(all_labels)
        print(f"Label distribution: {dict(label_dist)}")
    
    def load_training_data(self, filepath):
        """从 JSON 文件加载训练数据"""
        with open(filepath, 'r', encoding='utf-8') as f:
            training_data = json.load(f)
        
        print(f"Loaded {len(training_data)} training queries")
        return training_data


# 使用示例
if __name__ == '__main__':
    from elasticsearch import Elasticsearch
    from sentence_transformers import CrossEncoder
    
    # 初始化
    es = Elasticsearch([{'host': 'elasticsearch', 'port': 9200, 'scheme': 'http'}])
    cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
    
    generator = TrainingDataGenerator(es, cross_encoder)
    
    # 生成训练查询
    queries = generator.generate_training_queries(num_queries=50)
    
    # 生成训练数据
    training_data = generator.generate_training_data(queries, docs_per_query=30)
    
    # 保存
    generator.save_training_data(training_data, 'data/ltr_training_data.json')
