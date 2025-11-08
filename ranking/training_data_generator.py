import json
import random
from elasticsearch import Elasticsearch
from sentence_transformers import CrossEncoder


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
        
    def generate_training_queries(self, num_queries=100):
        """
        从文档集合中生成训练查询
        
        策略：
        1. 从文档标题提取关键短语
        2. 从文档内容提取问题型查询
        3. 使用 related_queries 字段
        """
        queries = []
        
        # 从 Elasticsearch 获取随机文档
        response = self.es.search(
            index="documents",
            body={
                "query": {"match_all": {}},
                "size": 100
            }
        )
        
        documents = response['hits']['hits']
        
        for doc in documents[:num_queries]:
            source = doc['_source']
            
            # 策略 1: 使用标题的部分作为查询
            title = source.get('title', '')
            if title:
                title_words = title.split()
                if len(title_words) >= 3:
                    # 取标题的前几个词
                    query = ' '.join(title_words[:random.randint(2, min(5, len(title_words)))])
                    queries.append(query)
            
            # 策略 2: 使用 related_queries
            related = source.get('related_queries', [])
            if related:
                queries.extend(related[:2])
            
            # 策略 3: 从内容中提取关键短语
            content = source.get('content', '')
            if content:
                # 简单策略：提取前几个实体词
                words = content.split()[:50]
                important_words = [w for w in words if len(w) > 4 and w[0].isupper()]
                if len(important_words) >= 2:
                    query = ' '.join(important_words[:3])
                    queries.append(query)
        
        # 去重
        queries = list(set(queries))
        return queries[:num_queries]
    
    def generate_training_data(self, queries, docs_per_query=50):
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
            print(f"Processing query: {query}")
            
            # 从 Elasticsearch 检索文档
            es_results = self._search_documents(query, size=docs_per_query)
            
            if len(es_results) < 5:
                continue  # 跳过结果太少的查询
            
            # 使用 Cross-Encoder 生成伪标签
            documents = []
            relevance_scores = []
            
            for result in es_results:
                doc = result['_source']
                content = doc.get('content', '')
                
                # Cross-Encoder 打分
                score = self.cross_encoder.predict([[query, content]])[0]
                
                documents.append({
                    'id': result['_id'],
                    'title': doc.get('title', ''),
                    'content': content,
                    'related_queries': doc.get('related_queries', []),
                    'es_score': result['_score']
                })
                
                relevance_scores.append(float(score))
            
            # 将连续分数转换为离散标签 (0-4)
            relevance_labels = self._score_to_label(relevance_scores)
            
            # 应用启发式规则调整标签
            relevance_labels = self._apply_heuristic_rules(
                query, documents, relevance_labels
            )
            
            training_data.append({
                'query': query,
                'documents': documents,
                'relevance_labels': relevance_labels
            })
        
        return training_data
    
    def _search_documents(self, query, size=50):
        """使用 Elasticsearch 检索文档"""
        try:
            response = self.es.search(
                index="documents",
                body={
                    "query": {
                        "multi_match": {
                            "query": query,
                            "fields": ["title^3", "content^2", "combined_text"],
                            "type": "best_fields",
                            "fuzziness": "AUTO"
                        }
                    },
                    "size": size
                }
            )
            return response['hits']['hits']
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