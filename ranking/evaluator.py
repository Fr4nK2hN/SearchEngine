"""
排序模型评估模块
实现 NDCG, MAP, MRR, Precision@K 等指标
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict


class RankingEvaluator:
    """
    排序模型评估器
    支持多种评估指标和可视化
    """
    
    def __init__(self):
        self.metrics_history = defaultdict(list)
    
    # ==================== NDCG ====================
    
    def ndcg_at_k(self, relevance_labels, predicted_ranks, k=10):
        """
        计算 NDCG@K (Normalized Discounted Cumulative Gain)
        
        Args:
            relevance_labels: 真实相关性标签列表
            predicted_ranks: 预测排序后的索引列表
            k: 截断位置
        
        Returns:
            float: NDCG@K 分数
        """
        if not relevance_labels or predicted_ranks.size == 0:
            return 0.0
        
        # DCG@K
        dcg = self._dcg_at_k(relevance_labels, predicted_ranks, k)
        
        # IDCG@K (理想情况下的 DCG)
        ideal_ranks = np.argsort(relevance_labels)[::-1]
        idcg = self._dcg_at_k(relevance_labels, ideal_ranks, k)
        
        if idcg == 0:
            return 0.0
        
        return dcg / idcg
    
    def _dcg_at_k(self, relevance_labels, ranks, k):
        """计算 DCG@K"""
        dcg = 0.0
        for i, rank in enumerate(ranks[:k]):
            if rank < len(relevance_labels):
                rel = relevance_labels[rank]
                # DCG 公式: sum(rel / log2(rank + 2))
                dcg += (2 ** rel - 1) / np.log2(i + 2)
        
        return dcg
    
    # ==================== MAP ====================
    
    def map_score(self, relevance_labels, predicted_ranks):
        """
        计算 MAP (Mean Average Precision)
        
        MAP = mean(AP for each query)
        AP = mean(Precision@k for relevant docs)
        """
        if not relevance_labels or predicted_ranks.size == 0:
            return 0.0
        
        relevant_positions = []
        num_relevant = 0
        
        for i, rank in enumerate(predicted_ranks):
            if rank < len(relevance_labels):
                if relevance_labels[rank] > 0:  # 相关文档
                    num_relevant += 1
                    # Precision at position i+1
                    precision = num_relevant / (i + 1)
                    relevant_positions.append(precision)
        
        if not relevant_positions:
            return 0.0
        
        return sum(relevant_positions) / len(relevant_positions)
    
    # ==================== MRR ====================
    
    def mrr_score(self, relevance_labels, predicted_ranks):
        """
        计算 MRR (Mean Reciprocal Rank)
        
        MRR = 1 / rank_of_first_relevant_doc
        """
        if not relevance_labels or predicted_ranks.size == 0:
            return 0.0
        
        for i, rank in enumerate(predicted_ranks):
            if rank < len(relevance_labels):
                if relevance_labels[rank] > 0:  # 找到第一个相关文档
                    return 1.0 / (i + 1)
        
        return 0.0
    
    # ==================== Precision & Recall ====================
    
    def precision_at_k(self, relevance_labels, predicted_ranks, k=10):
        """
        计算 Precision@K
        
        Precision@K = (相关文档数) / K
        """
        if not relevance_labels or predicted_ranks.size == 0:
            return 0.0
        
        relevant_count = 0
        for i, rank in enumerate(predicted_ranks[:k]):
            if rank < len(relevance_labels):
                if relevance_labels[rank] > 0:
                    relevant_count += 1
        
        return relevant_count / k
    
    def recall_at_k(self, relevance_labels, predicted_ranks, k=10):
        """
        计算 Recall@K
        
        Recall@K = (检索到的相关文档数) / (总相关文档数)
        """
        if not relevance_labels or predicted_ranks.size == 0:
            return 0.0
        
        total_relevant = sum(1 for label in relevance_labels if label > 0)
        
        if total_relevant == 0:
            return 0.0
        
        relevant_count = 0
        for i, rank in enumerate(predicted_ranks[:k]):
            if rank < len(relevance_labels):
                if relevance_labels[rank] > 0:
                    relevant_count += 1
        
        return relevant_count / total_relevant
    
    # ==================== 批量评估 ====================
    
    def evaluate_ranker(self, test_data, ranker, k_values=[1, 3, 5, 10]):
        """
        在测试集上评估排序器
        
        Args:
            test_data: 测试数据列表
            ranker: LTR 排序器对象
            k_values: 要计算的 K 值列表
        
        Returns:
            dict: 评估结果
        """
        results = {
            'ndcg': {k: [] for k in k_values},
            'map': [],
            'mrr': [],
            'precision': {k: [] for k in k_values},
            'recall': {k: [] for k in k_values}
        }
        
        print(f"Evaluating on {len(test_data)} queries...")
        
        for i, item in enumerate(test_data):
            if (i + 1) % 10 == 0:
                print(f"Processed {i + 1}/{len(test_data)} queries...")
            
            query = item['query']
            documents = item['documents']
            relevance_labels = item['relevance_labels']
            
            # 使用排序器预测
            scores = ranker.predict(query, documents)
            
            # 获取排序后的索引
            predicted_ranks = np.argsort(scores)[::-1]  # 降序排列
            
            # 计算各项指标
            for k in k_values:
                ndcg = self.ndcg_at_k(relevance_labels, predicted_ranks, k)
                results['ndcg'][k].append(ndcg)
                
                precision = self.precision_at_k(relevance_labels, predicted_ranks, k)
                results['precision'][k].append(precision)
                
                recall = self.recall_at_k(relevance_labels, predicted_ranks, k)
                results['recall'][k].append(recall)
            
            # MAP 和 MRR（不依赖 k）
            map_score = self.map_score(relevance_labels, predicted_ranks)
            results['map'].append(map_score)
            
            mrr = self.mrr_score(relevance_labels, predicted_ranks)
            results['mrr'].append(mrr)
        
        # 计算平均值
        avg_results = {
            'ndcg': {k: np.mean(results['ndcg'][k]) for k in k_values},
            'map': np.mean(results['map']),
            'mrr': np.mean(results['mrr']),
            'precision': {k: np.mean(results['precision'][k]) for k in k_values},
            'recall': {k: np.mean(results['recall'][k]) for k in k_values}
        }
        
        return avg_results, results
    
    def compare_rankers(self, test_data, rankers_dict, k_values=[1, 3, 5, 10]):
        """
        比较多个排序器的性能
        
        Args:
            test_data: 测试数据
            rankers_dict: {'Ranker_Name': ranker_object} 字典
            k_values: K 值列表
        
        Returns:
            dict: 对比结果
        """
        comparison = {}
        
        for name, ranker in rankers_dict.items():
            print(f"\nEvaluating {name}...")
            avg_results, _ = self.evaluate_ranker(test_data, ranker, k_values)
            comparison[name] = avg_results
        
        return comparison
    
    # ==================== 可视化 ====================
    
    def plot_comparison(self, comparison_results, save_path=None):
        """
        可视化对比结果
        
        Args:
            comparison_results: compare_rankers 的输出
            save_path: 保存路径
        """
        ranker_names = list(comparison_results.keys())
        
        # 准备数据
        metrics_data = defaultdict(lambda: defaultdict(list))
        
        for ranker_name, results in comparison_results.items():
            # NDCG@K
            for k, value in results['ndcg'].items():
                metrics_data['NDCG'][f'@{k}'].append(value)
            
            # Precision@K
            for k, value in results['precision'].items():
                metrics_data['Precision'][f'@{k}'].append(value)
            
            # MAP & MRR
            metrics_data['Other']['MAP'].append(results['map'])
            metrics_data['Other']['MRR'].append(results['mrr'])
        
        # 创建子图
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        
        # 1. NDCG@K
        self._plot_metric_group(
            axes[0], metrics_data['NDCG'], ranker_names, 'NDCG@K'
        )
        
        # 2. Precision@K
        self._plot_metric_group(
            axes[1], metrics_data['Precision'], ranker_names, 'Precision@K'
        )
        
        # 3. MAP & MRR
        self._plot_metric_group(
            axes[2], metrics_data['Other'], ranker_names, 'MAP & MRR'
        )
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Comparison plot saved to {save_path}")
        
        plt.show()
    
    def _plot_metric_group(self, ax, metric_data, ranker_names, title):
        """绘制一组指标"""
        x = np.arange(len(metric_data))
        width = 0.8 / len(ranker_names)
        
        for i, ranker_name in enumerate(ranker_names):
            values = [metric_data[metric][i] for metric in metric_data]
            ax.bar(x + i * width, values, width, label=ranker_name, alpha=0.8)
        
        ax.set_ylabel('Score', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xticks(x + width * (len(ranker_names) - 1) / 2)
        ax.set_xticklabels(list(metric_data.keys()))
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
        ax.set_ylim([0, 1.0])
    
    def print_results(self, results):
        """格式化打印评估结果"""
        print("\n" + "="*60)
        print("EVALUATION RESULTS")
        print("="*60)
        
        print("\nNDCG@K:")
        for k, score in results['ndcg'].items():
            print(f"  NDCG@{k:2d}: {score:.4f}")
        
        print(f"\nMAP:     {results['map']:.4f}")
        print(f"MRR:     {results['mrr']:.4f}")
        
        print("\nPrecision@K:")
        for k, score in results['precision'].items():
            print(f"  P@{k:2d}:    {score:.4f}")
        
        print("\nRecall@K:")
        for k, score in results['recall'].items():
            print(f"  R@{k:2d}:    {score:.4f}")
        
        print("="*60 + "\n")


# 使用示例
if __name__ == '__main__':
    from feature_extractor import FeatureExtractor
    from ranker import LTRRanker
    from training_data_generator import TrainingDataGenerator
    from elasticsearch import Elasticsearch
    from sentence_transformers import CrossEncoder
    
    # 加载数据和模型
    generator = TrainingDataGenerator(None, None)
    test_data = generator.load_training_data('data/ltr_training_data.json')
    
    # 划分测试集
    test_set = test_data[-10:]  # 最后10个查询作为测试
    
    # 加载训练好的模型
    feature_extractor = FeatureExtractor()
    ranker = LTRRanker(feature_extractor)
    ranker.load_model('models/ltr_model.pkl')
    
    # 评估
    evaluator = RankingEvaluator()
    avg_results, full_results = evaluator.evaluate_ranker(
        test_set, 
        ranker, 
        k_values=[1, 3, 5, 10]
    )
    
    # 打印结果
    evaluator.print_results(avg_results)