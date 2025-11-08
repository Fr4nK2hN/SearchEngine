"""
Learning to Rank 排序模型
使用 LightGBM LambdaMART 算法
"""

import numpy as np
import pickle
from lightgbm import LGBMRanker
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns


class LTRRanker:
    """
    Learning to Rank 排序器
    
    使用 LightGBM 的 LambdaMART 算法训练排序模型
    支持特征标准化、模型评估、特征重要性分析
    """
    
    def __init__(self, feature_extractor):
        self.feature_extractor = feature_extractor
        self.model = None
        self.scaler = StandardScaler()
        self.feature_names = feature_extractor.get_feature_names()
        self.is_trained = False
        
    def train(self, training_data, validation_data=None, 
              n_estimators=200, learning_rate=0.05, max_depth=6):
        """
        训练 LTR 模型
        
        Args:
            training_data: 训练数据列表
            validation_data: 验证数据列表（可选）
            n_estimators: 树的数量
            learning_rate: 学习率
            max_depth: 树的最大深度
        """
        print("Preparing training data...")
        X_train, y_train, groups_train = self._prepare_data(training_data)
        
        print(f"Training data shape: {X_train.shape}")
        print(f"Number of queries: {len(groups_train)}")
        print(f"Label distribution: {np.bincount(y_train.astype(int))}")
        
        # 特征标准化
        X_train_scaled = self.scaler.fit_transform(X_train)
        
        # 初始化模型
        self.model = LGBMRanker(
            objective='lambdarank',
            metric='ndcg',
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=max_depth,
            num_leaves=31,
            min_child_samples=20,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=-1,
            importance_type='gain'
        )
        
        # 准备验证数据（如果有）
        eval_set = None
        eval_group = None
        if validation_data:
            X_val, y_val, groups_val = self._prepare_data(validation_data)
            X_val_scaled = self.scaler.transform(X_val)
            eval_set = [(X_val_scaled, y_val)]
            eval_group = [groups_val]
        
        # 训练模型
        print("Training LambdaMART model...")
        self.model.fit(
            X_train_scaled, 
            y_train,
            group=groups_train,
            eval_set=eval_set,
            eval_group=eval_group,
            eval_metric='ndcg',
            callbacks=[
                # 早停
                # callback.early_stopping(stopping_rounds=20, verbose=True)
            ] if validation_data else None
        )
        
        self.is_trained = True
        print("Training completed!")
        
        # 打印特征重要性
        self._print_feature_importance()
        
        return self
    
    def _prepare_data(self, training_data):
        """
        将训练数据转换为模型输入格式
        
        Returns:
            X: 特征矩阵 (n_samples, n_features)
            y: 标签向量 (n_samples,)
            groups: 每个查询的文档数量 (n_queries,)
        """
        X = []
        y = []
        groups = []
        
        for item in training_data:
            query = item['query']
            documents = item['documents']
            labels = item['relevance_labels']
            
            group_size = 0
            for doc, label in zip(documents, labels):
                # 提取特征
                features = self.feature_extractor.extract_all_features(
                    query, doc, es_score=doc.get('es_score')
                )
                
                # 转换为特征向量（按固定顺序）
                feature_vector = [features.get(name, 0.0) for name in self.feature_names]
                
                X.append(feature_vector)
                y.append(label)
                group_size += 1
            
            groups.append(group_size)
        
        return np.array(X), np.array(y), groups
    
    def predict(self, query, documents):
        """
        预测查询-文档对的相关性分数
        
        Args:
            query: 查询字符串
            documents: 文档列表
        
        Returns:
            scores: 预测的相关性分数数组
        """
        if not self.is_trained:
            raise ValueError("Model not trained yet. Call train() first.")
        
        X = []
        for doc in documents:
            features = self.feature_extractor.extract_all_features(
                query, doc, es_score=doc.get('es_score')
            )
            feature_vector = [features.get(name, 0.0) for name in self.feature_names]
            X.append(feature_vector)
        
        X = np.array(X)
        X_scaled = self.scaler.transform(X)
        
        scores = self.model.predict(X_scaled)
        return scores
    
    def rerank(self, query, search_results):
        """
        对搜索结果进行重排序
        
        Args:
            query: 查询字符串
            search_results: Elasticsearch 搜索结果
        
        Returns:
            重排序后的搜索结果
        """
        if not search_results:
            return []
        
        documents = [hit['_source'] for hit in search_results]
        
        # 预测分数
        scores = self.predict(query, documents)
        
        # 添加预测分数到结果中
        for i, (hit, score) in enumerate(zip(search_results, scores)):
            hit['_ltr_score'] = float(score)
            hit['_original_rank'] = i + 1
        
        # 按 LTR 分数排序
        search_results.sort(key=lambda x: x['_ltr_score'], reverse=True)
        
        return search_results
    
    def _print_feature_importance(self, top_k=15):
        """打印特征重要性"""
        if self.model is None:
            return
        
        importances = self.model.feature_importances_
        
        # 创建特征重要性字典
        feature_importance = {
            name: importance 
            for name, importance in zip(self.feature_names, importances)
        }
        
        # 排序
        sorted_features = sorted(
            feature_importance.items(), 
            key=lambda x: x[1], 
            reverse=True
        )
        
        print(f"\nTop {top_k} Most Important Features:")
        print("-" * 50)
        for i, (feature, importance) in enumerate(sorted_features[:top_k], 1):
            print(f"{i:2d}. {feature:30s}: {importance:8.2f}")
    
    def plot_feature_importance(self, top_k=20, save_path=None):
        """可视化特征重要性"""
        if self.model is None:
            print("Model not trained yet.")
            return
        
        importances = self.model.feature_importances_
        
        # 创建 DataFrame
        import pandas as pd
        feature_df = pd.DataFrame({
            'feature': self.feature_names,
            'importance': importances
        })
        
        # 排序并选择 top_k
        feature_df = feature_df.sort_values('importance', ascending=False).head(top_k)
        
        # 绘图
        plt.figure(figsize=(10, 8))
        sns.barplot(data=feature_df, y='feature', x='importance', palette='viridis')
        plt.title(f'Top {top_k} Feature Importances', fontsize=14, fontweight='bold')
        plt.xlabel('Importance', fontsize=12)
        plt.ylabel('Feature', fontsize=12)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Feature importance plot saved to {save_path}")
        
        plt.show()
    
    def save_model(self, filepath):
        """保存模型到文件"""
        if not self.is_trained:
            raise ValueError("Model not trained yet.")
        
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'feature_names': self.feature_names
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath):
        """从文件加载模型"""
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        self.model = model_data['model']
        self.scaler = model_data['scaler']
        self.feature_names = model_data['feature_names']
        self.is_trained = True
        
        print(f"Model loaded from {filepath}")
        return self


# 完整的训练流程示例
if __name__ == '__main__':
    from feature_extractor import FeatureExtractor
    from training_data_generator import TrainingDataGenerator
    from elasticsearch import Elasticsearch
    from sentence_transformers import CrossEncoder
    
    # 1. 初始化组件
    print("Initializing components...")
    es = Elasticsearch([{'host': 'elasticsearch', 'port': 9200, 'scheme': 'http'}])
    cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
    feature_extractor = FeatureExtractor()
    
    # 2. 生成训练数据
    print("\nGenerating training data...")
    generator = TrainingDataGenerator(es, cross_encoder)
    
    # 尝试加载已有数据，如果没有则生成新数据
    try:
        training_data = generator.load_training_data('data/ltr_training_data.json')
    except FileNotFoundError:
        queries = generator.generate_training_queries(num_queries=50)
        training_data = generator.generate_training_data(queries, docs_per_query=30)
        generator.save_training_data(training_data, 'data/ltr_training_data.json')
    
    # 3. 划分训练集和验证集
    split_idx = int(len(training_data) * 0.8)
    train_set = training_data[:split_idx]
    val_set = training_data[split_idx:]
    
    print(f"\nTraining set: {len(train_set)} queries")
    print(f"Validation set: {len(val_set)} queries")
    
    # 4. 训练模型
    print("\nTraining LTR model...")
    ranker = LTRRanker(feature_extractor)
    ranker.train(
        training_data=train_set,
        validation_data=val_set,
        n_estimators=200,
        learning_rate=0.05,
        max_depth=6
    )
    
    # 5. 保存模型
    ranker.save_model('models/ltr_model.pkl')
    
    # 6. 可视化特征重要性
    ranker.plot_feature_importance(top_k=20, save_path='models/feature_importance.png')
    
    print("\nTraining pipeline completed!")