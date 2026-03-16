import os
import sys
import argparse
import json

# 添加项目根目录到 Python 路径
sys.path.insert(
    0,
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
)

print("正在导入依赖...")

try:
    from elasticsearch import Elasticsearch
    print("✓ Elasticsearch 已导入")
except ImportError:
    print("✗ 请安装: pip install elasticsearch")
    sys.exit(1)

try:
    from sentence_transformers import CrossEncoder
    print("✓ SentenceTransformers 已导入")
except ImportError:
    print("✗ 请安装: pip install sentence-transformers")
    sys.exit(1)

try:
    import lightgbm
    print("✓ LightGBM 已导入")
except ImportError:
    print("✗ 请安装: pip install lightgbm")
    sys.exit(1)

# 导入自定义模块
try:
    from ranking.feature_extractor import FeatureExtractor
    from ranking.training_data_generator import TrainingDataGenerator
    from ranking.ranker import LTRRanker
    from ranking.evaluator import RankingEvaluator
    print("✓ 自定义模块已导入")
except ImportError as e:
    print(f"✗ 导入自定义模块失败: {e}")
    print("\n请确保:")
    print("1. ranking/ 目录下有 __init__.py 文件")
    print("2. 所有 Python 文件都在 ranking/ 目录中")
    sys.exit(1)


def main(args):
    """主训练流程"""
    
    print("\n" + "="*70)
    print("🚀 Learning to Rank Model Training Pipeline")
    print("="*70)
    
    # ========== 1. 初始化组件 ==========
    print("\n[Step 1/6] 初始化组件...")
    
    es = Elasticsearch([{
        'host': args.es_host,
        'port': args.es_port,
        'scheme': 'http'
    }])
    
    # 测试连接
    try:
        if not es.ping():
            print("✗ 无法连接到 Elasticsearch")
            print(f"  请确保 Elasticsearch 运行在 {args.es_host}:{args.es_port}")
            sys.exit(1)
        print(f"✓ 已连接到 Elasticsearch ({args.es_host}:{args.es_port})")
    except Exception as e:
        print(f"✗ Elasticsearch 连接错误: {e}")
        sys.exit(1)
    
    print("正在加载 Cross-Encoder 模型（首次运行会下载，请稍候）...")
    try:
        cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
        print("✓ Cross-Encoder 模型已加载")
    except Exception as e:
        print(f"✗ 加载 Cross-Encoder 失败: {e}")
        sys.exit(1)
    
    feature_extractor = FeatureExtractor()
    print(f"✓ 特征提取器已初始化 ({len(feature_extractor.get_feature_names())} 个特征)")
    
    # ========== 2. 生成/加载训练数据 ==========
    print(f"\n[Step 2/6] 准备训练数据...")
    
    generator = TrainingDataGenerator(es, cross_encoder)
    training_data_path = args.training_data
    
    if os.path.exists(training_data_path) and not args.regenerate_data:
        print(f"从 {training_data_path} 加载已有训练数据...")
        try:
            training_data = generator.load_training_data(training_data_path)
        except Exception as e:
            print(f"✗ 加载失败: {e}")
            print("将重新生成训练数据...")
            training_data = None
    else:
        training_data = None
    
    if training_data is None:
        print(f"生成新的训练数据...")
        print(f"  - 查询数量: {args.num_queries}")
        print(f"  - 每个查询的文档数: {args.docs_per_query}")
        
        try:
            # 生成查询
            queries = generator.generate_training_queries(num_queries=args.num_queries)
            print(f"✓ 生成了 {len(queries)} 个训练查询")
            
            # 生成训练数据
            training_data = generator.generate_training_data(
                queries, 
                docs_per_query=args.docs_per_query
            )
            
            # 保存
            os.makedirs(os.path.dirname(training_data_path), exist_ok=True)
            generator.save_training_data(training_data, training_data_path)
            print(f"✓ 训练数据已保存到 {training_data_path}")
        except Exception as e:
            print(f"✗ 生成训练数据失败: {e}")
            import traceback
            traceback.print_exc()
            sys.exit(1)
    
    # ========== 3. 划分数据集 ==========
    print(f"\n[Step 3/6] 划分数据集...")
    
    # 80% 训练，10% 验证，10% 测试
    n_total = len(training_data)
    n_train = int(n_total * 0.8)
    n_val = int(n_total * 0.1)
    
    train_set = training_data[:n_train]
    val_set = training_data[n_train:n_train + n_val]
    test_set = training_data[n_train + n_val:]
    
    print(f"  - 训练集:   {len(train_set)} 个查询")
    print(f"  - 验证集: {len(val_set)} 个查询")
    print(f"  - 测试集:   {len(test_set)} 个查询")
    
    if len(train_set) < 5:
        print("✗ 训练数据太少！请增加 --num-queries 参数")
        sys.exit(1)
    
    # ========== 4. 训练模型 ==========
    print(f"\n[Step 4/6] 训练 LTR 模型...")
    print(f"  - 算法: LambdaMART (LightGBM)")
    print(f"  - n_estimators: {args.n_estimators}")
    print(f"  - learning_rate: {args.learning_rate}")
    print(f"  - max_depth: {args.max_depth}")
    
    try:
        ranker = LTRRanker(feature_extractor)
        ranker.train(
            training_data=train_set,
            validation_data=val_set if len(val_set) > 0 else None,
            n_estimators=args.n_estimators,
            learning_rate=args.learning_rate,
            max_depth=args.max_depth
        )
        print("✓ 模型训练完成")
    except Exception as e:
        print(f"✗ 训练失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    # ========== 5. 评估模型 ==========
    print(f"\n[Step 5/6] 在测试集上评估模型...")
    
    if len(test_set) == 0:
        print("⚠ 没有测试数据，跳过评估")
        avg_results = None
    else:
        try:
            evaluator = RankingEvaluator()
            avg_results, full_results = evaluator.evaluate_ranker(
                test_set,
                ranker,
                k_values=[1, 3, 5, 10]
            )
            evaluator.print_results(avg_results)
        except Exception as e:
            print(f"✗ 评估失败: {e}")
            import traceback
            traceback.print_exc()
            avg_results = None
    
    # ========== 6. 保存模型和结果 ==========
    print(f"\n[Step 6/6] 保存模型和相关文件...")
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    try:
        # 保存模型
        model_path = os.path.join(args.output_dir, 'ltr_model.pkl')
        ranker.save_model(model_path)
        print(f"✓ 模型已保存到 {model_path}")
        
        # 保存特征重要性图
        try:
            feature_plot_path = os.path.join(args.output_dir, 'feature_importance.png')
            ranker.plot_feature_importance(top_k=20, save_path=feature_plot_path)
        except Exception as e:
            print(f"⚠ 无法保存特征重要性图: {e}")
        
        # 保存评估结果
        if avg_results:
            results_path = os.path.join(args.output_dir, 'evaluation_results.json')
            # 转换 numpy 类型为 Python 原生类型
            def convert_numpy(obj):
                import numpy as np
                if isinstance(obj, np.integer):
                    return int(obj)
                elif isinstance(obj, np.floating):
                    return float(obj)
                elif isinstance(obj, np.ndarray):
                    return obj.tolist()
                elif isinstance(obj, dict):
                    return {key: convert_numpy(value) for key, value in obj.items()}
                return obj
            
            avg_results_converted = convert_numpy(avg_results)
            
            with open(results_path, 'w', encoding='utf-8') as f:
                json.dump(avg_results_converted, f, indent=2)
            print(f"✓ 评估结果已保存到 {results_path}")
    except Exception as e:
        print(f"✗ 保存文件失败: {e}")
        import traceback
        traceback.print_exc()
    
    # ========== 额外：对比实验（可选） ==========
    if args.run_comparison and len(test_set) > 0:
        print(f"\n[Bonus] 运行对比实验...")
        try:
            run_comparison_experiments(
                test_set, 
                ranker, 
                feature_extractor, 
                cross_encoder, 
                args.output_dir
            )
        except Exception as e:
            print(f"⚠ 对比实验失败: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "="*70)
    print("✅ 训练流程成功完成！")
    print("="*70)
    print(f"\n模型和相关文件已保存到: {args.output_dir}/")
    print(f"\n要在 Flask 应用中使用此模型:")
    print(f"  1. 确保 {model_path} 存在")
    print(f"  2. 运行: python app.py")
    print(f"  3. 访问: http://localhost:5000/search?q=your_query&mode=ltr")


def run_comparison_experiments(test_set, ltr_ranker, feature_extractor, 
                               cross_encoder, output_dir):
    """运行对比实验：Baseline vs LTR"""
    from ranking.evaluator import RankingEvaluator
    
    print("设置对比实验...")
    
    # 创建基线排序器
    class BaselineRanker:
        """基线排序器 - 只使用 Elasticsearch 分数"""
        def predict(self, query, documents):
            return [doc.get('es_score', 0.0) for doc in documents]
    
    class CrossEncoderRanker:
        """Cross-Encoder 排序器"""
        def __init__(self, model):
            self.model = model
        
        def predict(self, query, documents):
            passages = [doc['content'][:512] for doc in documents]  # 限制长度
            pairs = [[query, p] for p in passages]
            return self.model.predict(pairs)
    
    # 准备排序器
    rankers = {
        'Baseline (ES)': BaselineRanker(),
        'Cross-Encoder': CrossEncoderRanker(cross_encoder),
        'LTR (Ours)': ltr_ranker
    }
    
    # 评估
    evaluator = RankingEvaluator()
    comparison = evaluator.compare_rankers(
        test_set,
        rankers,
        k_values=[1, 3, 5, 10]
    )
    
    # 打印对比结果
    print("\n" + "="*70)
    print("对比实验结果")
    print("="*70)
    
    for ranker_name, results in comparison.items():
        print(f"\n{ranker_name}:")
        print(f"  NDCG@10: {results['ndcg'][10]:.4f}")
        print(f"  MAP:     {results['map']:.4f}")
        print(f"  MRR:     {results['mrr']:.4f}")
    
    # 可视化对比
    try:
        plot_path = os.path.join(output_dir, 'comparison_results.png')
        evaluator.plot_comparison(comparison, save_path=plot_path)
    except Exception as e:
        print(f"⚠ 无法生成对比图: {e}")
    
    # 保存对比结果
    def convert_numpy(obj):
        import numpy as np
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {key: convert_numpy(value) for key, value in obj.items()}
        return obj
    
    comparison_converted = convert_numpy(comparison)
    comparison_path = os.path.join(output_dir, 'comparison_results.json')
    with open(comparison_path, 'w', encoding='utf-8') as f:
        json.dump(comparison_converted, f, indent=2)
    
    print(f"\n✓ 对比结果已保存到 {comparison_path}")


def run_cli():
    parser = argparse.ArgumentParser(
        description='训练 Learning to Rank 搜索排序模型'
    )
    
    # Elasticsearch 配置
    parser.add_argument('--es-host', type=str, default='localhost',
        help='Elasticsearch 主机地址 (默认: localhost)')
    parser.add_argument('--es-port', type=int, default=9200,
                       help='Elasticsearch 端口 (默认: 9200)')
    
    # 训练数据配置
    parser.add_argument('--training-data', type=str, 
                       default='data/ltr_training_data.json',
                       help='训练数据 JSON 文件路径')
    parser.add_argument('--regenerate-data', action='store_true',
                       help='即使文件存在也重新生成训练数据')
    parser.add_argument('--num-queries', type=int, default=50,
                       help='生成的训练查询数量 (默认: 50)')
    parser.add_argument('--docs-per-query', type=int, default=30,
                       help='每个查询的文档数量 (默认: 30)')
    
    # 模型参数
    parser.add_argument('--n-estimators', type=int, default=100,
                       help='Boosting 迭代次数 (默认: 100)')
    parser.add_argument('--learning-rate', type=float, default=0.05,
                       help='学习率 (默认: 0.05)')
    parser.add_argument('--max-depth', type=int, default=6,
                       help='树的最大深度 (默认: 6)')
    
    # 输出配置
    parser.add_argument('--output-dir', type=str, default='models',
                       help='保存模型的目录 (默认: models)')
    
    # 实验配置
    parser.add_argument('--run-comparison', action='store_true',
                       help='运行对比实验 (Baseline vs LTR)')
    
    args = parser.parse_args()

    main(args)


if __name__ == '__main__':
    run_cli()
