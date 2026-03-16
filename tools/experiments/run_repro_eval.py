#!/usr/bin/env python3
"""
一键可复现实验流水线：
1. 固定随机种子与切分策略
2. 评估 Baseline / LTR / Cross-Encoder / Hybrid
3. 输出 JSON + Markdown + 图表报告
"""

import argparse
import hashlib
import json
import os
import random
import sys
from datetime import datetime, timezone

import numpy as np

sys.path.insert(
    0,
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
)

from ranking.evaluator import RankingEvaluator
from ranking.feature_extractor import FeatureExtractor
from ranking.ranker import LTRRanker


def set_global_seed(seed):
    random.seed(seed)
    np.random.seed(seed)


def file_sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def convert_numpy(obj):
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, dict):
        return {k: convert_numpy(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [convert_numpy(x) for x in obj]
    return obj


def sanitize_dataset(data):
    """过滤异常样本并对文档/标签长度做对齐。"""
    cleaned = []
    dropped = 0
    for item in data:
        if not isinstance(item, dict):
            dropped += 1
            continue
        query = item.get("query")
        docs = item.get("documents")
        labels = item.get("relevance_labels")
        if not isinstance(query, str) or not isinstance(docs, list) or not isinstance(labels, list):
            dropped += 1
            continue
        n = min(len(docs), len(labels))
        if n < 2:
            dropped += 1
            continue
        cleaned.append(
            {
                "query": query,
                "documents": docs[:n],
                "relevance_labels": labels[:n],
            }
        )
    return cleaned, dropped


def ensure_dataset(args):
    """加载训练数据；若缺失则按固定 seed 生成。"""
    if os.path.exists(args.data):
        with open(args.data, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data, False

    if not args.generate_if_missing:
        raise FileNotFoundError(
            f"训练数据不存在: {args.data}\n"
            "可开启 --generate-if-missing 自动生成。"
        )

    if args.no_cross:
        raise RuntimeError("数据缺失且启用了 --no-cross，无法生成伪标签训练数据。")

    from elasticsearch import Elasticsearch
    from sentence_transformers import CrossEncoder
    from ranking.training_data_generator import TrainingDataGenerator

    es = Elasticsearch([{"host": args.es_host, "port": args.es_port, "scheme": "http"}])
    if not es.ping():
        raise RuntimeError(f"无法连接 Elasticsearch: {args.es_host}:{args.es_port}")

    cross_encoder = CrossEncoder(args.cross_model)
    generator = TrainingDataGenerator(es, cross_encoder)

    queries = generator.generate_training_queries(num_queries=args.generate_queries)
    training_data = generator.generate_training_data(
        queries,
        docs_per_query=args.generate_docs_per_query,
    )

    os.makedirs(os.path.dirname(args.data) or ".", exist_ok=True)
    generator.save_training_data(training_data, args.data)
    return training_data, True


def split_indices(total, train_ratio, val_ratio, seed, max_queries=None):
    if total < 3:
        raise ValueError("样本数量不足，至少需要 3 条查询样本。")

    indices = list(range(total))
    rng = random.Random(seed)
    rng.shuffle(indices)

    if max_queries is not None and max_queries > 0:
        indices = indices[: min(max_queries, len(indices))]

    n_total = len(indices)
    if n_total < 3:
        raise ValueError("max_queries 过小，切分后样本不足。")

    n_train = int(n_total * train_ratio)
    n_val = int(n_total * val_ratio)

    n_train = max(1, n_train)
    n_val = max(0, n_val)

    if n_train + n_val >= n_total:
        n_val = max(0, n_total - n_train - 1)
    if n_train + n_val >= n_total:
        n_train = n_total - 1
    if n_train < 1:
        n_train = 1

    n_test = n_total - n_train - n_val
    if n_test < 1:
        if n_val > 0:
            n_val -= 1
        else:
            n_train -= 1
        n_test = n_total - n_train - n_val

    train_idx = indices[:n_train]
    val_idx = indices[n_train:n_train + n_val]
    test_idx = indices[n_train + n_val:]
    return train_idx, val_idx, test_idx


class BaselineRanker:
    def predict(self, query, documents):
        return [float(doc.get("es_score", 0.0)) for doc in documents]


class CrossEncoderRanker:
    def __init__(self, model, content_chars=512):
        self.model = model
        self.content_chars = content_chars

    def predict(self, query, documents):
        passages = [(doc.get("content", "") or "")[: self.content_chars] for doc in documents]
        pairs = [[query, p] for p in passages]
        scores = self.model.predict(pairs)
        return [float(s) for s in scores]


class HybridRanker:
    def __init__(self, ltr_ranker, cross_ranker, top_n=20):
        self.ltr = ltr_ranker
        self.cross = cross_ranker
        self.top_n = top_n

    def predict(self, query, documents):
        ltr_scores = self.ltr.predict(query, documents)
        idx_sorted = sorted(
            range(len(documents)),
            key=lambda i: ltr_scores[i],
            reverse=True,
        )
        top_idx = idx_sorted[: self.top_n]
        top_docs = [documents[i] for i in top_idx]
        cross_scores = self.cross.predict(query, top_docs)
        final_scores = list(ltr_scores)
        for j, i in enumerate(top_idx):
            final_scores[i] = 0.6 * float(ltr_scores[i]) + 0.4 * float(cross_scores[j])
        return final_scores


def build_markdown_table(comparison, k_values):
    max_k = max(k_values)
    headers = ["Ranker"] + [f"NDCG@{k}" for k in k_values] + ["MAP", "MRR", f"P@{max_k}", f"R@{max_k}"]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]

    preferred_order = ["Baseline", "LTR", "Cross-Encoder", "Hybrid"]
    names = [n for n in preferred_order if n in comparison] + [
        n for n in comparison.keys() if n not in preferred_order
    ]

    for name in names:
        row = [name]
        metrics = comparison[name]
        for k in k_values:
            row.append(f"{metrics['ndcg'][k]:.4f}")
        row.extend(
            [
                f"{metrics['map']:.4f}",
                f"{metrics['mrr']:.4f}",
                f"{metrics['precision'][max_k]:.4f}",
                f"{metrics['recall'][max_k]:.4f}",
            ]
        )
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description="可复现实验：Baseline/LTR/Cross/Hybrid 离线对比")
    parser.add_argument("--data", default="data/ltr_training_data.json")
    parser.add_argument("--output-dir", default="models/repro_eval")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-queries", type=int, default=60, help="最多使用多少查询样本")
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--k", type=int, nargs="*", default=[1, 3, 5, 10])
    parser.add_argument("--no-cross", action="store_true")
    parser.add_argument("--cross-model", default="cross-encoder/ms-marco-MiniLM-L-6-v2")
    parser.add_argument("--cross-content-chars", type=int, default=512)
    parser.add_argument("--hybrid-top-n", type=int, default=20)

    parser.add_argument("--ltr-source", choices=["train", "pretrained", "auto"], default="train")
    parser.add_argument("--ltr-model-path", default="models/ltr_model.pkl")
    parser.add_argument("--n-estimators", type=int, default=100)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--max-depth", type=int, default=6)
    parser.add_argument("--lgbm-jobs", type=int, default=1)
    parser.add_argument("--save-trained-model", action="store_true")
    parser.set_defaults(deterministic=True, generate_if_missing=True)
    parser.add_argument("--deterministic", dest="deterministic", action="store_true")
    parser.add_argument("--non-deterministic", dest="deterministic", action="store_false")

    parser.add_argument("--generate-if-missing", dest="generate_if_missing", action="store_true")
    parser.add_argument("--no-generate-if-missing", dest="generate_if_missing", action="store_false")
    parser.add_argument("--generate-queries", type=int, default=50)
    parser.add_argument("--generate-docs-per-query", type=int, default=30)
    parser.add_argument("--es-host", default="elasticsearch")
    parser.add_argument("--es-port", type=int, default=9200)

    args = parser.parse_args()

    set_global_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    data, generated = ensure_dataset(args)
    data, dropped = sanitize_dataset(data)
    if len(data) < 3:
        raise RuntimeError(f"清洗后样本不足: {len(data)}")

    train_idx, val_idx, test_idx = split_indices(
        total=len(data),
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        seed=args.seed,
        max_queries=args.max_queries,
    )

    train_set = [data[i] for i in train_idx]
    val_set = [data[i] for i in val_idx]
    test_set = [data[i] for i in test_idx]

    feature_extractor = FeatureExtractor()
    evaluator = RankingEvaluator()
    rankers = {"Baseline": BaselineRanker()}

    ltr_ranker = None
    ltr_source_used = "unavailable"
    should_train = args.ltr_source == "train" or (
        args.ltr_source == "auto" and not os.path.exists(args.ltr_model_path)
    )

    if should_train:
        ltr_ranker = LTRRanker(feature_extractor)
        ltr_ranker.train(
            training_data=train_set,
            validation_data=val_set if val_set else None,
            n_estimators=args.n_estimators,
            learning_rate=args.learning_rate,
            max_depth=args.max_depth,
            random_state=args.seed,
            n_jobs=args.lgbm_jobs,
            deterministic=args.deterministic,
        )
        ltr_source_used = "train"
        if args.save_trained_model:
            os.makedirs(os.path.dirname(args.ltr_model_path) or ".", exist_ok=True)
            ltr_ranker.save_model(args.ltr_model_path)
    else:
        if os.path.exists(args.ltr_model_path):
            ltr_ranker = LTRRanker(feature_extractor)
            ltr_ranker.load_model(args.ltr_model_path)
            ltr_source_used = "pretrained"

    if ltr_ranker is not None:
        rankers["LTR"] = ltr_ranker

    cross_ranker = None
    cross_status = "disabled"
    if not args.no_cross:
        try:
            from sentence_transformers import CrossEncoder

            cross_model = CrossEncoder(args.cross_model)
            cross_ranker = CrossEncoderRanker(
                cross_model,
                content_chars=args.cross_content_chars,
            )
            rankers["Cross-Encoder"] = cross_ranker
            cross_status = "enabled"
        except Exception as e:
            cross_status = f"unavailable: {e}"

    if ltr_ranker is not None and cross_ranker is not None:
        rankers["Hybrid"] = HybridRanker(
            ltr_ranker,
            cross_ranker,
            top_n=args.hybrid_top_n,
        )

    comparison = evaluator.compare_rankers(test_set, rankers, k_values=args.k)
    comparison = convert_numpy(comparison)

    comparison_json_path = os.path.join(args.output_dir, "comparison_results.json")
    with open(comparison_json_path, "w", encoding="utf-8") as f:
        json.dump(comparison, f, indent=2, ensure_ascii=False)

    split_path = os.path.join(args.output_dir, "split_indices.json")
    with open(split_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "seed": args.seed,
                "train_indices": train_idx,
                "val_indices": val_idx,
                "test_indices": test_idx,
            },
            f,
            indent=2,
            ensure_ascii=False,
        )

    md_table_path = os.path.join(args.output_dir, "comparison_table.md")
    with open(md_table_path, "w", encoding="utf-8") as f:
        f.write(build_markdown_table(comparison, args.k))

    plot_path = os.path.join(args.output_dir, "comparison_plot.png")
    try:
        evaluator.plot_comparison(comparison, save_path=plot_path)
    except Exception:
        plot_path = None

    manifest = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "seed": args.seed,
        "deterministic": args.deterministic,
        "data_path": args.data,
        "data_generated_this_run": generated,
        "data_sha256": file_sha256(args.data) if os.path.exists(args.data) else None,
        "data_total_raw": len(data) + dropped,
        "data_dropped_invalid": dropped,
        "data_total_clean": len(data),
        "split_counts": {
            "train": len(train_set),
            "val": len(val_set),
            "test": len(test_set),
        },
        "rankers_evaluated": list(comparison.keys()),
        "ltr_source": ltr_source_used,
        "cross_status": cross_status,
        "k_values": args.k,
        "artifacts": {
            "comparison_results_json": comparison_json_path,
            "split_indices_json": split_path,
            "comparison_table_md": md_table_path,
            "comparison_plot_png": plot_path,
        },
        "args": vars(args),
    }

    manifest_path = os.path.join(args.output_dir, "run_manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(convert_numpy(manifest), f, indent=2, ensure_ascii=False)

    print("Repro evaluation completed.")
    print(f"- rankers: {', '.join(comparison.keys())}")
    print(
        f"- split: train={len(train_set)} val={len(val_set)} test={len(test_set)}"
    )
    print(f"- results: {comparison_json_path}")
    print(f"- table:   {md_table_path}")
    print(f"- manifest:{manifest_path}")
    if plot_path:
        print(f"- plot:    {plot_path}")


if __name__ == "__main__":
    main()
