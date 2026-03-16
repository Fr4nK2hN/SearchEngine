#!/usr/bin/env python3
"""
Grid search adaptive router configs:
- hard threshold
- hard top-k for hybrid reranking

Outputs:
- tuning_results.json
- tuning_table.md
- recommended_config.json
"""

import argparse
import json
import os
import pickle
import random
import sys
import time
from dataclasses import dataclass

import numpy as np

sys.path.insert(
    0,
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
)

from ranking.evaluator import RankingEvaluator
from ranking.feature_extractor import FeatureExtractor
from ranking.query_router import query_feature_vector
from ranking.ranker import LTRRanker


def percentile(values, p):
    if not values:
        return 0.0
    arr = np.array(values, dtype=float)
    return float(np.percentile(arr, p))


def parse_grid_float(v):
    out = []
    for x in str(v).split(","):
        x = x.strip()
        if not x:
            continue
        out.append(float(x))
    if not out:
        raise ValueError("float grid is empty")
    return out


def parse_grid_int(v):
    out = []
    for x in str(v).split(","):
        x = x.strip()
        if not x:
            continue
        out.append(int(x))
    if not out:
        raise ValueError("int grid is empty")
    return out


def sanitize_dataset(data):
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
                "query": " ".join(query.strip().split()),
                "documents": docs[:n],
                "relevance_labels": labels[:n],
            }
        )
    return cleaned, dropped


def sample_data(data, sample_method, max_queries, seed):
    total = len(data)
    if max_queries is None or max_queries <= 0 or max_queries >= total:
        return data
    if sample_method == "head":
        return data[:max_queries]
    if sample_method == "tail":
        return data[-max_queries:]
    rng = random.Random(seed)
    idx = sorted(rng.sample(range(total), max_queries))
    return [data[i] for i in idx]


@dataclass
class RouterModel:
    model: object
    scaler: object
    easy_mode: str
    hard_mode: str
    hard_threshold: float

    @classmethod
    def load(cls, path):
        with open(path, "rb") as f:
            payload = pickle.load(f)
        return cls(
            model=payload["model"],
            scaler=payload["scaler"],
            easy_mode=payload.get("easy_mode", "ltr"),
            hard_mode=payload.get("hard_mode", "hybrid"),
            hard_threshold=float(payload.get("hard_threshold", 0.5)),
        )

    def hard_prob(self, query):
        feat = np.array([query_feature_vector(query)], dtype=float)
        feat_s = self.scaler.transform(feat)
        prob = self.model.predict_proba(feat_s)[0]
        if len(prob) == 1:
            return float(prob[0])
        return float(prob[1])


def build_markdown(rows, recommendation):
    lines = [
        "| Threshold | Hard Top-K | nDCG@10 | MRR@10 | Recall@50 | Avg Latency (ms) | P95 Latency (ms) | Hard Rate |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for r in rows:
        lines.append(
            f"| {r['threshold']:.2f} | {r['hard_top_k']} | {r['ndcg']:.4f} | {r['mrr']:.4f} | "
            f"{r['recall']:.4f} | {r['latency_avg_ms']:.2f} | {r['latency_p95_ms']:.2f} | {r['hard_rate']:.2%} |"
        )
    lines.append("")
    lines.append("Recommended config:")
    lines.append(
        f"- threshold={recommendation['threshold']:.2f}, hard_top_k={recommendation['hard_top_k']}, "
        f"nDCG@10={recommendation['ndcg']:.4f}, avg_latency={recommendation['latency_avg_ms']:.2f}ms"
    )
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description="Tune adaptive router threshold/top-k")
    parser.add_argument("--data", default="data/ltr_training_data_v2_quantile.json")
    parser.add_argument("--router-model", default="models/query_router.pkl")
    parser.add_argument("--ltr-model", default="models/ltr_model.pkl")
    parser.add_argument("--output-dir", default="models/adaptive_tuning")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sample", choices=["head", "tail", "random"], default="random")
    parser.add_argument("--max-queries", type=int, default=60)
    parser.add_argument("--threshold-grid", default="0.35,0.45,0.50,0.55,0.65")
    parser.add_argument("--topk-grid", default="10,20,30")
    parser.add_argument("--cross-model", default="cross-encoder/ms-marco-MiniLM-L-6-v2")
    parser.add_argument("--cross-content-chars", type=int, default=512)
    parser.add_argument("--ndcg-k", type=int, default=10)
    parser.add_argument("--mrr-k", type=int, default=10)
    parser.add_argument("--recall-k", type=int, default=50)
    parser.add_argument(
        "--latency-ndcg-tolerance",
        type=float,
        default=0.005,
        help="Among near-best nDCG configs (within tolerance), pick lower latency.",
    )
    parser.add_argument(
        "--max-hard-rate",
        type=float,
        default=1.0,
        help="Upper bound for hard-route ratio when selecting recommendation.",
    )
    parser.add_argument(
        "--max-latency-ms",
        type=float,
        default=0.0,
        help="If >0, apply avg latency budget when selecting recommendation.",
    )
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    with open(args.data, "r", encoding="utf-8") as f:
        raw_data = json.load(f)
    clean_data, dropped = sanitize_dataset(raw_data)
    if len(clean_data) < 10:
        raise RuntimeError(f"clean data too small: {len(clean_data)}")
    data = sample_data(clean_data, args.sample, args.max_queries, args.seed)
    if len(data) < 8:
        raise RuntimeError(f"sampled data too small: {len(data)}")

    router = RouterModel.load(args.router_model)
    feature_extractor = FeatureExtractor()
    ltr = LTRRanker(feature_extractor)
    ltr.load_model(args.ltr_model)

    from sentence_transformers import CrossEncoder

    cross_model = CrossEncoder(args.cross_model)
    evaluator = RankingEvaluator()

    thresholds = parse_grid_float(args.threshold_grid)
    topks = parse_grid_int(args.topk_grid)

    # Precompute per-query scores and reusable timings
    cache = []
    for item in data:
        query = item["query"]
        docs = item["documents"]
        labels = item["relevance_labels"]

        # baseline
        base_scores = np.array([float(d.get("es_score", 0.0)) for d in docs], dtype=float)

        # ltr
        t0 = time.perf_counter()
        ltr_scores = np.array(ltr.predict(query, docs), dtype=float)
        t1 = time.perf_counter()
        ltr_latency = (t1 - t0) * 1000.0

        # full cross score once, reuse per top-k
        passages = [(d.get("content", "") or "")[: args.cross_content_chars] for d in docs]
        pairs = [[query, p] for p in passages]
        c0 = time.perf_counter()
        cross_scores = np.array(cross_model.predict(pairs), dtype=float)
        c1 = time.perf_counter()
        cross_latency_full = (c1 - c0) * 1000.0
        cross_latency_per_doc = cross_latency_full / max(1, len(docs))

        # router probability
        hard_prob = router.hard_prob(query)

        cache.append(
            {
                "query": query,
                "labels": labels,
                "docs_n": len(docs),
                "hard_prob": hard_prob,
                "base_scores": base_scores,
                "ltr_scores": ltr_scores,
                "cross_scores": cross_scores,
                "ltr_latency": ltr_latency,
                "cross_latency_per_doc": cross_latency_per_doc,
            }
        )

    rows = []
    for threshold in thresholds:
        for hard_top_k in topks:
            ndcgs = []
            mrrs = []
            recalls = []
            latencies = []
            hard_cnt = 0

            for q in cache:
                labels = q["labels"]
                base_scores = q["base_scores"]
                ltr_scores = q["ltr_scores"]
                cross_scores = q["cross_scores"]
                docs_n = q["docs_n"]

                is_hard = q["hard_prob"] >= threshold
                if is_hard:
                    hard_cnt += 1
                    top_n = min(max(1, int(hard_top_k)), docs_n)
                    idx_sorted = np.argsort(ltr_scores)[::-1]
                    top_idx = idx_sorted[:top_n]
                    final_scores = np.array(ltr_scores, copy=True)
                    for i in top_idx:
                        final_scores[i] = 0.6 * ltr_scores[i] + 0.4 * cross_scores[i]
                    scores = final_scores
                    latency_ms = q["ltr_latency"] + q["cross_latency_per_doc"] * top_n
                else:
                    scores = ltr_scores
                    latency_ms = q["ltr_latency"]

                ranks = np.argsort(scores)[::-1]
                ndcgs.append(evaluator.ndcg_at_k(labels, ranks, k=args.ndcg_k))
                mrrs.append(evaluator.mrr_score(labels, ranks[: args.mrr_k]))
                recalls.append(evaluator.recall_at_k(labels, ranks, k=args.recall_k))
                latencies.append(latency_ms)

            row = {
                "threshold": float(threshold),
                "hard_top_k": int(hard_top_k),
                "ndcg": float(np.mean(ndcgs)),
                "mrr": float(np.mean(mrrs)),
                "recall": float(np.mean(recalls)),
                "latency_avg_ms": float(np.mean(latencies)),
                "latency_p95_ms": percentile(latencies, 95),
                "hard_rate": float(hard_cnt / max(1, len(cache))),
                "query_count": len(cache),
            }
            rows.append(row)

    # sort for table readability: higher ndcg then lower avg latency
    rows_sorted = sorted(rows, key=lambda r: (-r["ndcg"], r["latency_avg_ms"]))
    best_ndcg = rows_sorted[0]["ndcg"]
    constrained = [
        r for r in rows_sorted
        if r["hard_rate"] <= float(args.max_hard_rate)
        and (float(args.max_latency_ms) <= 0.0 or r["latency_avg_ms"] <= float(args.max_latency_ms))
    ]
    candidate_pool = constrained if constrained else rows_sorted
    pool_best_ndcg = candidate_pool[0]["ndcg"]
    near_best = [r for r in candidate_pool if (pool_best_ndcg - r["ndcg"]) <= args.latency_ndcg_tolerance]
    recommendation = sorted(near_best, key=lambda r: (r["latency_avg_ms"], r["latency_p95_ms"]))[0]

    results = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "data_path": args.data,
        "data_raw_count": len(raw_data),
        "data_dropped_invalid": dropped,
        "data_used_count": len(data),
        "seed": args.seed,
        "sample_method": args.sample,
        "router_model_path": args.router_model,
        "ltr_model_path": args.ltr_model,
        "router_model_defaults": {
            "easy_mode": router.easy_mode,
            "hard_mode": router.hard_mode,
            "hard_threshold": router.hard_threshold,
        },
        "grids": {
            "thresholds": thresholds,
            "hard_top_k": topks,
        },
        "metrics": {
            "ndcg_k": args.ndcg_k,
            "mrr_k": args.mrr_k,
            "recall_k": args.recall_k,
        },
        "rows": rows_sorted,
        "recommendation": recommendation,
        "recommendation_rule": {
            "max_ndcg": best_ndcg,
            "max_ndcg_under_constraints": pool_best_ndcg,
            "ndcg_tolerance": args.latency_ndcg_tolerance,
            "max_hard_rate": args.max_hard_rate,
            "max_latency_ms": args.max_latency_ms,
            "constraint_applied": bool(constrained),
            "tie_break": "min avg_latency_ms then min p95_latency_ms",
        },
    }

    results_path = os.path.join(args.output_dir, "tuning_results.json")
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    table_path = os.path.join(args.output_dir, "tuning_table.md")
    with open(table_path, "w", encoding="utf-8") as f:
        f.write(build_markdown(rows_sorted, recommendation))

    rec_path = os.path.join(args.output_dir, "recommended_config.json")
    with open(rec_path, "w", encoding="utf-8") as f:
        json.dump(recommendation, f, indent=2, ensure_ascii=False)

    print("Adaptive tuning completed.")
    print(f"- rows: {len(rows_sorted)}")
    print(
        f"- recommendation: threshold={recommendation['threshold']:.2f}, "
        f"hard_top_k={recommendation['hard_top_k']}, "
        f"ndcg={recommendation['ndcg']:.4f}, "
        f"avg_latency_ms={recommendation['latency_avg_ms']:.2f}"
    )
    print(f"- output: {args.output_dir}")


if __name__ == "__main__":
    main()
