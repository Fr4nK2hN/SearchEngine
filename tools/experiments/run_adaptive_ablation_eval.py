#!/usr/bin/env python3
"""
Run multi-seed offline evaluation for:
1) Main table: Baseline / LTR / Cross-Encoder / Hybrid / Adaptive
2) Ablation table:
   - fixed reranker (no routing)
   - router + fixed top-k
   - router + adaptive top-k
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


def parse_seeds(text):
    out = []
    for part in str(text).split(","):
        part = part.strip()
        if not part:
            continue
        out.append(int(part))
    if not out:
        raise ValueError("seeds cannot be empty")
    return out


def parse_adaptive_topk(text):
    """
    Parse policy string like: "0.10:10,0.25:20,1.00:30"
    Means by delta=(hard_prob-threshold):
      delta <= 0.10 -> 10
      delta <= 0.25 -> 20
      delta <= 1.00 -> 30
    """
    pairs = []
    for part in str(text).split(","):
        part = part.strip()
        if not part:
            continue
        delta_s, topk_s = part.split(":")
        pairs.append((float(delta_s), int(topk_s)))
    if not pairs:
        raise ValueError("adaptive_topk policy is empty")
    pairs = sorted(pairs, key=lambda x: x[0])
    for _, topk in pairs:
        if topk <= 0:
            raise ValueError("adaptive top-k must be positive")
    return pairs


def sanitize_dataset(data):
    cleaned = []
    dropped = 0
    for item in data:
        if not isinstance(item, dict):
            dropped += 1
            continue
        q = item.get("query")
        docs = item.get("documents")
        labels = item.get("relevance_labels")
        if not isinstance(q, str) or not isinstance(docs, list) or not isinstance(labels, list):
            dropped += 1
            continue
        n = min(len(docs), len(labels))
        if n < 2:
            dropped += 1
            continue
        cleaned.append(
            {
                "query": " ".join(q.strip().split()),
                "documents": docs[:n],
                "relevance_labels": labels[:n],
            }
        )
    return cleaned, dropped


def minmax_norm(values):
    if len(values) == 0:
        return np.array(values, dtype=float)
    arr = np.asarray(values, dtype=float)
    vmin = float(np.min(arr))
    vmax = float(np.max(arr))
    if vmax == vmin:
        return np.full_like(arr, 0.5, dtype=float)
    return (arr - vmin) / (vmax - vmin)


def percentile(values, p):
    if not values:
        return 0.0
    return float(np.percentile(np.asarray(values, dtype=float), p))


def mean_std(values):
    arr = np.asarray(values, dtype=float)
    return float(np.mean(arr)), float(np.std(arr))


def safe_hard_prob(model, scaler, query):
    feat = np.asarray([query_feature_vector(query)], dtype=float)
    feat_s = scaler.transform(feat)
    prob = model.predict_proba(feat_s)[0]
    classes = list(getattr(model, "classes_", []))
    if len(prob) == 1:
        if classes and int(classes[0]) == 1:
            return float(prob[0])
        return 0.0
    if 1 in classes:
        return float(prob[classes.index(1)])
    return float(prob[-1])


@dataclass
class QueryCache:
    query: str
    labels: list
    docs_n: int
    es_order: np.ndarray
    es_scores: np.ndarray
    ltr_scores: np.ndarray
    cross_scores: np.ndarray
    ltr_latency_ms: float
    cross_latency_per_doc_ms: float
    hard_prob: float


class EvaluatorRunner:
    def __init__(self, query_caches, ndcg_k=10, mrr_k=10, recall_k=50):
        self.query_caches = query_caches
        self.ev = RankingEvaluator()
        self.ndcg_k = ndcg_k
        self.mrr_k = mrr_k
        self.recall_k = recall_k

    def _baseline_ranks(self, qc):
        # Baseline follows ES retrieval order.
        return qc.es_order, 0.0

    def _ltr_ranks(self, qc):
        ranks = np.argsort(qc.ltr_scores)[::-1]
        return ranks, qc.ltr_latency_ms

    def _cross_ranks(self, qc, top_n=None):
        if top_n is None:
            top_n = qc.docs_n
        top_n = int(max(1, min(top_n, qc.docs_n)))
        head_idx = qc.es_order[:top_n]
        tail_idx = qc.es_order[top_n:]
        es_norm = minmax_norm(qc.es_scores[head_idx])
        ce_norm = minmax_norm(qc.cross_scores[head_idx])
        fused = 0.3 * es_norm + 0.7 * ce_norm
        head_sorted = head_idx[np.argsort(fused)[::-1]]
        final_ranks = np.concatenate([head_sorted, tail_idx], axis=0)
        latency = qc.cross_latency_per_doc_ms * top_n
        return final_ranks, latency

    def _hybrid_ranks(self, qc, top_n=10):
        top_n = int(max(1, min(top_n, qc.docs_n)))
        ltr_ranks = np.argsort(qc.ltr_scores)[::-1]
        head_idx = ltr_ranks[:top_n]
        tail_idx = ltr_ranks[top_n:]
        ltr_norm = minmax_norm(qc.ltr_scores[head_idx])
        ce_norm = minmax_norm(qc.cross_scores[head_idx])
        fused = 0.6 * ltr_norm + 0.4 * ce_norm
        head_sorted = head_idx[np.argsort(fused)[::-1]]
        final_ranks = np.concatenate([head_sorted, tail_idx], axis=0)
        latency = qc.ltr_latency_ms + qc.cross_latency_per_doc_ms * top_n
        return final_ranks, latency

    def _adaptive_topk(self, hard_prob, threshold, fixed_topk, topk_policy):
        if hard_prob < threshold:
            return None
        delta = hard_prob - threshold
        for bound, topk in topk_policy:
            if delta <= bound:
                return int(topk)
        return int(fixed_topk)

    def _route_method(self, qc, router_cfg, adaptive_topk=False, topk_policy=None):
        easy_mode = router_cfg["easy_mode"]
        hard_mode = router_cfg["hard_mode"]
        threshold = float(router_cfg["hard_threshold"])
        fixed_topk = int(router_cfg["hard_top_k"])
        is_hard = qc.hard_prob >= threshold
        if not is_hard:
            return self._mode_ranks(qc, easy_mode, top_n=None), 0

        top_n = fixed_topk
        if adaptive_topk and topk_policy:
            top_n = self._adaptive_topk(qc.hard_prob, threshold, fixed_topk, topk_policy)
            top_n = int(max(1, min(top_n, qc.docs_n)))
        return self._mode_ranks(qc, hard_mode, top_n=top_n), 1

    def _mode_ranks(self, qc, mode, top_n=None):
        if mode == "baseline":
            return self._baseline_ranks(qc)
        if mode == "ltr":
            return self._ltr_ranks(qc)
        if mode == "cross_encoder":
            return self._cross_ranks(qc, top_n=top_n)
        if mode == "hybrid":
            use_top_n = top_n if top_n is not None else 10
            return self._hybrid_ranks(qc, top_n=use_top_n)
        # fallback
        return self._ltr_ranks(qc)

    def evaluate_seed(self, indices, router_cfg, topk_policy):
        methods = {
            "Baseline": {"type": "mode", "mode": "baseline"},
            "LTR": {"type": "mode", "mode": "ltr"},
            "Cross-Encoder": {"type": "mode", "mode": "cross_encoder"},
            "Hybrid": {"type": "mode", "mode": "hybrid", "top_n": 10},
            "Adaptive": {"type": "adaptive", "adaptive_topk": False},
            "Fixed-LTR (No Routing)": {"type": "mode", "mode": "ltr"},
            "Fixed-Hybrid@30 (No Routing)": {"type": "mode", "mode": "hybrid", "top_n": 30},
            "Router + Fixed Top-K": {"type": "adaptive", "adaptive_topk": False},
            "Router + Adaptive Top-K": {"type": "adaptive", "adaptive_topk": True},
        }

        per_method = {}
        for name, cfg in methods.items():
            ndcgs = []
            mrrs = []
            recalls = []
            latencies = []
            hard_count = 0

            for idx in indices:
                qc = self.query_caches[idx]
                if cfg["type"] == "mode":
                    mode = cfg["mode"]
                    top_n = cfg.get("top_n")
                    ranks, latency = self._mode_ranks(qc, mode, top_n=top_n)
                else:
                    (ranks, latency), is_hard = self._route_method(
                        qc,
                        router_cfg=router_cfg,
                        adaptive_topk=bool(cfg.get("adaptive_topk", False)),
                        topk_policy=topk_policy,
                    )
                    hard_count += int(is_hard)

                labels = qc.labels
                ndcgs.append(self.ev.ndcg_at_k(labels, ranks, k=self.ndcg_k))
                mrrs.append(self.ev.mrr_score(labels, ranks[: self.mrr_k]))
                recalls.append(self.ev.recall_at_k(labels, ranks, k=self.recall_k))
                latencies.append(float(latency))

            item = {
                "ndcg": float(np.mean(ndcgs)),
                "mrr": float(np.mean(mrrs)),
                "recall": float(np.mean(recalls)),
                "latency_avg_ms": float(np.mean(latencies)),
                "latency_p95_ms": percentile(latencies, 95),
            }
            if cfg["type"] == "adaptive":
                item["hard_rate"] = float(hard_count / max(1, len(indices)))
            per_method[name] = item

        main_rows = ["Baseline", "LTR", "Cross-Encoder", "Hybrid", "Adaptive"]
        ablation_rows = [
            "Fixed-LTR (No Routing)",
            "Fixed-Hybrid@30 (No Routing)",
            "Router + Fixed Top-K",
            "Router + Adaptive Top-K",
        ]
        return per_method, main_rows, ablation_rows


def build_md_table(rows, agg, include_hard_rate):
    headers = [
        "Method",
        "nDCG@10 (mean±std)",
        "MRR@10 (mean±std)",
        "Recall@50 (mean±std)",
        "Avg Latency ms (mean±std)",
        "P95 Latency ms (mean±std)",
    ]
    if include_hard_rate:
        headers.append("Hard Rate (mean±std)")

    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]

    for name in rows:
        item = agg[name]
        row = [
            name,
            f"{item['ndcg_mean']:.4f}±{item['ndcg_std']:.4f}",
            f"{item['mrr_mean']:.4f}±{item['mrr_std']:.4f}",
            f"{item['recall_mean']:.4f}±{item['recall_std']:.4f}",
            f"{item['latency_avg_mean']:.2f}±{item['latency_avg_std']:.2f}",
            f"{item['latency_p95_mean']:.2f}±{item['latency_p95_std']:.2f}",
        ]
        if include_hard_rate:
            hr_mean = item.get("hard_rate_mean")
            hr_std = item.get("hard_rate_std")
            if hr_mean is None:
                row.append("-")
            else:
                row.append(f"{hr_mean:.2%}±{hr_std:.2%}")
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines) + "\n"


def aggregate_over_seeds(seed_results, method_names):
    agg = {}
    for name in method_names:
        ndcg = [x[name]["ndcg"] for x in seed_results]
        mrr = [x[name]["mrr"] for x in seed_results]
        recall = [x[name]["recall"] for x in seed_results]
        lat_avg = [x[name]["latency_avg_ms"] for x in seed_results]
        lat_p95 = [x[name]["latency_p95_ms"] for x in seed_results]
        ndcg_mean, ndcg_std = mean_std(ndcg)
        mrr_mean, mrr_std = mean_std(mrr)
        recall_mean, recall_std = mean_std(recall)
        lat_avg_mean, lat_avg_std = mean_std(lat_avg)
        lat_p95_mean, lat_p95_std = mean_std(lat_p95)
        item = {
            "ndcg_mean": ndcg_mean,
            "ndcg_std": ndcg_std,
            "mrr_mean": mrr_mean,
            "mrr_std": mrr_std,
            "recall_mean": recall_mean,
            "recall_std": recall_std,
            "latency_avg_mean": lat_avg_mean,
            "latency_avg_std": lat_avg_std,
            "latency_p95_mean": lat_p95_mean,
            "latency_p95_std": lat_p95_std,
        }
        hrs = [x[name].get("hard_rate") for x in seed_results if "hard_rate" in x[name]]
        if hrs:
            hr_mean, hr_std = mean_std(hrs)
            item["hard_rate_mean"] = hr_mean
            item["hard_rate_std"] = hr_std
        agg[name] = item
    return agg


def main():
    parser = argparse.ArgumentParser(description="Run adaptive main+ablation evaluation")
    parser.add_argument("--data", default="data/ltr_training_data_v2_quantile.json")
    parser.add_argument("--router-model", default="models/query_router.pkl")
    parser.add_argument("--ltr-model", default="models/ltr_model.pkl")
    parser.add_argument("--cross-model", default="cross-encoder/ms-marco-MiniLM-L-6-v2")
    parser.add_argument("--cross-content-chars", type=int, default=512)
    parser.add_argument("--output-dir", default="models/adaptive_ablation_eval")
    parser.add_argument("--seeds", default="0,1,2,3,4")
    parser.add_argument("--max-queries", type=int, default=60)
    parser.add_argument("--adaptive-topk", default="0.10:10,0.25:20,1.00:30")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    seeds = parse_seeds(args.seeds)
    topk_policy = parse_adaptive_topk(args.adaptive_topk)

    with open(args.data, "r", encoding="utf-8") as f:
        raw_data = json.load(f)
    data, dropped = sanitize_dataset(raw_data)
    if len(data) < 8:
        raise RuntimeError(f"clean data too small: {len(data)}")

    with open(args.router_model, "rb") as f:
        router_payload = pickle.load(f)
    router_model = router_payload["model"]
    router_scaler = router_payload["scaler"]
    router_cfg = {
        "easy_mode": router_payload.get("easy_mode", "ltr"),
        "hard_mode": router_payload.get("hard_mode", "hybrid"),
        "hard_threshold": float(router_payload.get("hard_threshold", 0.5)),
        "hard_top_k": int(router_payload.get("hard_top_k", 30)),
    }

    feature_extractor = FeatureExtractor()
    ltr = LTRRanker(feature_extractor)
    ltr.load_model(args.ltr_model)

    from sentence_transformers import CrossEncoder

    cross_model = CrossEncoder(args.cross_model)

    # Build per-seed sampled indices and precompute union.
    all_indices = set()
    seed_indices = {}
    total = len(data)
    sample_n = min(max(1, args.max_queries), total)
    for s in seeds:
        rng = random.Random(s)
        idx = sorted(rng.sample(range(total), sample_n))
        seed_indices[s] = idx
        all_indices.update(idx)

    caches = {}
    sorted_union = sorted(all_indices)
    for i in sorted_union:
        item = data[i]
        query = item["query"]
        docs = item["documents"]
        labels = item["relevance_labels"]
        docs_n = len(docs)

        es_scores = np.asarray([float(d.get("es_score", 0.0)) for d in docs], dtype=float)
        es_order = np.arange(docs_n, dtype=int)

        t0 = time.perf_counter()
        ltr_scores = np.asarray(ltr.predict(query, docs), dtype=float)
        t1 = time.perf_counter()
        ltr_latency = (t1 - t0) * 1000.0

        passages = [(d.get("content", "") or "")[: args.cross_content_chars] for d in docs]
        pairs = [[query, p] for p in passages]
        c0 = time.perf_counter()
        cross_scores = np.asarray(cross_model.predict(pairs), dtype=float)
        c1 = time.perf_counter()
        cross_latency_per_doc = ((c1 - c0) * 1000.0) / max(1, docs_n)

        hard_prob = safe_hard_prob(router_model, router_scaler, query)

        caches[i] = QueryCache(
            query=query,
            labels=labels,
            docs_n=docs_n,
            es_order=es_order,
            es_scores=es_scores,
            ltr_scores=ltr_scores,
            cross_scores=cross_scores,
            ltr_latency_ms=float(ltr_latency),
            cross_latency_per_doc_ms=float(cross_latency_per_doc),
            hard_prob=float(hard_prob),
        )

    runner = EvaluatorRunner(query_caches=caches)
    seed_results = []
    main_rows = None
    ablation_rows = None
    for s in seeds:
        per_method, main_rows, ablation_rows = runner.evaluate_seed(
            seed_indices[s],
            router_cfg=router_cfg,
            topk_policy=topk_policy,
        )
        seed_results.append(per_method)

    all_method_names = list(seed_results[0].keys())
    agg = aggregate_over_seeds(seed_results, all_method_names)

    main_md = build_md_table(main_rows, agg, include_hard_rate=True)
    ablation_md = build_md_table(ablation_rows, agg, include_hard_rate=True)

    result_json = {
        "config": {
            "data": args.data,
            "router_model": args.router_model,
            "ltr_model": args.ltr_model,
            "cross_model": args.cross_model,
            "seeds": seeds,
            "max_queries": sample_n,
            "adaptive_topk_policy": args.adaptive_topk,
            "router_cfg": router_cfg,
        },
        "data_info": {
            "raw_count": len(raw_data),
            "clean_count": len(data),
            "dropped_invalid": dropped,
            "seed_sample_size": sample_n,
            "union_precomputed_count": len(sorted_union),
        },
        "seed_indices": seed_indices,
        "seed_results": seed_results,
        "aggregated": agg,
        "main_rows": main_rows,
        "ablation_rows": ablation_rows,
    }

    json_path = os.path.join(args.output_dir, "results.json")
    main_md_path = os.path.join(args.output_dir, "main_results.md")
    ablation_md_path = os.path.join(args.output_dir, "ablation_results.md")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(result_json, f, indent=2, ensure_ascii=False)
    with open(main_md_path, "w", encoding="utf-8") as f:
        f.write(main_md)
    with open(ablation_md_path, "w", encoding="utf-8") as f:
        f.write(ablation_md)

    print("Adaptive ablation evaluation completed.")
    print(f"- output dir: {args.output_dir}")
    print(f"- json:       {json_path}")
    print(f"- main table: {main_md_path}")
    print(f"- ablation:   {ablation_md_path}")


if __name__ == "__main__":
    main()
