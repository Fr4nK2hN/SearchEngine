#!/usr/bin/env python3
"""
Run two current-system tradeoff scans:
1) Hard-rate scan by deriving threshold values that approximate target hard rates
2) Cross-encoder rerank-depth scan on the current adaptive setup
"""

import argparse
import copy
import json
import math
import os
import sys
import time
import warnings
from collections import Counter
from datetime import datetime, timezone

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app import (  # noqa: E402
    ADAPTIVE_HARD_TOP_K_CAP,
    RECALL_RELAX_THRESHOLD,
    _apply_ranking_mode,
    _is_ltr_available,
    es,
    query_router,
)
from ranking.query_router import adaptive_guardrail, query_feature_vector  # noqa: E402
from retrieval import search_documents_with_fallback  # noqa: E402


warnings.filterwarnings(
    "ignore",
    message="X does not have valid feature names, but LGBMRanker was fitted with feature names",
)


ALL_GUARDRAILS = {
    "topical_easy_ltr",
    "hard_question_prefix_baseline",
    "hard_long_mix_ltr",
}


def _to_positive_int(value):
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return None
    return parsed if parsed > 0 else None


def percentile(values, p):
    if not values:
        return 0.0
    arr = sorted(float(v) for v in values)
    if len(arr) == 1:
        return arr[0]
    pos = (len(arr) - 1) * (p / 100.0)
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return arr[lo]
    weight = pos - lo
    return arr[lo] * (1 - weight) + arr[hi] * weight


def dcg_at_k(labels, k):
    score = 0.0
    for i, rel in enumerate(labels[:k]):
        score += (2 ** float(rel) - 1.0) / math.log2(i + 2.0)
    return score


def ndcg_from_sparse_labels(returned_labels, missing_labels, k=10):
    if k <= 0:
        return 0.0
    dcg = dcg_at_k(returned_labels, k)
    ideal_labels = sorted(list(returned_labels) + list(missing_labels), reverse=True)
    idcg = dcg_at_k(ideal_labels, k)
    if idcg <= 0:
        return 0.0
    return dcg / idcg


def mrr_at_k(returned_labels, k=10):
    for i, rel in enumerate(returned_labels[:k]):
        if rel > 0:
            return 1.0 / (i + 1)
    return 0.0


def recall_at_k(returned_labels, missing_labels, k=10):
    total_relevant = sum(1 for rel in list(returned_labels) + list(missing_labels) if rel > 0)
    if total_relevant <= 0:
        return 0.0
    found = sum(1 for rel in returned_labels[:k] if rel > 0)
    return found / total_relevant


def load_dataset(path):
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    cleaned = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        query = item.get("query")
        docs = item.get("documents")
        labels = item.get("relevance_labels")
        if not isinstance(query, str) or not isinstance(docs, list) or not isinstance(labels, list):
            continue
        n = min(len(docs), len(labels))
        if n < 1:
            continue
        cleaned.append(
            {
                "query": " ".join(query.strip().split()),
                "documents": docs[:n],
                "relevance_labels": labels[:n],
            }
        )
    return cleaned


def label_map_from_item(item):
    out = {}
    for doc, label in zip(item["documents"], item["relevance_labels"]):
        if not isinstance(doc, dict):
            continue
        doc_id = doc.get("id") or doc.get("_id")
        if doc_id is None:
            continue
        out[str(doc_id)] = int(label)
    return out


def hard_prob_for_query(query):
    if not query_router.loaded or query_router.model is None or query_router.scaler is None:
        raise RuntimeError("query router model is not loaded; cannot run hard-rate scan")
    feat = np.asarray([query_feature_vector(query)], dtype=float)
    feat_s = query_router.scaler.transform(feat)
    prob = query_router.model.predict_proba(feat_s)[0]
    classes = list(getattr(query_router.model, "classes_", []))
    if len(prob) == 1:
        if classes and int(classes[0]) == 1:
            return float(prob[0])
        return 0.0
    if 1 in classes:
        return float(prob[classes.index(1)])
    return float(prob[-1])


def resolve_hard_top_k(hard_prob, threshold):
    base_top_k = max(1, int(query_router.hard_top_k))
    if not query_router.hard_topk_policy:
        return base_top_k
    delta = max(0.0, float(hard_prob) - float(threshold))
    for bound, top_k in query_router.hard_topk_policy:
        if delta <= float(bound):
            return max(1, int(top_k))
    return base_top_k


def apply_guardrails(query, route, enabled_guardrails):
    if not isinstance(route, dict):
        return route
    selected_mode = route.get("selected_mode") or "baseline"
    override = adaptive_guardrail(
        query=query,
        route_label=route.get("route_label"),
        selected_mode=selected_mode,
        ltr_available=_is_ltr_available(),
        enabled_guardrails=enabled_guardrails,
    )
    if not override:
        return dict(route)
    guarded = dict(route)
    guarded.update(override)
    return guarded


def resolve_route(query, threshold, enabled_guardrails, ce_top_n_override=None):
    hard_prob = hard_prob_for_query(query)
    if hard_prob >= float(threshold):
        route = {
            "route_label": "hard",
            "route_confidence": hard_prob,
            "route_source": "model",
            "selected_mode": query_router.hard_mode,
            "hard_top_k": resolve_hard_top_k(hard_prob, threshold),
        }
    else:
        route = {
            "route_label": "easy",
            "route_confidence": 1.0 - hard_prob,
            "route_source": "model",
            "selected_mode": query_router.easy_mode,
            "hard_top_k": int(query_router.hard_top_k),
        }

    route = apply_guardrails(query, route, enabled_guardrails)
    selected_mode = route.get("selected_mode") or "baseline"
    route_label = route.get("route_label", "easy")
    hard_top_k = _to_positive_int(route.get("hard_top_k")) or 30
    if ADAPTIVE_HARD_TOP_K_CAP is not None:
        hard_top_k = min(hard_top_k, ADAPTIVE_HARD_TOP_K_CAP)

    if selected_mode in ("ltr", "hybrid") and not _is_ltr_available():
        selected_mode = "cross_encoder" if route_label == "hard" else "baseline"

    if route_label == "hard" and selected_mode in ("cross_encoder", "hybrid") and ce_top_n_override is not None:
        rerank_top_n = max(1, int(ce_top_n_override))
    else:
        rerank_top_n = hard_top_k if route_label == "hard" and selected_mode in ("cross_encoder", "hybrid") else None

    route["selected_mode"] = selected_mode
    route["hard_top_k"] = hard_top_k
    route["rerank_top_n"] = rerank_top_n
    route["hard_prob"] = hard_prob
    route["threshold_used"] = float(threshold)
    return route


def evaluate_config(query_cache, threshold, enabled_guardrails, top_k, ce_top_n_override=None, measure_latency=True):
    ndcgs = []
    mrrs = []
    recalls = []
    latencies = []
    route_labels = []
    selected_modes = []
    guardrail_counts = Counter()
    ce_depths = []

    for item in query_cache:
        query = item["query"]
        route = resolve_route(
            query,
            threshold=threshold,
            enabled_guardrails=enabled_guardrails,
            ce_top_n_override=ce_top_n_override,
        )

        results = copy.deepcopy(item["retrieval_results"])
        rank_t0 = time.perf_counter()
        ranked, _, _, _ = _apply_ranking_mode(
            query,
            results,
            route.get("selected_mode", "baseline"),
            rerank_top_n=route.get("rerank_top_n"),
        )
        rank_t1 = time.perf_counter()

        final_results = ranked[: max(1, int(top_k))]
        returned_ids = [str(hit.get("_id")) for hit in final_results if hit.get("_id") is not None]
        returned_id_set = set(returned_ids)
        returned_labels = [int(item["label_map"].get(doc_id, 0)) for doc_id in returned_ids]
        missing_labels = [
            int(label)
            for doc_id, label in item["label_map"].items()
            if int(label) > 0 and doc_id not in returned_id_set
        ]

        ndcgs.append(ndcg_from_sparse_labels(returned_labels, missing_labels, k=top_k))
        mrrs.append(mrr_at_k(returned_labels, k=top_k))
        recalls.append(recall_at_k(returned_labels, missing_labels, k=top_k))
        if measure_latency:
            latencies.append(float(item["retrieval_ms"] + (rank_t1 - rank_t0) * 1000.0))

        route_labels.append(str(route.get("route_label") or "unknown"))
        selected_modes.append(str(route.get("selected_mode") or "unknown"))
        if route.get("route_guardrail"):
            guardrail_counts[str(route["route_guardrail"])] += 1
        if route.get("route_label") == "hard" and route.get("selected_mode") in ("cross_encoder", "hybrid"):
            ce_depths.append(int(route.get("rerank_top_n") or 0))

    total = max(1, len(route_labels))
    hard_count = sum(1 for x in route_labels if x == "hard")
    return {
        "ndcg@10": float(sum(ndcgs) / len(ndcgs)),
        "mrr@10": float(sum(mrrs) / len(mrrs)),
        "recall@10": float(sum(recalls) / len(recalls)),
        "latency_ms_avg": float(sum(latencies) / len(latencies)) if latencies else 0.0,
        "latency_ms_p95": float(percentile(latencies, 95)) if latencies else 0.0,
        "hard_rate": float(hard_count / total),
        "route_counts": dict(Counter(route_labels)),
        "selected_mode_counts": dict(Counter(selected_modes)),
        "guardrail_counts": dict(guardrail_counts),
        "guardrail_hit_rate": float(sum(guardrail_counts.values()) / total),
        "ce_query_count": len(ce_depths),
        "avg_ce_top_n": float(sum(ce_depths) / len(ce_depths)) if ce_depths else 0.0,
        "threshold": float(threshold),
        "sample_count": len(query_cache),
    }


def threshold_for_target_rate(hard_probs, target_rate):
    target_rate = max(0.0, min(1.0, float(target_rate)))
    n = len(hard_probs)
    target_hard = int(round(target_rate * n))
    sorted_probs = sorted(float(x) for x in hard_probs)
    if target_hard <= 0:
        return 1.01
    if target_hard >= n:
        return -0.01
    split = n - target_hard
    low = sorted_probs[split - 1]
    high = sorted_probs[split]
    if high == low:
        return high
    return (low + high) / 2.0


def build_threshold_markdown(rows, dataset_path, query_count):
    lines = [
        f"# Hard Rate Scan ({datetime.now(timezone.utc).date().isoformat()})",
        "",
        f"- Dataset: `{dataset_path}`",
        f"- Queries: `{query_count}`",
        "",
        "| Target Hard Rate | Actual Hard Rate | Threshold | nDCG@10 | MRR@10 | Recall@10 | Avg Latency (ms) | P95 Latency (ms) | CE Queries |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            f"| {row['target_hard_rate']:.0%} | {row['hard_rate']:.1%} | {row['threshold']:.4f} | "
            f"{row['ndcg@10']:.4f} | {row['mrr@10']:.4f} | {row['recall@10']:.4f} | "
            f"{row['latency_ms_avg']:.1f} | {row['latency_ms_p95']:.1f} | {row['ce_query_count']} |"
        )
    return "\n".join(lines) + "\n"


def build_ce_depth_markdown(rows, dataset_path, query_count):
    lines = [
        f"# CE Depth Scan ({datetime.now(timezone.utc).date().isoformat()})",
        "",
        f"- Dataset: `{dataset_path}`",
        f"- Queries: `{query_count}`",
        "",
        "| Config | CE Top-N | nDCG@10 | MRR@10 | Recall@10 | Avg Latency (ms) | P95 Latency (ms) | Hard Rate | CE Queries |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            f"| {row['config_name']} | {row['ce_top_n_label']} | {row['ndcg@10']:.4f} | {row['mrr@10']:.4f} | "
            f"{row['recall@10']:.4f} | {row['latency_ms_avg']:.1f} | {row['latency_ms_p95']:.1f} | "
            f"{row['hard_rate']:.1%} | {row['ce_query_count']} |"
        )
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description="Run current adaptive hard-rate and CE-depth scans")
    parser.add_argument("--data", default="data/ltr_training_data_feedback_combined.json")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--retrieval-size", type=int, default=50)
    parser.add_argument("--warmup", type=int, default=1, help="Number of full warmup passes before measuring")
    parser.add_argument("--hard-rate-targets", default="0.1,0.2,0.3,0.4,0.5")
    parser.add_argument("--ce-depths", default="5,10,20,30")
    parser.add_argument("--out-dir", default="reports/adaptive_tradeoff_scans_20260315")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    data = load_dataset(args.data)
    if not data:
        raise RuntimeError(f"dataset is empty: {args.data}")

    query_cache = []
    for item in data:
        query = item["query"]
        t0 = time.perf_counter()
        results, retrieval_strategy = search_documents_with_fallback(
            es,
            query,
            size=max(1, int(args.retrieval_size)),
            hl=False,
            relax_threshold=RECALL_RELAX_THRESHOLD,
            index_name="documents",
        )
        t1 = time.perf_counter()
        query_cache.append(
            {
                "query": query,
                "retrieval_results": results,
                "retrieval_ms": (t1 - t0) * 1000.0,
                "retrieval_strategy": retrieval_strategy,
                "label_map": label_map_from_item(item),
            }
        )

    current_threshold = float(query_router.hard_threshold)
    current_ce_depth = int(max(1, int(query_router.hard_top_k)))

    hard_rate_targets = [float(x.strip()) for x in args.hard_rate_targets.split(",") if x.strip()]
    ce_depths = [int(x.strip()) for x in args.ce_depths.split(",") if x.strip()]

    hard_probs = [hard_prob_for_query(item["query"]) for item in query_cache]
    threshold_scan_specs = [
        {
            "target_hard_rate": target,
            "threshold": threshold_for_target_rate(hard_probs, target),
        }
        for target in hard_rate_targets
    ]
    ce_scan_specs = [{"config_name": "current_policy", "ce_top_n": None}] + [
        {"config_name": f"ce_top_{depth}", "ce_top_n": depth}
        for depth in ce_depths
    ]

    enabled_guardrails = set(ALL_GUARDRAILS)

    for _ in range(max(0, int(args.warmup))):
        for spec in threshold_scan_specs:
            evaluate_config(
                query_cache,
                threshold=spec["threshold"],
                enabled_guardrails=enabled_guardrails,
                top_k=args.top_k,
                ce_top_n_override=None,
                measure_latency=False,
            )
        for spec in ce_scan_specs:
            evaluate_config(
                query_cache,
                threshold=current_threshold,
                enabled_guardrails=enabled_guardrails,
                top_k=args.top_k,
                ce_top_n_override=spec["ce_top_n"],
                measure_latency=False,
            )

    threshold_rows = []
    for spec in threshold_scan_specs:
        evaluate_config(
            query_cache,
            threshold=spec["threshold"],
            enabled_guardrails=enabled_guardrails,
            top_k=args.top_k,
            ce_top_n_override=None,
            measure_latency=False,
        )
        row = evaluate_config(
            query_cache,
            threshold=spec["threshold"],
            enabled_guardrails=enabled_guardrails,
            top_k=args.top_k,
            ce_top_n_override=None,
            measure_latency=True,
        )
        row["target_hard_rate"] = spec["target_hard_rate"]
        threshold_rows.append(row)

    ce_rows = []
    for spec in ce_scan_specs:
        evaluate_config(
            query_cache,
            threshold=current_threshold,
            enabled_guardrails=enabled_guardrails,
            top_k=args.top_k,
            ce_top_n_override=spec["ce_top_n"],
            measure_latency=False,
        )
        row = evaluate_config(
            query_cache,
            threshold=current_threshold,
            enabled_guardrails=enabled_guardrails,
            top_k=args.top_k,
            ce_top_n_override=spec["ce_top_n"],
            measure_latency=True,
        )
        row["config_name"] = spec["config_name"]
        row["ce_top_n"] = spec["ce_top_n"]
        row["ce_top_n_label"] = "policy" if spec["ce_top_n"] is None else str(spec["ce_top_n"])
        ce_rows.append(row)

    output = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "dataset": args.data,
        "top_k": int(args.top_k),
        "retrieval_size": int(args.retrieval_size),
        "warmup_passes": int(args.warmup),
        "current_threshold": current_threshold,
        "current_hard_top_k": current_ce_depth,
        "guardrails": sorted(enabled_guardrails),
        "hard_rate_scan": threshold_rows,
        "ce_depth_scan": ce_rows,
    }

    json_path = os.path.join(args.out_dir, "adaptive_tradeoff_scans.json")
    threshold_md_path = os.path.join(args.out_dir, "hard_rate_scan.md")
    ce_md_path = os.path.join(args.out_dir, "ce_depth_scan.md")

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    with open(threshold_md_path, "w", encoding="utf-8") as f:
        f.write(build_threshold_markdown(threshold_rows, args.data, len(query_cache)))
    with open(ce_md_path, "w", encoding="utf-8") as f:
        f.write(build_ce_depth_markdown(ce_rows, args.data, len(query_cache)))

    print("Adaptive tradeoff scans completed.")
    print(f"- json: {json_path}")
    print(f"- hard rate md: {threshold_md_path}")
    print(f"- ce depth md:  {ce_md_path}")


if __name__ == "__main__":
    main()
