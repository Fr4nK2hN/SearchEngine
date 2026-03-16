#!/usr/bin/env python3
"""
Evaluate the current adaptive system with configurable guardrail subsets.

This script reuses the live ES retrieval + current router/reranker stack, then
scores the returned ranking against sparse feedback labels built from
`data/ltr_training_data_feedback_combined.json`.
"""

import argparse
import copy
import json
import math
import os
import sys
import time
from collections import Counter
from datetime import datetime, timezone

sys.path.insert(
    0,
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
)

from app import (  # noqa: E402
    ADAPTIVE_HARD_TOP_K_CAP,
    RECALL_RELAX_THRESHOLD,
    _apply_ranking_mode,
    _is_ltr_available,
    es,
    query_router,
)
from ranking.query_router import adaptive_guardrail  # noqa: E402
from engine.retrieval import search_documents_with_fallback  # noqa: E402


ALL_GUARDRAILS = [
    "topical_easy_ltr",
    "hard_question_prefix_baseline",
    "hard_long_mix_ltr",
]


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


def apply_guardrail_subset(query, route, enabled_guardrails):
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


def resolve_route(query, enabled_guardrails):
    route = apply_guardrail_subset(query, query_router.route(query), enabled_guardrails)
    selected_mode = route.get("selected_mode") or "baseline"
    route_label = route.get("route_label", "easy")
    hard_top_k = _to_positive_int(route.get("hard_top_k")) or 30
    if ADAPTIVE_HARD_TOP_K_CAP is not None:
        hard_top_k = min(hard_top_k, ADAPTIVE_HARD_TOP_K_CAP)

    if selected_mode in ("ltr", "hybrid") and not _is_ltr_available():
        selected_mode = "cross_encoder" if route_label == "hard" else "baseline"

    route["selected_mode"] = selected_mode
    route["hard_top_k"] = hard_top_k
    route["rerank_top_n"] = (
        hard_top_k if route_label == "hard" and selected_mode in ("cross_encoder", "hybrid") else None
    )
    return route


def default_configs():
    all_enabled = set(ALL_GUARDRAILS)
    return [
        ("router_only", set()),
        ("all_guardrails", set(all_enabled)),
        ("minus_topical_easy_ltr", all_enabled - {"topical_easy_ltr"}),
        (
            "minus_hard_question_prefix_baseline",
            all_enabled - {"hard_question_prefix_baseline"},
        ),
        ("minus_hard_long_mix_ltr", all_enabled - {"hard_long_mix_ltr"}),
    ]


def run_config_pass(query_cache, enabled_guardrails, top_k, measure_latency):
    ndcgs = []
    mrrs = []
    recalls = []
    latencies = []
    route_labels = []
    selected_modes = []
    guardrail_counts = Counter()
    strategies = Counter()

    for item in query_cache:
        query = item["query"]
        results = copy.deepcopy(item["retrieval_results"])
        route = resolve_route(query, enabled_guardrails=enabled_guardrails)
        exec_mode = route.get("selected_mode", "baseline")
        rerank_top_n = route.get("rerank_top_n")

        rank_t0 = time.perf_counter()
        ranked, _, _, _ = _apply_ranking_mode(
            query,
            results,
            exec_mode,
            rerank_top_n=rerank_top_n,
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
        strategies[str(item["retrieval_strategy"])] += 1

    hard_count = sum(1 for x in route_labels if x == "hard")
    total = max(1, len(route_labels))
    return {
        "ndcg@10": float(sum(ndcgs) / len(ndcgs)),
        "mrr@10": float(sum(mrrs) / len(mrrs)),
        "recall@10": float(sum(recalls) / len(recalls)),
        "latency_ms_avg": float(sum(latencies) / len(latencies)) if latencies else 0.0,
        "latency_ms_p95": float(percentile(latencies, 95)) if latencies else 0.0,
        "hard_rate": float(hard_count / total),
        "guardrail_hit_rate": float(sum(guardrail_counts.values()) / total),
        "route_counts": dict(Counter(route_labels)),
        "selected_mode_counts": dict(Counter(selected_modes)),
        "guardrail_counts": dict(guardrail_counts),
        "retrieval_strategy_counts": dict(strategies),
        "enabled_guardrails": sorted(enabled_guardrails),
        "sample_count": len(query_cache),
    }


def build_markdown(results, config_order, dataset_path, query_count):
    lines = [
        f"# Guardrail Ablation ({datetime.now(timezone.utc).date().isoformat()})",
        "",
        f"- Dataset: `{dataset_path}`",
        f"- Queries: `{query_count}`",
        "",
        "| Config | nDCG@10 | MRR@10 | Recall@10 | Avg Latency (ms) | P95 Latency (ms) | Hard Rate | Guardrail Hits |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]

    baseline_ndcg = results["router_only"]["ndcg@10"]
    all_ndcg = results["all_guardrails"]["ndcg@10"]

    for name in config_order:
        item = results[name]
        lines.append(
            f"| {name} | {item['ndcg@10']:.4f} | {item['mrr@10']:.4f} | {item['recall@10']:.4f} | "
            f"{item['latency_ms_avg']:.1f} | {item['latency_ms_p95']:.1f} | "
            f"{item['hard_rate']:.1%} | {item['guardrail_hit_rate']:.1%} |"
        )

    lines.extend(
        [
            "",
            "## Key Deltas",
            "",
            f"- `all_guardrails` vs `router_only`: nDCG@10 `{all_ndcg - baseline_ndcg:+.4f}`",
        ]
    )

    for name in config_order:
        if name in {"router_only", "all_guardrails"}:
            continue
        item = results[name]
        lines.append(
            f"- `all_guardrails` vs `{name}`: nDCG@10 `{all_ndcg - item['ndcg@10']:+.4f}`"
        )

    lines.extend(
        [
            "",
            "## Guardrail Trigger Counts (all_guardrails)",
            "",
        ]
    )
    counts = results["all_guardrails"].get("guardrail_counts", {})
    for guardrail in ALL_GUARDRAILS:
        lines.append(f"- `{guardrail}`: {counts.get(guardrail, 0)}")

    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description="Run current guardrail ablation on live retrieval")
    parser.add_argument("--data", default="data/ltr_training_data_feedback_combined.json")
    parser.add_argument("--output-json", default="reports/guardrail_ablation_20260315.json")
    parser.add_argument("--output-md", default="reports/guardrail_ablation_20260315.md")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--retrieval-size", type=int, default=50)
    parser.add_argument("--warmup", type=int, default=1, help="Number of full warmup passes before measuring")
    args = parser.parse_args()

    data = load_dataset(args.data)
    if not data:
        raise RuntimeError(f"dataset is empty: {args.data}")

    query_cache = []
    for item in data:
        query = item["query"]
        label_map = label_map_from_item(item)
        rel_labels = [int(v) for v in label_map.values() if int(v) > 0]

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
                "label_map": label_map,
                "total_known_relevant_labels": rel_labels,
            }
        )

    configs = default_configs()
    results_by_config = {}

    for _ in range(max(0, int(args.warmup))):
        for _, enabled_guardrails in configs:
            run_config_pass(query_cache, enabled_guardrails, top_k=args.top_k, measure_latency=False)

    for config_name, enabled_guardrails in configs:
        results_by_config[config_name] = run_config_pass(
            query_cache,
            enabled_guardrails,
            top_k=args.top_k,
            measure_latency=True,
        )

    output = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "dataset": args.data,
        "top_k": int(args.top_k),
        "retrieval_size": int(args.retrieval_size),
        "warmup_queries": int(args.warmup),
        "guardrails": list(ALL_GUARDRAILS),
        "results": results_by_config,
    }

    os.makedirs(os.path.dirname(args.output_json) or ".", exist_ok=True)
    os.makedirs(os.path.dirname(args.output_md) or ".", exist_ok=True)
    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    with open(args.output_md, "w", encoding="utf-8") as f:
        f.write(build_markdown(output["results"], [name for name, _ in configs], args.data, len(query_cache)))

    print("Guardrail ablation completed.")
    print(f"- json: {args.output_json}")
    print(f"- md:   {args.output_md}")


if __name__ == "__main__":
    main()
