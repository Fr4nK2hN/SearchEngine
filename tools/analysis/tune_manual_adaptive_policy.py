import argparse
import csv
import json
import math
import os
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone


sys.path.insert(
    0,
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
)

from engine.retrieval import search_documents_with_fallback  # noqa: E402
from ranking.query_router import query_feature_vector  # noqa: E402


def parse_label(value):
    value = str(value or "").strip()
    if value == "":
        return None
    try:
        return max(0, min(3, int(float(value))))
    except ValueError:
        return None


def load_judgments(path, label_column, min_labels_per_query):
    labels = defaultdict(dict)
    with open(path, "r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            query = " ".join(str(row.get("query") or "").split())
            doc_id = str(row.get("doc_id") or "").strip()
            label = parse_label(row.get(label_column))
            if query and doc_id and label is not None:
                labels[query][doc_id] = label
    return {
        query: label_map
        for query, label_map in labels.items()
        if len(label_map) >= min_labels_per_query
    }


def dcg_at_k(labels, k):
    return sum(
        (2.0 ** float(rel) - 1.0) / math.log2(idx + 2.0)
        for idx, rel in enumerate(labels[:k])
    )


def ndcg_at_k(returned_labels, missing_labels, k):
    dcg = dcg_at_k(returned_labels, k)
    ideal = sorted(list(returned_labels) + list(missing_labels), reverse=True)
    idcg = dcg_at_k(ideal, k)
    return (dcg / idcg) if idcg > 0 else 0.0


def mrr_at_k(returned_labels, k):
    for idx, label in enumerate(returned_labels[:k], start=1):
        if label > 0:
            return 1.0 / idx
    return 0.0


def recall_at_k(returned_labels, missing_labels, k):
    all_labels = list(returned_labels) + list(missing_labels)
    total_positive = sum(1 for label in all_labels if label > 0)
    if total_positive <= 0:
        return 0.0
    return sum(1 for label in returned_labels[:k] if label > 0) / total_positive


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
    return arr[lo] * (hi - pos) + arr[hi] * (pos - lo)


def score_result_ids(result_ids, label_map, top_k):
    returned = [int(label_map.get(doc_id, 0)) for doc_id in result_ids[:top_k]]
    returned_set = set(result_ids[:top_k])
    missing = [
        int(label)
        for doc_id, label in label_map.items()
        if int(label) > 0 and doc_id not in returned_set
    ]
    return {
        "ndcg": ndcg_at_k(returned, missing, top_k),
        "mrr": mrr_at_k(returned, top_k),
        "recall": recall_at_k(returned, missing, top_k),
    }


def hard_probability(query_router, query):
    if not getattr(query_router, "loaded", False):
        route = query_router.route(query)
        return float(route.get("route_confidence", 0.0)) if route.get("route_label") == "hard" else 0.0
    feats = [query_feature_vector(query)]
    scaled = query_router.scaler.transform(feats)
    prob = query_router.model.predict_proba(scaled)[0]
    return float(prob[1]) if len(prob) > 1 else float(prob[0])


def eval_mode(app_state, query, base_results, mode, rerank_top_n):
    results = [dict(hit) for hit in base_results]
    for idx, hit in enumerate(base_results):
        results[idx]["_source"] = hit.get("_source", {})
    t0 = time.perf_counter()
    ranked, _method, _feature_ms, _inference_ms = app_state.search_pipeline.apply_ranking_mode(
        query,
        results,
        mode,
        rerank_top_n=rerank_top_n,
    )
    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    return [str(hit["_id"]) for hit in ranked[:10]], elapsed_ms


def build_query_cache(app_state, judgments, top_k, ce_top_ns):
    cache = {}
    for query, label_map in judgments.items():
        t0 = time.perf_counter()
        base_results, retrieval_strategy = search_documents_with_fallback(
            app_state.es,
            query,
            size=50,
            hl=False,
            relax_threshold=app_state.runtime_config.recall_relax_threshold,
            index_name=app_state.runtime_config.index_name,
        )
        retrieval_ms = (time.perf_counter() - t0) * 1000.0
        base_ids = [str(hit["_id"]) for hit in base_results[:top_k]]
        score0 = float(base_results[0].get("_score", 0.0) or 0.0) if base_results else 0.0
        score1 = float(base_results[1].get("_score", 0.0) or 0.0) if len(base_results) > 1 else 0.0
        hard_prob = hard_probability(app_state.query_router, query)
        entry = {
            "query": query,
            "hard_prob": hard_prob,
            "base_top_score": score0,
            "base_score_gap": score0 - score1,
            "base_score_ratio": (score1 / score0) if score0 > 0 else 0.0,
            "retrieval_ms": retrieval_ms,
            "retrieval_strategy": retrieval_strategy,
            "modes": {
                "baseline": {
                    "ids": base_ids,
                    "rank_ms": 0.0,
                    **score_result_ids(base_ids, label_map, top_k),
                }
            },
        }
        for ce_top_n in ce_top_ns:
            ids, rank_ms = eval_mode(app_state, query, base_results, "cross_encoder", ce_top_n)
            entry["modes"][f"cross_encoder@{ce_top_n}"] = {
                "ids": ids,
                "rank_ms": rank_ms,
                **score_result_ids(ids, label_map, top_k),
            }
        cache[query] = entry
    return cache


def summarize_policy(name, cache, threshold, ce_top_n, top_k, min_top_score_for_baseline=None):
    ndcgs = []
    mrrs = []
    recalls = []
    latencies = []
    selected = Counter()
    per_query = []

    ce_key = f"cross_encoder@{ce_top_n}"
    for query, entry in cache.items():
        lexical_confident = (
            min_top_score_for_baseline is not None
            and entry["base_top_score"] >= min_top_score_for_baseline
        )
        use_ce = entry["hard_prob"] >= threshold and not lexical_confident
        mode_key = ce_key if use_ce else "baseline"
        selected["cross_encoder" if use_ce else "baseline"] += 1
        metrics = entry["modes"][mode_key]
        latency = entry["retrieval_ms"] + metrics["rank_ms"]
        ndcgs.append(metrics["ndcg"])
        mrrs.append(metrics["mrr"])
        recalls.append(metrics["recall"])
        latencies.append(latency)
        per_query.append(
            {
                "policy": name,
                "query": query,
                "hard_prob": entry["hard_prob"],
                "base_top_score": entry["base_top_score"],
                "base_score_gap": entry["base_score_gap"],
                "base_score_ratio": entry["base_score_ratio"],
                "selected_mode": "cross_encoder" if use_ce else "baseline",
                "ce_top_n": ce_top_n if use_ce else 0,
                "min_top_score_for_baseline": min_top_score_for_baseline,
                "ndcg@10": metrics["ndcg"],
                "mrr@10": metrics["mrr"],
                "recall@10": metrics["recall"],
                "latency_ms": latency,
            }
        )

    return {
        "policy": name,
        "threshold": threshold,
        "ce_top_n": ce_top_n,
        "min_top_score_for_baseline": min_top_score_for_baseline,
        "query_count": len(cache),
        "ndcg@10": sum(ndcgs) / len(ndcgs),
        "mrr@10": sum(mrrs) / len(mrrs),
        "recall@10": sum(recalls) / len(recalls),
        "latency_avg_ms": sum(latencies) / len(latencies),
        "latency_p50_ms": percentile(latencies, 50),
        "latency_p95_ms": percentile(latencies, 95),
        "selected_counts": dict(selected),
        "per_query": per_query,
    }


def write_csv(path, rows):
    if not rows:
        return
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--judgments", default="reports/manual_relevance_pool_20260411/judgment_pool.csv")
    parser.add_argument("--label-column", default="manual_label")
    parser.add_argument("--min-labels-per-query", type=int, default=10)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--ce-top-n", default="5,10,20,50")
    parser.add_argument("--thresholds", default="0.30,0.40,0.50,0.55,0.60,0.6062,0.65,0.70,0.75,0.80")
    parser.add_argument("--baseline-top-score-thresholds", default="")
    parser.add_argument("--out-dir", default="reports/manual_relevance_pool_20260411/adaptive_policy_tuning")
    args = parser.parse_args()

    from app import app_state  # noqa: E402

    judgments = load_judgments(args.judgments, args.label_column, args.min_labels_per_query)
    if not judgments:
        raise RuntimeError("no usable manual judgments")
    ce_top_ns = [int(x.strip()) for x in args.ce_top_n.split(",") if x.strip()]
    thresholds = [float(x.strip()) for x in args.thresholds.split(",") if x.strip()]
    score_thresholds = [
        float(x.strip())
        for x in args.baseline_top_score_thresholds.split(",")
        if x.strip()
    ]
    os.makedirs(args.out_dir, exist_ok=True)

    cache = build_query_cache(app_state, judgments, args.top_k, ce_top_ns)
    rows = []
    per_query_rows = []

    for ce_top_n in ce_top_ns:
        for threshold in thresholds:
            name = f"baseline_or_ce_t{threshold:g}_k{ce_top_n}"
            summary = summarize_policy(name, cache, threshold, ce_top_n, args.top_k)
            per_query_rows.extend(summary.pop("per_query"))
            rows.append(summary)
            for score_threshold in score_thresholds:
                name = f"baseline_or_ce_t{threshold:g}_k{ce_top_n}_baseScore{score_threshold:g}"
                summary = summarize_policy(
                    name,
                    cache,
                    threshold,
                    ce_top_n,
                    args.top_k,
                    min_top_score_for_baseline=score_threshold,
                )
                per_query_rows.extend(summary.pop("per_query"))
                rows.append(summary)

    rows = sorted(rows, key=lambda row: (-row["ndcg@10"], row["latency_avg_ms"]))
    csv_rows = [
        {
            **{k: v for k, v in row.items() if k != "selected_counts"},
            "selected_counts": json.dumps(row["selected_counts"], ensure_ascii=False),
        }
        for row in rows
    ]
    write_csv(os.path.join(args.out_dir, "policy_scan.csv"), csv_rows)
    write_csv(os.path.join(args.out_dir, "policy_per_query.csv"), per_query_rows)
    with open(os.path.join(args.out_dir, "policy_scan.json"), "w", encoding="utf-8") as handle:
        json.dump(
            {
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "rows": rows,
            },
            handle,
            indent=2,
            ensure_ascii=False,
        )

    print(f"output_dir: {args.out_dir}")
    for row in rows[:12]:
        print(
            f"{row['policy']}: ndcg@10={row['ndcg@10']:.4f}, "
            f"avg_ms={row['latency_avg_ms']:.2f}, p95={row['latency_p95_ms']:.2f}, "
            f"selected={row['selected_counts']}"
        )


if __name__ == "__main__":
    main()
