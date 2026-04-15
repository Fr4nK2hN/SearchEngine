import argparse
import csv
import json
import math
import os
import pickle
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone

import numpy as np
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler


sys.path.insert(
    0,
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
)

from engine.retrieval import search_documents_with_fallback  # noqa: E402
from ranking.query_router import EXTENDED_FEATURE_NAMES, router_feature_vector  # noqa: E402


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


def safe_hard_prob(model, x_scaled):
    proba = model.predict_proba(x_scaled)
    classes = list(getattr(model, "classes_", []))
    if proba.shape[1] == 1:
        return np.ones(proba.shape[0], dtype=float) if classes and int(classes[0]) == 1 else np.zeros(
            proba.shape[0], dtype=float
        )
    if 1 in classes:
        return proba[:, classes.index(1)]
    return proba[:, -1]


def rank_with_mode(app_state, query, base_results, mode, top_n):
    results = [dict(hit) for hit in base_results]
    for idx, hit in enumerate(base_results):
        results[idx]["_source"] = hit.get("_source", {})
    t0 = time.perf_counter()
    ranked, _method, _feature_ms, _inference_ms = app_state.search_pipeline.apply_ranking_mode(
        query,
        results,
        mode,
        rerank_top_n=top_n,
    )
    rank_ms = (time.perf_counter() - t0) * 1000.0
    return [str(hit["_id"]) for hit in ranked[:10]], rank_ms


def build_dataset(app_state, judgments, top_k, hard_top_k, ndcg_margin):
    rows = []
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
        ce_ids, ce_rank_ms = rank_with_mode(app_state, query, base_results, "cross_encoder", hard_top_k)
        baseline_metrics = score_result_ids(base_ids, label_map, top_k)
        ce_metrics = score_result_ids(ce_ids, label_map, top_k)
        hard_gain = ce_metrics["ndcg"] - baseline_metrics["ndcg"]
        label = 1 if hard_gain > ndcg_margin else 0
        rows.append(
            {
                "query": query,
                "features": router_feature_vector(
                    query,
                    results=base_results,
                    feature_names=EXTENDED_FEATURE_NAMES,
                ),
                "label": label,
                "hard_gain": hard_gain,
                "retrieval_ms": retrieval_ms,
                "baseline_ndcg": baseline_metrics["ndcg"],
                "baseline_mrr": baseline_metrics["mrr"],
                "baseline_recall": baseline_metrics["recall"],
                "baseline_latency_ms": retrieval_ms,
                "cross_ndcg": ce_metrics["ndcg"],
                "cross_mrr": ce_metrics["mrr"],
                "cross_recall": ce_metrics["recall"],
                "cross_latency_ms": retrieval_ms + ce_rank_ms,
                "retrieval_strategy": retrieval_strategy,
                "base_top_score": base_results[0].get("_score", 0.0) if base_results else 0.0,
            }
        )
    return rows


def fit_model(x, y, seed):
    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(x)
    if len(set(y.tolist())) <= 1:
        model = DummyClassifier(strategy="most_frequent")
    else:
        model = LogisticRegression(
            random_state=seed,
            max_iter=1000,
            class_weight="balanced",
        )
    model.fit(x_scaled, y)
    return model, scaler


def cross_val_probs(x, y, seed, folds):
    class_counts = Counter(int(v) for v in y.tolist())
    if len(class_counts) <= 1 or min(class_counts.values()) < 2:
        return np.zeros(len(y), dtype=float)
    n_splits = min(int(folds), min(class_counts.values()))
    splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    probs = np.zeros(len(y), dtype=float)
    for train_idx, test_idx in splitter.split(x, y):
        model, scaler = fit_model(x[train_idx], y[train_idx], seed)
        probs[test_idx] = safe_hard_prob(model, scaler.transform(x[test_idx]))
    return probs


def summarize_policy(rows, hard_probs, threshold):
    ndcgs = []
    mrrs = []
    recalls = []
    latencies = []
    selected = Counter()
    for row, prob in zip(rows, hard_probs):
        use_hard = float(prob) >= float(threshold)
        selected["cross_encoder" if use_hard else "baseline"] += 1
        prefix = "cross" if use_hard else "baseline"
        ndcgs.append(float(row[f"{prefix}_ndcg"]))
        mrrs.append(float(row[f"{prefix}_mrr"]))
        recalls.append(float(row[f"{prefix}_recall"]))
        latencies.append(float(row[f"{prefix}_latency_ms"]))
    return {
        "threshold": threshold,
        "ndcg@10": sum(ndcgs) / len(ndcgs),
        "mrr@10": sum(mrrs) / len(mrrs),
        "recall@10": sum(recalls) / len(recalls),
        "latency_avg_ms": sum(latencies) / len(latencies),
        "latency_p95_ms": percentile(latencies, 95),
        "selected_counts": dict(selected),
    }


def choose_threshold(policy_rows, baseline_ndcg, max_latency_ms):
    viable = [
        row
        for row in policy_rows
        if row["ndcg@10"] >= baseline_ndcg and row["latency_avg_ms"] <= max_latency_ms
    ]
    candidates = viable or policy_rows
    return sorted(candidates, key=lambda row: (-row["ndcg@10"], row["latency_avg_ms"]))[0]


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
    parser.add_argument("--hard-top-k", type=int, default=10)
    parser.add_argument("--ndcg-margin", type=float, default=0.0)
    parser.add_argument("--thresholds", default="0.10,0.20,0.30,0.40,0.50,0.60,0.70,0.80,0.90")
    parser.add_argument("--max-latency-ms", type=float, default=120.0)
    parser.add_argument("--cv-folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out-dir", default="reports/manual_relevance_pool_20260411/retrieval_router_training")
    parser.add_argument("--model-output", default="models/query_router_retrieval.pkl")
    args = parser.parse_args()

    from app import app_state  # noqa: E402

    judgments = load_judgments(args.judgments, args.label_column, args.min_labels_per_query)
    if not judgments:
        raise RuntimeError("no usable manual judgments")
    os.makedirs(args.out_dir, exist_ok=True)

    rows = build_dataset(
        app_state,
        judgments,
        top_k=args.top_k,
        hard_top_k=args.hard_top_k,
        ndcg_margin=args.ndcg_margin,
    )
    x = np.array([row["features"] for row in rows], dtype=float)
    y = np.array([row["label"] for row in rows], dtype=int)
    hard_probs_cv = cross_val_probs(x, y, args.seed, args.cv_folds)
    thresholds = [float(item.strip()) for item in args.thresholds.split(",") if item.strip()]

    baseline_summary = summarize_policy(rows, np.zeros(len(rows)), threshold=1.0)
    cross_summary = summarize_policy(rows, np.ones(len(rows)), threshold=0.0)
    policy_rows = [summarize_policy(rows, hard_probs_cv, threshold) for threshold in thresholds]
    recommendation = choose_threshold(
        policy_rows,
        baseline_ndcg=baseline_summary["ndcg@10"],
        max_latency_ms=args.max_latency_ms,
    )

    model, scaler = fit_model(x, y, args.seed)
    payload = {
        "model": model,
        "scaler": scaler,
        "feature_names": EXTENDED_FEATURE_NAMES,
        "hard_threshold": recommendation["threshold"],
        "hard_top_k": int(args.hard_top_k),
        "hard_topk_policy": [],
        "easy_mode": "baseline",
        "hard_mode": "cross_encoder",
        "meta": {
            "source": args.judgments,
            "label_column": args.label_column,
            "samples_total": len(rows),
            "label_balance": dict(Counter(int(v) for v in y.tolist())),
            "ndcg_margin": args.ndcg_margin,
            "recommended_threshold": recommendation,
            "baseline_summary": baseline_summary,
            "cross_encoder_summary": cross_summary,
            "feature_names": EXTENDED_FEATURE_NAMES,
            "fit_on_full_data": True,
            "generated_at": datetime.now(timezone.utc).isoformat(),
        },
    }
    with open(args.model_output, "wb") as handle:
        pickle.dump(payload, handle)

    per_query_rows = []
    for row, prob in zip(rows, hard_probs_cv):
        out = {
            "query": row["query"],
            "label": row["label"],
            "hard_prob_cv": prob,
            "hard_gain": row["hard_gain"],
            "baseline_ndcg": row["baseline_ndcg"],
            "cross_ndcg": row["cross_ndcg"],
            "baseline_latency_ms": row["baseline_latency_ms"],
            "cross_latency_ms": row["cross_latency_ms"],
            "base_top_score": row["base_top_score"],
        }
        per_query_rows.append(out)

    summary = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "model_output": args.model_output,
        "feature_names": EXTENDED_FEATURE_NAMES,
        "label_balance": dict(Counter(int(v) for v in y.tolist())),
        "baseline_summary": baseline_summary,
        "cross_encoder_summary": cross_summary,
        "policy_scan_cv": policy_rows,
        "recommendation": recommendation,
        "classification_cv": {
            "confusion_matrix": confusion_matrix(y, (hard_probs_cv >= recommendation["threshold"]).astype(int), labels=[0, 1]).tolist(),
            "report": classification_report(
                y,
                (hard_probs_cv >= recommendation["threshold"]).astype(int),
                labels=[0, 1],
                target_names=["easy", "hard"],
                output_dict=True,
                zero_division=0,
            ),
            "auc": float(roc_auc_score(y, hard_probs_cv)) if len(set(y.tolist())) > 1 else None,
        },
    }
    with open(os.path.join(args.out_dir, "summary.json"), "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)
    write_csv(os.path.join(args.out_dir, "policy_scan_cv.csv"), [
        {
            **{k: v for k, v in row.items() if k != "selected_counts"},
            "selected_counts": json.dumps(row["selected_counts"], ensure_ascii=False),
        }
        for row in policy_rows
    ])
    write_csv(os.path.join(args.out_dir, "per_query_training.csv"), per_query_rows)

    print(f"model: {args.model_output}")
    print(f"summary: {os.path.join(args.out_dir, 'summary.json')}")
    print(f"label_balance: {summary['label_balance']}")
    print(
        "baseline: "
        f"ndcg={baseline_summary['ndcg@10']:.4f}, avg_ms={baseline_summary['latency_avg_ms']:.2f}"
    )
    print(
        "cross_encoder@top_k: "
        f"ndcg={cross_summary['ndcg@10']:.4f}, avg_ms={cross_summary['latency_avg_ms']:.2f}"
    )
    print(
        "recommended_cv: "
        f"threshold={recommendation['threshold']:.2f}, ndcg={recommendation['ndcg@10']:.4f}, "
        f"avg_ms={recommendation['latency_avg_ms']:.2f}, selected={recommendation['selected_counts']}"
    )


if __name__ == "__main__":
    main()
