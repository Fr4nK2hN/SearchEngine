#!/usr/bin/env python3
"""
Train and evaluate a lightweight easy/hard router from oracle outputs.

Key update:
- Supports mode-aligned labels (`mode_ndcg`) so router target matches runtime policy
  (e.g., easy=ltr, hard=hybrid) instead of mixed expert-pool labels.
"""

import argparse
import json
import os
import pickle
import random
import sys
from dataclasses import dataclass

import numpy as np
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ranking.query_router import (
    FEATURE_NAMES,
    expert_to_mode,
    is_prefix_like_query,
    parse_hard_topk_policy,
    query_feature_vector,
    stopword_ratio,
    tokenize_query,
)


MODE_TO_EXPERT = {
    "baseline": "Baseline",
    "ltr": "LTR",
    "cross_encoder": "Cross-Encoder",
    "hybrid": "Hybrid",
}


@dataclass
class RouterConfig:
    easy_expert: str
    hard_expert: str
    easy_mode: str
    hard_mode: str
    hard_threshold: float
    hard_top_k: int
    hard_topk_policy: list
    label_policy: str
    ndcg_margin: float


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def sanitize_dataset(data):
    cleaned = []
    for item in data:
        if not isinstance(item, dict):
            continue
        q = item.get("query")
        docs = item.get("documents")
        labels = item.get("relevance_labels")
        if not isinstance(q, str) or not isinstance(docs, list) or not isinstance(labels, list):
            continue
        n = min(len(docs), len(labels))
        if n < 2:
            continue
        cleaned.append(
            {
                "query": " ".join(q.strip().split()),
                "documents": docs[:n],
                "relevance_labels": labels[:n],
            }
        )
    return cleaned


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


def filter_data_quality(data, quality_args):
    min_query_terms = int(quality_args.get("min_query_terms", 0))
    min_query_chars = int(quality_args.get("min_query_chars", 0))
    max_stopword_ratio = float(quality_args.get("max_stopword_ratio", 1.0))
    drop_prefix_queries = bool(quality_args.get("drop_prefix_queries", False))
    min_relevant_docs = int(quality_args.get("min_relevant_docs", 0))

    kept = []
    for item in data:
        query = item["query"]
        tokens = tokenize_query(query)
        labels = item["relevance_labels"]
        if min_query_terms > 0 and len(tokens) < min_query_terms:
            continue
        if min_query_chars > 0 and len(query) < min_query_chars:
            continue
        if max_stopword_ratio < 1.0 and stopword_ratio(tokens) > max_stopword_ratio:
            continue
        if drop_prefix_queries and is_prefix_like_query(tokens):
            continue
        if min_relevant_docs > 0 and sum(1 for x in labels if x > 0) < min_relevant_docs:
            continue
        kept.append(item)
    return kept


def reconstruct_queries(oracle_results, dataset):
    sampled = sample_data(
        dataset,
        oracle_results.get("sample_method", "random"),
        oracle_results.get("max_queries"),
        oracle_results.get("seed", 42),
    )
    quality_args = oracle_results.get("quality_filter_args", {})
    sampled = filter_data_quality(sampled, quality_args) if quality_args else sampled
    return [x["query"] for x in sampled]


def choose_experts(selection_labels):
    easy_candidates = [e for e in selection_labels if e in ("Baseline", "LTR")]
    hard_candidates = [e for e in selection_labels if e in ("Cross-Encoder", "Hybrid")]
    easy_expert = max(set(easy_candidates), key=easy_candidates.count) if easy_candidates else "Baseline"
    hard_expert = max(set(hard_candidates), key=hard_candidates.count) if hard_candidates else "Cross-Encoder"
    return easy_expert, hard_expert


def resolve_mode(mode_arg, fallback_expert):
    if mode_arg == "auto":
        return expert_to_mode(fallback_expert, "baseline")
    return mode_arg


def mode_to_expert_name(mode):
    return MODE_TO_EXPERT.get(mode)


def expert_metric(experts, expert_name, qidx, metric_name):
    return float(experts[expert_name][qidx][metric_name])


def build_targets(selected, experts, policy, easy_mode, hard_mode, ndcg_margin):
    n = len(selected)
    if policy == "expert_pool":
        y = np.array([0 if s in ("Baseline", "LTR") else 1 for s in selected], dtype=int)
        return y, {
            "policy": "expert_pool",
            "description": "0=Baseline/LTR, 1=Cross-Encoder/Hybrid",
            "ties_by_latency": 0,
        }

    easy_expert = mode_to_expert_name(easy_mode)
    hard_expert = mode_to_expert_name(hard_mode)
    if easy_expert not in experts or hard_expert not in experts:
        raise RuntimeError(
            f"mode_ndcg requires experts for easy={easy_mode}({easy_expert}) and "
            f"hard={hard_mode}({hard_expert})"
        )

    labels = []
    ties_by_latency = 0
    for i in range(n):
        easy_ndcg = expert_metric(experts, easy_expert, i, "ndcg")
        hard_ndcg = expert_metric(experts, hard_expert, i, "ndcg")
        if hard_ndcg > easy_ndcg + ndcg_margin:
            labels.append(1)
            continue
        if easy_ndcg > hard_ndcg + ndcg_margin:
            labels.append(0)
            continue
        ties_by_latency += 1
        easy_lat = expert_metric(experts, easy_expert, i, "latency_ms")
        hard_lat = expert_metric(experts, hard_expert, i, "latency_ms")
        labels.append(1 if hard_lat < easy_lat else 0)

    return np.array(labels, dtype=int), {
        "policy": "mode_ndcg",
        "description": (
            f"0={easy_mode}, 1={hard_mode}; margin={ndcg_margin:.4f}, "
            "tie-break by lower latency"
        ),
        "easy_mode_expert": easy_expert,
        "hard_mode_expert": hard_expert,
        "ties_by_latency": ties_by_latency,
    }


def safe_hard_prob(model, x_scaled):
    proba = model.predict_proba(x_scaled)
    classes = list(getattr(model, "classes_", []))
    if proba.shape[1] == 1:
        return np.ones(proba.shape[0], dtype=float) if classes and int(classes[0]) == 1 else np.zeros(
            proba.shape[0], dtype=float
        )
    if 1 in classes:
        pos_idx = classes.index(1)
        return proba[:, pos_idx]
    return proba[:, -1]


def mean(values):
    return float(np.mean(values)) if values else 0.0


def parse_args():
    parser = argparse.ArgumentParser(description="Train lightweight router baseline")
    parser.add_argument("--oracle-results", required=True)
    parser.add_argument("--oracle-per-query", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--model-output", default="", help="Optional .pkl path for trained router model")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--test-size", type=float, default=0.3)
    parser.add_argument("--hard-threshold", type=float, default=0.5)
    parser.add_argument("--hard-top-k", type=int, default=30)
    parser.add_argument(
        "--hard-topk-policy",
        default="",
        help='Optional dynamic hard top-k policy, e.g. "0.08:30,0.10:20,1.00:30"',
    )
    parser.add_argument(
        "--easy-mode",
        choices=["auto", "baseline", "ltr", "cross_encoder", "hybrid"],
        default="ltr",
    )
    parser.add_argument(
        "--hard-mode",
        choices=["auto", "baseline", "ltr", "cross_encoder", "hybrid"],
        default="hybrid",
    )
    parser.add_argument(
        "--label-policy",
        choices=["mode_ndcg", "expert_pool"],
        default="mode_ndcg",
        help="mode_ndcg aligns labels with runtime easy/hard modes",
    )
    parser.add_argument(
        "--ndcg-margin",
        type=float,
        default=0.01,
        help="Treat differences within margin as tie; tie broken by lower latency",
    )
    parser.add_argument(
        "--fit-on-full-data",
        action="store_true",
        help="Refit scaler/model on all samples before exporting model_output",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)

    results = load_json(args.oracle_results)
    per_query = load_json(args.oracle_per_query)
    data = sanitize_dataset(load_json(args.data))

    queries = reconstruct_queries(results, data)
    selected = per_query.get("selected_expert", [])
    experts = per_query.get("experts", {})

    n = min(len(queries), len(selected))
    queries = queries[:n]
    selected = selected[:n]
    if n < 8:
        raise RuntimeError(f"not enough samples for router: {n}")
    if not experts:
        raise RuntimeError("oracle_per_query.experts is empty")

    fallback_easy_expert, fallback_hard_expert = choose_experts(selected)
    easy_mode = resolve_mode(args.easy_mode, fallback_easy_expert)
    hard_mode = resolve_mode(args.hard_mode, fallback_hard_expert)
    easy_expert = mode_to_expert_name(easy_mode) or fallback_easy_expert
    hard_expert = mode_to_expert_name(hard_mode) or fallback_hard_expert

    y, target_meta = build_targets(
        selected=selected,
        experts=experts,
        policy=args.label_policy,
        easy_mode=easy_mode,
        hard_mode=hard_mode,
        ndcg_margin=float(args.ndcg_margin),
    )
    X = np.array([query_feature_vector(q) for q in queries], dtype=float)

    uniq, counts = np.unique(y, return_counts=True)
    class_count = {int(k): int(v) for k, v in zip(uniq.tolist(), counts.tolist())}
    can_stratify = len(uniq) > 1 and min(class_count.values()) >= 2
    stratify = y if can_stratify else None

    X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
        X,
        y,
        np.arange(n),
        test_size=args.test_size,
        random_state=args.seed,
        stratify=stratify,
    )

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    if len(np.unique(y_train)) <= 1:
        clf = DummyClassifier(strategy="most_frequent")
    else:
        clf = LogisticRegression(random_state=args.seed, max_iter=1000, class_weight="balanced")
    clf.fit(X_train_s, y_train)

    hard_prob = safe_hard_prob(clf, X_test_s)
    y_pred = (hard_prob >= float(args.hard_threshold)).astype(int)

    def ndcg_for_query(qidx, expert_name):
        return expert_metric(experts, expert_name, qidx, "ndcg")

    def latency_for_query(qidx, expert_name):
        return expert_metric(experts, expert_name, qidx, "latency_ms")

    routed_ndcg = []
    routed_latency = []
    pair_oracle_ndcg = []
    global_oracle_ndcg = []
    easy_ndcg_on_test = []
    hard_ndcg_on_test = []
    best_single_name = results.get("oracle", {}).get("best_single_expert", "")
    best_single_ndcg = []

    all_expert_names = list(experts.keys())
    if easy_expert not in all_expert_names or hard_expert not in all_expert_names:
        raise RuntimeError(
            f"resolved experts missing from oracle_per_query.experts: easy={easy_expert}, hard={hard_expert}"
        )

    for qi, pred in zip(idx_test.tolist(), y_pred.tolist()):
        chosen = hard_expert if pred == 1 else easy_expert
        routed_ndcg.append(ndcg_for_query(qi, chosen))
        routed_latency.append(latency_for_query(qi, chosen))
        easy_v = ndcg_for_query(qi, easy_expert)
        hard_v = ndcg_for_query(qi, hard_expert)
        easy_ndcg_on_test.append(easy_v)
        hard_ndcg_on_test.append(hard_v)
        pair_oracle_ndcg.append(max(easy_v, hard_v))
        global_oracle_ndcg.append(max(ndcg_for_query(qi, name) for name in all_expert_names))
        if best_single_name in experts:
            best_single_ndcg.append(ndcg_for_query(qi, best_single_name))

    policy_best_name = easy_expert if mean(easy_ndcg_on_test) >= mean(hard_ndcg_on_test) else hard_expert
    policy_best_ndcg = max(mean(easy_ndcg_on_test), mean(hard_ndcg_on_test))

    auc = None
    if len(np.unique(y_test)) > 1:
        auc = float(roc_auc_score(y_test, hard_prob))

    router_cfg = RouterConfig(
        easy_expert=easy_expert,
        hard_expert=hard_expert,
        easy_mode=easy_mode,
        hard_mode=hard_mode,
        hard_threshold=float(args.hard_threshold),
        hard_top_k=max(1, int(args.hard_top_k)),
        hard_topk_policy=parse_hard_topk_policy(args.hard_topk_policy),
        label_policy=args.label_policy,
        ndcg_margin=float(args.ndcg_margin),
    )

    output = {
        "generated_at": results.get("generated_at"),
        "source": {
            "oracle_results": args.oracle_results,
            "oracle_per_query": args.oracle_per_query,
            "data": args.data,
        },
        "samples_total": n,
        "train_size": len(idx_train),
        "test_size": len(idx_test),
        "class_balance": {
            "easy": int((y == 0).sum()),
            "hard": int((y == 1).sum()),
        },
        "target_meta": target_meta,
        "router_config": {
            "easy_expert": router_cfg.easy_expert,
            "hard_expert": router_cfg.hard_expert,
            "easy_mode": router_cfg.easy_mode,
            "hard_mode": router_cfg.hard_mode,
            "hard_threshold": router_cfg.hard_threshold,
            "hard_top_k": router_cfg.hard_top_k,
            "hard_topk_policy": router_cfg.hard_topk_policy,
            "label_policy": router_cfg.label_policy,
            "ndcg_margin": router_cfg.ndcg_margin,
        },
        "classification": {
            "confusion_matrix": confusion_matrix(y_test, y_pred, labels=[0, 1]).tolist(),
            "report": classification_report(
                y_test,
                y_pred,
                labels=[0, 1],
                target_names=["easy", "hard"],
                output_dict=True,
                zero_division=0,
            ),
            "auc": auc,
            "hard_rate_pred": float(np.mean(y_pred)),
            "hard_prob_mean": float(np.mean(hard_prob)),
        },
        "ranking_eval_on_test": {
            "router_ndcg_mean": mean(routed_ndcg),
            "router_latency_ms_mean": mean(routed_latency),
            "policy_best_single_expert": policy_best_name,
            "policy_best_single_ndcg_mean": float(policy_best_ndcg),
            "policy_pair_oracle_ndcg_mean": mean(pair_oracle_ndcg),
            "global_oracle_ndcg_mean": mean(global_oracle_ndcg),
            "best_single_expert_global": best_single_name,
            "best_single_ndcg_mean_global": mean(best_single_ndcg),
            "router_vs_policy_best_gain_abs": mean(routed_ndcg) - float(policy_best_ndcg),
            "router_vs_pair_oracle_gap_abs": mean(routed_ndcg) - mean(pair_oracle_ndcg),
            "router_vs_global_best_gain_abs": mean(routed_ndcg) - mean(best_single_ndcg),
        },
    }

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    if args.model_output:
        export_scaler = scaler
        export_model = clf
        if args.fit_on_full_data:
            export_scaler = StandardScaler()
            X_all_s = export_scaler.fit_transform(X)
            if len(np.unique(y)) <= 1:
                export_model = DummyClassifier(strategy="most_frequent")
            else:
                export_model = LogisticRegression(random_state=args.seed, max_iter=1000, class_weight="balanced")
            export_model.fit(X_all_s, y)

        payload = {
            "model": export_model,
            "scaler": export_scaler,
            "feature_names": FEATURE_NAMES,
            "hard_threshold": router_cfg.hard_threshold,
            "hard_top_k": router_cfg.hard_top_k,
            "hard_topk_policy": router_cfg.hard_topk_policy,
            "easy_mode": router_cfg.easy_mode,
            "hard_mode": router_cfg.hard_mode,
            "meta": {
                "router_config": output["router_config"],
                "samples_total": n,
                "seed": args.seed,
                "test_size": args.test_size,
                "source": output["source"],
                "target_meta": target_meta,
                "fit_on_full_data": bool(args.fit_on_full_data),
            },
        }
        os.makedirs(os.path.dirname(args.model_output) or ".", exist_ok=True)
        with open(args.model_output, "wb") as f:
            pickle.dump(payload, f)
        output["model_output"] = args.model_output
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(output, f, indent=2, ensure_ascii=False)

    print("Router baseline completed.")
    print(f"- output: {args.output}")
    if args.model_output:
        print(f"- model: {args.model_output}")
    print(
        f"- target policy: {args.label_policy} "
        f"(easy_mode={easy_mode}, hard_mode={hard_mode}, ndcg_margin={args.ndcg_margin:.4f})"
    )
    print(
        f"- router modes: {router_cfg.easy_mode}/{router_cfg.hard_mode} "
        f"(threshold={router_cfg.hard_threshold:.2f}, hard_top_k={router_cfg.hard_top_k}, "
        f"hard_topk_policy={router_cfg.hard_topk_policy})"
    )
    print(
        f"- test ndcg (router / policy_best / pair_oracle / global_best): "
        f"{output['ranking_eval_on_test']['router_ndcg_mean']:.4f} / "
        f"{output['ranking_eval_on_test']['policy_best_single_ndcg_mean']:.4f} / "
        f"{output['ranking_eval_on_test']['policy_pair_oracle_ndcg_mean']:.4f} / "
        f"{output['ranking_eval_on_test']['best_single_ndcg_mean_global']:.4f}"
    )


if __name__ == "__main__":
    main()
