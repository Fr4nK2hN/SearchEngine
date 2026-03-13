#!/usr/bin/env python3
"""
Train and evaluate a lightweight easy/hard router from oracle outputs.

Router target:
- easy: experts in {Baseline, LTR}
- hard: experts in {Cross-Encoder, Hybrid}
"""

import argparse
import json
import random
import re
from dataclasses import dataclass

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from",
    "how", "in", "is", "it", "of", "on", "or", "that", "the", "to",
    "was", "were", "what", "when", "where", "which", "who", "why",
    "will", "with", "you", "your",
}
TOKEN_PATTERN = re.compile(r"[a-z0-9]+")


@dataclass
class RouterConfig:
    easy_expert: str
    hard_expert: str


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def tokenize(query):
    return TOKEN_PATTERN.findall((query or "").lower())


def stopword_ratio(tokens):
    if not tokens:
        return 1.0
    return sum(1 for t in tokens if t in STOPWORDS) / len(tokens)


def is_prefix_like(tokens):
    if not tokens:
        return True
    if len(tokens) <= 2:
        return True
    if len(tokens) <= 3 and tokens[-1] in STOPWORDS:
        return True
    return False


def query_features(query):
    tokens = tokenize(query)
    uniq = set(tokens)
    return [
        len(query or ""),
        len(tokens),
        len(uniq),
        stopword_ratio(tokens),
        1.0 if is_prefix_like(tokens) else 0.0,
        (len(uniq) / len(tokens)) if tokens else 0.0,
    ]


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
        cleaned.append({"query": " ".join(q.strip().split()), "documents": docs[:n], "relevance_labels": labels[:n]})
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
        tokens = tokenize(query)
        labels = item["relevance_labels"]
        if min_query_terms > 0 and len(tokens) < min_query_terms:
            continue
        if min_query_chars > 0 and len(query) < min_query_chars:
            continue
        if max_stopword_ratio < 1.0 and stopword_ratio(tokens) > max_stopword_ratio:
            continue
        if drop_prefix_queries and is_prefix_like(tokens):
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


def choose_router_experts(selection_labels):
    easy_candidates = [e for e in selection_labels if e in ("Baseline", "LTR")]
    hard_candidates = [e for e in selection_labels if e in ("Cross-Encoder", "Hybrid")]

    easy_expert = "Baseline"
    hard_expert = "Cross-Encoder"
    if easy_candidates:
        easy_expert = max(set(easy_candidates), key=easy_candidates.count)
    if hard_candidates:
        hard_expert = max(set(hard_candidates), key=hard_candidates.count)
    return RouterConfig(easy_expert=easy_expert, hard_expert=hard_expert)


def main():
    parser = argparse.ArgumentParser(description="Train lightweight router baseline")
    parser.add_argument("--oracle-results", required=True)
    parser.add_argument("--oracle-per-query", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--test-size", type=float, default=0.3)
    args = parser.parse_args()

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

    y = np.array([0 if s in ("Baseline", "LTR") else 1 for s in selected], dtype=int)  # 0=easy,1=hard
    X = np.array([query_features(q) for q in queries], dtype=float)

    X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
        X,
        y,
        np.arange(n),
        test_size=args.test_size,
        random_state=args.seed,
        stratify=y if len(set(y)) > 1 else None,
    )

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    clf = LogisticRegression(random_state=args.seed, max_iter=1000, class_weight="balanced")
    clf.fit(X_train_s, y_train)
    y_pred = clf.predict(X_test_s)

    router_cfg = choose_router_experts([selected[i] for i in idx_train.tolist()])

    def ndcg_for_query(qidx, expert_name):
        return float(experts[expert_name][qidx]["ndcg"])

    routed_ndcg = []
    oracle_ndcg = []
    best_single_name = results["oracle"]["best_single_expert"]
    best_single_ndcg = []
    for qi, pred in zip(idx_test.tolist(), y_pred.tolist()):
        chosen = router_cfg.easy_expert if pred == 0 else router_cfg.hard_expert
        routed_ndcg.append(ndcg_for_query(qi, chosen))
        oracle_ndcg.append(max(float(experts[name][qi]["ndcg"]) for name in experts.keys()))
        best_single_ndcg.append(ndcg_for_query(qi, best_single_name))

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
        "router_config": {
            "easy_expert": router_cfg.easy_expert,
            "hard_expert": router_cfg.hard_expert,
        },
        "classification": {
            "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
            "report": classification_report(y_test, y_pred, output_dict=True, zero_division=0),
        },
        "ranking_eval_on_test": {
            "router_ndcg_mean": float(np.mean(routed_ndcg)),
            "best_single_expert": best_single_name,
            "best_single_ndcg_mean": float(np.mean(best_single_ndcg)),
            "oracle_ndcg_mean": float(np.mean(oracle_ndcg)),
            "router_vs_best_single_gain_abs": float(np.mean(routed_ndcg) - np.mean(best_single_ndcg)),
            "router_vs_best_single_gain_rel": (
                float((np.mean(routed_ndcg) - np.mean(best_single_ndcg)) / np.mean(best_single_ndcg))
                if np.mean(best_single_ndcg) > 1e-12
                else 0.0
            ),
        },
    }

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print("Router baseline completed.")
    print(f"- output: {args.output}")
    print(
        f"- router easy/hard experts: {router_cfg.easy_expert}/{router_cfg.hard_expert}"
    )
    print(
        f"- test ndcg (router vs best_single vs oracle): "
        f"{output['ranking_eval_on_test']['router_ndcg_mean']:.4f} / "
        f"{output['ranking_eval_on_test']['best_single_ndcg_mean']:.4f} / "
        f"{output['ranking_eval_on_test']['oracle_ndcg_mean']:.4f}"
    )


if __name__ == "__main__":
    main()
