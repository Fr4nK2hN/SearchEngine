#!/usr/bin/env python3
"""
Oracle upper-bound experiment for adaptive routing.

For each query, compare experts:
- Baseline (ES score)
- LTR
- Cross-Encoder
- Hybrid (LTR + CE on top-N)

Also includes:
- query/data quality diagnostics
- optional query-quality filtering
- LTR model-data consistency checks and optional auto-retrain
"""

import argparse
import hashlib
import json
import os
import random
import re
import sys
import time
from collections import Counter
from datetime import datetime, timezone

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ranking.evaluator import RankingEvaluator
from ranking.feature_extractor import FeatureExtractor
from ranking.ranker import LTRRanker


STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from",
    "how", "in", "is", "it", "of", "on", "or", "that", "the", "to",
    "was", "were", "what", "when", "where", "which", "who", "why",
    "will", "with", "you", "your",
}
TOKEN_PATTERN = re.compile(r"[a-z0-9]+")


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


def percentile(values, p):
    if not values:
        return 0.0
    arr = np.array(values, dtype=float)
    return float(np.percentile(arr, p))


def hash_json_obj(obj):
    payload = json.dumps(obj, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def tokenize_query(query):
    return TOKEN_PATTERN.findall((query or "").lower())


def stopword_ratio(tokens):
    if not tokens:
        return 1.0
    return sum(1 for t in tokens if t in STOPWORDS) / len(tokens)


def is_prefix_like_query(tokens):
    if not tokens:
        return True
    if len(tokens) <= 3 and tokens[-1] in STOPWORDS:
        return True
    if len(tokens) <= 2:
        return True
    return False


def sanitize_dataset(data):
    cleaned = []
    dropped = 0
    for item in data:
        if not isinstance(item, dict):
            dropped += 1
            continue
        query = item.get("query")
        documents = item.get("documents")
        labels = item.get("relevance_labels")
        if not isinstance(query, str) or not isinstance(documents, list) or not isinstance(labels, list):
            dropped += 1
            continue
        n = min(len(documents), len(labels))
        if n < 2:
            dropped += 1
            continue
        cleaned.append(
            {
                "query": " ".join(query.strip().split()),
                "documents": documents[:n],
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


def filter_data_quality(
    data,
    min_query_terms=0,
    min_query_chars=0,
    max_stopword_ratio=1.0,
    drop_prefix_queries=False,
    min_relevant_docs=0,
):
    kept = []
    dropped = []
    for item in data:
        query = item["query"]
        labels = item["relevance_labels"]
        tokens = tokenize_query(query)

        reasons = []
        if min_query_terms > 0 and len(tokens) < min_query_terms:
            reasons.append(f"terms<{min_query_terms}")
        if min_query_chars > 0 and len(query) < min_query_chars:
            reasons.append(f"chars<{min_query_chars}")
        ratio = stopword_ratio(tokens)
        if max_stopword_ratio < 1.0 and ratio > max_stopword_ratio:
            reasons.append(f"stop_ratio>{max_stopword_ratio}")
        if drop_prefix_queries and is_prefix_like_query(tokens):
            reasons.append("prefix_like")
        if min_relevant_docs > 0:
            rel_count = sum(1 for x in labels if x > 0)
            if rel_count < min_relevant_docs:
                reasons.append(f"relevant<{min_relevant_docs}")

        if reasons:
            dropped.append({"query": query, "reasons": reasons})
        else:
            kept.append(item)

    return kept, dropped


def build_data_profile(data):
    profile = {
        "query_count": len(data),
        "query_terms_avg": 0.0,
        "query_terms_p95": 0.0,
        "query_stopword_ratio_avg": 0.0,
        "short_query_ratio": 0.0,
        "prefix_like_ratio": 0.0,
        "relevant_docs_avg": 0.0,
        "single_relevant_ratio": 0.0,
        "first_doc_is_max_ratio": 0.0,
        "label_distribution": {},
    }
    warnings = []

    if not data:
        warnings.append("Dataset is empty after filtering.")
        return profile, warnings

    term_counts = []
    stop_ratios = []
    relevant_counts = []
    prefix_like = 0
    short_query = 0
    first_doc_is_max = 0
    label_counter = Counter()

    for item in data:
        query = item["query"]
        labels = item["relevance_labels"]
        tokens = tokenize_query(query)
        term_counts.append(len(tokens))
        ratio = stopword_ratio(tokens)
        stop_ratios.append(ratio)

        if len(tokens) < 3:
            short_query += 1
        if is_prefix_like_query(tokens):
            prefix_like += 1

        rel_count = sum(1 for x in labels if x > 0)
        relevant_counts.append(rel_count)
        label_counter.update(labels)
        if labels and labels[0] == max(labels):
            first_doc_is_max += 1

    n = len(data)
    profile.update(
        {
            "query_terms_avg": float(np.mean(term_counts)),
            "query_terms_p95": percentile(term_counts, 95),
            "query_stopword_ratio_avg": float(np.mean(stop_ratios)),
            "short_query_ratio": short_query / n,
            "prefix_like_ratio": prefix_like / n,
            "relevant_docs_avg": float(np.mean(relevant_counts)),
            "single_relevant_ratio": sum(1 for x in relevant_counts if x <= 1) / n,
            "first_doc_is_max_ratio": first_doc_is_max / n,
            "label_distribution": dict(sorted(label_counter.items())),
        }
    )

    if n < 30:
        warnings.append(f"Query count is only {n}; oracle variance is likely high.")
    if profile["short_query_ratio"] > 0.30:
        warnings.append(
            f"Short-query ratio is {profile['short_query_ratio']:.1%}; lexical baseline may be over-favored."
        )
    if profile["prefix_like_ratio"] > 0.25:
        warnings.append(
            f"Prefix-like query ratio is {profile['prefix_like_ratio']:.1%}; evaluation may not reflect real search intent."
        )
    if profile["single_relevant_ratio"] > 0.60:
        warnings.append(
            f"{profile['single_relevant_ratio']:.1%} queries have <=1 relevant doc; ranking gains are hard to observe."
        )
    if profile["first_doc_is_max_ratio"] > 0.70:
        warnings.append(
            f"{profile['first_doc_is_max_ratio']:.1%} queries already have top doc as max label; baseline ceiling is inflated."
        )
    return profile, warnings


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
        idx_sorted = sorted(range(len(documents)), key=lambda i: ltr_scores[i], reverse=True)
        top_idx = idx_sorted[: self.top_n]
        top_docs = [documents[i] for i in top_idx]
        cross_scores = self.cross.predict(query, top_docs)
        final_scores = list(ltr_scores)
        for j, i in enumerate(top_idx):
            final_scores[i] = 0.6 * float(ltr_scores[i]) + 0.4 * float(cross_scores[j])
        return final_scores


def evaluate_expert(expert, data, evaluator, ndcg_k=10, mrr_k=10, recall_k=50):
    rows = []
    for item in data:
        query = item["query"]
        docs = item["documents"]
        labels = item["relevance_labels"]
        t0 = time.perf_counter()
        scores = expert.predict(query, docs)
        t1 = time.perf_counter()
        ranks = np.argsort(np.array(scores))[::-1]
        ndcg = evaluator.ndcg_at_k(labels, ranks, k=ndcg_k)
        mrr = evaluator.mrr_score(labels, ranks[:mrr_k])
        recall = evaluator.recall_at_k(labels, ranks, k=recall_k)
        rows.append(
            {
                "ndcg": float(ndcg),
                "mrr": float(mrr),
                "recall": float(recall),
                "latency_ms": float((t1 - t0) * 1000.0),
            }
        )
    return rows


def aggregate_rows(rows):
    if not rows:
        return {
            "ndcg": 0.0,
            "mrr": 0.0,
            "recall": 0.0,
            "latency_avg_ms": 0.0,
            "latency_p95_ms": 0.0,
        }
    ndcg = [r["ndcg"] for r in rows]
    mrr = [r["mrr"] for r in rows]
    recall = [r["recall"] for r in rows]
    latency = [r["latency_ms"] for r in rows]
    return {
        "ndcg": float(np.mean(ndcg)),
        "mrr": float(np.mean(mrr)),
        "recall": float(np.mean(recall)),
        "latency_avg_ms": float(np.mean(latency)),
        "latency_p95_ms": percentile(latency, 95),
    }


def oracle_select(per_query, expert_names):
    selected_rows = []
    selected_experts = []
    for i in range(len(per_query[expert_names[0]])):
        candidates = []
        for name in expert_names:
            row = per_query[name][i]
            # maximize ndcg, then mrr, then recall; tie-break by lower latency
            score_key = (row["ndcg"], row["mrr"], row["recall"], -row["latency_ms"])
            candidates.append((score_key, name, row))
        candidates.sort(reverse=True, key=lambda x: x[0])
        _, best_name, best_row = candidates[0]
        selected_experts.append(best_name)
        selected_rows.append(best_row)
    return selected_rows, selected_experts


def model_meta_path(model_path):
    return f"{model_path}.meta.json"


def load_model_meta(model_path):
    path = model_meta_path(model_path)
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_model_meta(model_path, data_hash, args):
    path = model_meta_path(model_path)
    payload = {
        "saved_at": datetime.now(timezone.utc).isoformat(),
        "data_sha256": data_hash,
        "seed": args.seed,
        "n_estimators": args.n_estimators,
        "learning_rate": args.learning_rate,
        "max_depth": args.max_depth,
        "lgbm_jobs": args.lgbm_jobs,
        "deterministic": args.deterministic,
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    return payload


def prepare_ltr_ranker(args, feature_extractor, training_data, data_hash):
    warnings = []
    source_used = "unavailable"
    model_meta = None

    model_exists = os.path.exists(args.ltr_model_path)
    meta = load_model_meta(args.ltr_model_path) if model_exists else None
    needs_retrain = args.ltr_source == "retrain"

    if args.ltr_source == "pretrained":
        if not model_exists:
            raise FileNotFoundError(f"LTR model not found: {args.ltr_model_path}")
        if meta and meta.get("data_sha256") != data_hash:
            warnings.append(
                "Pretrained LTR model meta hash mismatches current data hash; continuing due to --ltr-source=pretrained."
            )
    elif args.ltr_source == "auto":
        if not model_exists:
            needs_retrain = True
            warnings.append("LTR model not found; retraining with current oracle data.")
        elif not meta:
            message = "LTR model meta missing; cannot verify model/data consistency."
            if args.auto_retrain_on_mismatch:
                needs_retrain = True
                warnings.append(f"{message} Auto-retraining is enabled.")
            else:
                warnings.append(f"{message} Using pretrained model.")
        elif meta.get("data_sha256") != data_hash:
            message = (
                f"LTR model hash mismatch (model={meta.get('data_sha256')} current={data_hash})."
            )
            if args.auto_retrain_on_mismatch:
                needs_retrain = True
                warnings.append(f"{message} Auto-retraining is enabled.")
            else:
                warnings.append(f"{message} Using pretrained model.")

    ltr = LTRRanker(feature_extractor)
    if needs_retrain:
        ltr.train(
            training_data=training_data,
            validation_data=None,
            n_estimators=args.n_estimators,
            learning_rate=args.learning_rate,
            max_depth=args.max_depth,
            random_state=args.seed,
            n_jobs=args.lgbm_jobs,
            deterministic=args.deterministic,
        )
        source_used = "retrain"
        if args.save_ltr_model:
            os.makedirs(os.path.dirname(args.ltr_model_path) or ".", exist_ok=True)
            ltr.save_model(args.ltr_model_path)
            model_meta = save_model_meta(args.ltr_model_path, data_hash, args)
    else:
        ltr.load_model(args.ltr_model_path)
        source_used = "pretrained"
        model_meta = meta

    return ltr, source_used, model_meta, warnings


def build_markdown(summary, oracle, best_single_name, ndcg_k, mrr_k, recall_k, diagnostics):
    lines = [
        f"| Expert | nDCG@{ndcg_k} | MRR@{mrr_k} | Recall@{recall_k} | Avg Latency (ms) | P95 Latency (ms) |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    order = [k for k in ["Baseline", "LTR", "Cross-Encoder", "Hybrid"] if k in summary]
    for name in order:
        m = summary[name]
        lines.append(
            f"| {name} | {m['ndcg']:.4f} | {m['mrr']:.4f} | {m['recall']:.4f} | "
            f"{m['latency_avg_ms']:.2f} | {m['latency_p95_ms']:.2f} |"
        )
    lines.append(
        f"| **Oracle (upper bound)** | **{oracle['ndcg']:.4f}** | **{oracle['mrr']:.4f}** | "
        f"**{oracle['recall']:.4f}** | **{oracle['latency_avg_ms']:.2f}** | **{oracle['latency_p95_ms']:.2f}** |"
    )
    gap = oracle["ndcg"] - summary[best_single_name]["ndcg"]
    rel = 0.0
    if summary[best_single_name]["ndcg"] > 1e-12:
        rel = gap / summary[best_single_name]["ndcg"]
    lines.append("")
    lines.append(f"- Best single expert: `{best_single_name}`")
    lines.append(f"- Oracle nDCG@{ndcg_k} gain: `{gap:.4f}` ({rel * 100:.2f}%)")
    lines.append(f"- LTR source: `{diagnostics['ltr_source']}`")
    lines.append(
        f"- Data profile: queries={diagnostics['data_profile']['query_count']} "
        f"prefix_like_ratio={diagnostics['data_profile']['prefix_like_ratio']:.1%} "
        f"first_doc_is_max_ratio={diagnostics['data_profile']['first_doc_is_max_ratio']:.1%}"
    )
    if diagnostics["warnings"]:
        lines.append("")
        lines.append("### Warnings")
        for warning in diagnostics["warnings"]:
            lines.append(f"- {warning}")
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description="Oracle upper-bound experiment for adaptive routing")
    parser.add_argument("--data", default="data/ltr_training_data.json")
    parser.add_argument("--output-dir", default="models/oracle_experiment")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sample", choices=["head", "tail", "random"], default="random")
    parser.add_argument("--max-queries", type=int, default=100)
    parser.add_argument("--ndcg-k", type=int, default=10)
    parser.add_argument("--mrr-k", type=int, default=10)
    parser.add_argument("--recall-k", type=int, default=50)

    parser.add_argument("--min-query-terms", type=int, default=0)
    parser.add_argument("--min-query-chars", type=int, default=0)
    parser.add_argument("--max-stopword-ratio", type=float, default=1.0)
    parser.add_argument("--drop-prefix-queries", action="store_true")
    parser.add_argument("--min-relevant-docs", type=int, default=0)

    parser.add_argument("--no-cross", action="store_true")
    parser.add_argument("--cross-model", default="cross-encoder/ms-marco-MiniLM-L-6-v2")
    parser.add_argument("--cross-content-chars", type=int, default=512)
    parser.add_argument("--hybrid-top-n", type=int, default=30)

    parser.add_argument("--ltr-model-path", default="models/ltr_model.pkl")
    parser.add_argument("--ltr-source", choices=["auto", "pretrained", "retrain"], default="auto")
    parser.add_argument("--n-estimators", type=int, default=80)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--max-depth", type=int, default=6)
    parser.add_argument("--lgbm-jobs", type=int, default=1)
    parser.set_defaults(
        deterministic=True,
        save_ltr_model=True,
        auto_retrain_on_mismatch=True,
    )
    parser.add_argument("--deterministic", dest="deterministic", action="store_true")
    parser.add_argument("--non-deterministic", dest="deterministic", action="store_false")
    parser.add_argument("--save-ltr-model", dest="save_ltr_model", action="store_true")
    parser.add_argument("--no-save-ltr-model", dest="save_ltr_model", action="store_false")
    parser.add_argument(
        "--auto-retrain-on-mismatch",
        dest="auto_retrain_on_mismatch",
        action="store_true",
    )
    parser.add_argument(
        "--no-auto-retrain-on-mismatch",
        dest="auto_retrain_on_mismatch",
        action="store_false",
    )

    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    with open(args.data, "r", encoding="utf-8") as f:
        raw_data = json.load(f)

    clean_data, dropped_invalid = sanitize_dataset(raw_data)
    if len(clean_data) < 3:
        raise RuntimeError(f"clean dataset too small: {len(clean_data)}")

    sampled_data = sample_data(clean_data, args.sample, args.max_queries, args.seed)
    filtered_data, dropped_quality = filter_data_quality(
        sampled_data,
        min_query_terms=args.min_query_terms,
        min_query_chars=args.min_query_chars,
        max_stopword_ratio=args.max_stopword_ratio,
        drop_prefix_queries=args.drop_prefix_queries,
        min_relevant_docs=args.min_relevant_docs,
    )
    if len(filtered_data) < 2:
        raise RuntimeError(
            f"filtered dataset too small: {len(filtered_data)} "
            f"(sampled={len(sampled_data)}, dropped_quality={len(dropped_quality)})"
        )

    data_profile, profile_warnings = build_data_profile(filtered_data)

    evaluator = RankingEvaluator()
    feature_extractor = FeatureExtractor()
    data_hash = hash_json_obj(filtered_data)

    experts = {"Baseline": BaselineRanker()}
    ltr, ltr_source_used, ltr_model_meta, ltr_warnings = prepare_ltr_ranker(
        args,
        feature_extractor,
        filtered_data,
        data_hash,
    )
    experts["LTR"] = ltr

    cross_status = "disabled"
    if not args.no_cross:
        try:
            from sentence_transformers import CrossEncoder

            cross_model = CrossEncoder(args.cross_model)
            cross = CrossEncoderRanker(cross_model, content_chars=args.cross_content_chars)
            experts["Cross-Encoder"] = cross
            experts["Hybrid"] = HybridRanker(experts["LTR"], cross, top_n=args.hybrid_top_n)
            cross_status = "enabled"
        except Exception as e:
            cross_status = f"unavailable: {e}"

    per_query = {}
    for name, expert in experts.items():
        per_query[name] = evaluate_expert(
            expert,
            filtered_data,
            evaluator,
            ndcg_k=args.ndcg_k,
            mrr_k=args.mrr_k,
            recall_k=args.recall_k,
        )

    summary = {name: aggregate_rows(rows) for name, rows in per_query.items()}

    expert_names = list(per_query.keys())
    oracle_rows, selected_experts = oracle_select(per_query, expert_names)
    oracle_summary = aggregate_rows(oracle_rows)
    selection_counter = Counter(selected_experts)

    best_single_name = max(summary.items(), key=lambda kv: kv[1]["ndcg"])[0]
    best_single = summary[best_single_name]

    ndcg_gain_abs = oracle_summary["ndcg"] - best_single["ndcg"]
    ndcg_gain_rel = 0.0
    if best_single["ndcg"] > 1e-12:
        ndcg_gain_rel = ndcg_gain_abs / best_single["ndcg"]

    warnings = profile_warnings + ltr_warnings
    diagnostics = {
        "data_profile": data_profile,
        "warnings": warnings,
        "ltr_source": ltr_source_used,
    }

    results = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "seed": args.seed,
        "data_path": args.data,
        "data_raw_count": len(raw_data),
        "data_dropped_invalid": dropped_invalid,
        "data_sampled_count": len(sampled_data),
        "data_dropped_quality_count": len(dropped_quality),
        "data_used_count": len(filtered_data),
        "sample_method": args.sample,
        "max_queries": args.max_queries,
        "data_sha256": data_hash,
        "metrics": {
            "ndcg_k": args.ndcg_k,
            "mrr_k": args.mrr_k,
            "recall_k": args.recall_k,
        },
        "quality_filter_args": {
            "min_query_terms": args.min_query_terms,
            "min_query_chars": args.min_query_chars,
            "max_stopword_ratio": args.max_stopword_ratio,
            "drop_prefix_queries": args.drop_prefix_queries,
            "min_relevant_docs": args.min_relevant_docs,
        },
        "diagnostics": diagnostics,
        "dropped_quality_examples": dropped_quality[:20],
        "ltr_source": ltr_source_used,
        "ltr_model_path": args.ltr_model_path,
        "ltr_model_meta": ltr_model_meta,
        "cross_status": cross_status,
        "experts": summary,
        "oracle": {
            "summary": oracle_summary,
            "selection_count": dict(selection_counter),
            "best_single_expert": best_single_name,
            "best_single_summary": best_single,
            "ndcg_gain_abs": ndcg_gain_abs,
            "ndcg_gain_rel": ndcg_gain_rel,
        },
    }

    with open(os.path.join(args.output_dir, "oracle_results.json"), "w", encoding="utf-8") as f:
        json.dump(convert_numpy(results), f, indent=2, ensure_ascii=False)

    with open(os.path.join(args.output_dir, "oracle_table.md"), "w", encoding="utf-8") as f:
        f.write(
            build_markdown(
                summary,
                oracle_summary,
                best_single_name,
                args.ndcg_k,
                args.mrr_k,
                args.recall_k,
                diagnostics,
            )
        )

    with open(os.path.join(args.output_dir, "oracle_per_query.json"), "w", encoding="utf-8") as f:
        json.dump(
            {
                "selected_expert": selected_experts,
                "experts": per_query,
            },
            f,
            indent=2,
            ensure_ascii=False,
        )

    print("Oracle experiment completed.")
    print(f"- experts: {', '.join(expert_names)}")
    print(f"- best single: {best_single_name}")
    print(f"- oracle nDCG gain abs: {ndcg_gain_abs:.4f}")
    print(f"- oracle nDCG gain rel: {ndcg_gain_rel * 100:.2f}%")
    print(f"- data used: {len(filtered_data)} (sampled={len(sampled_data)}, quality_drop={len(dropped_quality)})")
    print(f"- ltr source: {ltr_source_used}")
    if warnings:
        print("- warnings:")
        for warning in warnings:
            print(f"  * {warning}")
    print(f"- output: {args.output_dir}")


if __name__ == "__main__":
    main()
