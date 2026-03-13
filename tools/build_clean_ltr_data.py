#!/usr/bin/env python3
"""
Build a cleaner LTR dataset from existing query-document-label JSON.

Main goals:
- remove low-quality/prefix-like queries
- optionally relabel with Cross-Encoder only (without heuristic rules)
- enforce minimum relevant docs per query
"""

import argparse
import json
import os
import random
import re
from collections import Counter
from datetime import datetime, timezone


STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from",
    "how", "in", "is", "it", "of", "on", "or", "that", "the", "to",
    "was", "were", "what", "when", "where", "which", "who", "why",
    "will", "with", "you", "your",
}
TOKEN_PATTERN = re.compile(r"[a-z0-9]+")


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


def score_to_label(score):
    if score > 0.7:
        return 4
    if score > 0.5:
        return 3
    if score > 0.3:
        return 2
    if score > 0.1:
        return 1
    return 0


def scores_to_labels(scores, strategy):
    if strategy == "fixed":
        return [score_to_label(float(s)) for s in scores]
    if strategy == "quantile":
        n = len(scores)
        if n == 0:
            return []
        ranked_idx = sorted(range(n), key=lambda i: float(scores[i]), reverse=True)
        labels = [0] * n
        for pos, idx in enumerate(ranked_idx):
            frac = (pos + 1) / n
            if frac <= 0.05:
                labels[idx] = 4
            elif frac <= 0.15:
                labels[idx] = 3
            elif frac <= 0.30:
                labels[idx] = 2
            elif frac <= 0.50:
                labels[idx] = 1
            else:
                labels[idx] = 0
        return labels
    raise ValueError(f"Unknown labeling strategy: {strategy}")


def sanitize_dataset(data):
    cleaned = []
    dropped = 0
    for item in data:
        if not isinstance(item, dict):
            dropped += 1
            continue
        query = " ".join((item.get("query") or "").strip().split())
        docs = item.get("documents")
        labels = item.get("relevance_labels")
        if not query or not isinstance(docs, list) or not isinstance(labels, list):
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


def sample_data(data, max_queries, sample_method, seed):
    if max_queries is None or max_queries <= 0 or max_queries >= len(data):
        return data
    if sample_method == "head":
        return data[:max_queries]
    if sample_method == "tail":
        return data[-max_queries:]
    rng = random.Random(seed)
    idx = sorted(rng.sample(range(len(data)), max_queries))
    return [data[i] for i in idx]


def relabel_with_cross_encoder(data, model_name, content_chars, labeling_strategy):
    from sentence_transformers import CrossEncoder

    model = CrossEncoder(model_name)
    relabeled = []
    for item in data:
        query = item["query"]
        docs = item["documents"]
        pairs = [[query, (doc.get("content", "") or "")[:content_chars]] for doc in docs]
        scores = model.predict(pairs)
        labels = scores_to_labels(scores, labeling_strategy)
        relabeled.append(
            {
                "query": query,
                "documents": docs,
                "relevance_labels": labels,
            }
        )
    return relabeled


def filter_quality(
    data,
    min_query_terms,
    min_query_chars,
    max_stopword_ratio,
    drop_prefix_queries,
    min_relevant_docs,
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

        relevant_count = sum(1 for x in labels if x > 0)
        if relevant_count < min_relevant_docs:
            reasons.append(f"relevant<{min_relevant_docs}")

        if reasons:
            dropped.append({"query": query, "reasons": reasons})
        else:
            kept.append(item)
    return kept, dropped


def build_label_distribution(data):
    counter = Counter()
    for item in data:
        counter.update(item["relevance_labels"])
    return dict(sorted(counter.items()))


def main():
    parser = argparse.ArgumentParser(description="Build clean LTR dataset")
    parser.add_argument("--input", default="data/ltr_training_data.json")
    parser.add_argument("--output", default="data/ltr_training_data_clean.json")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-queries", type=int, default=0)
    parser.add_argument("--sample", choices=["head", "tail", "random"], default="random")

    parser.add_argument("--min-query-terms", type=int, default=3)
    parser.add_argument("--min-query-chars", type=int, default=8)
    parser.add_argument("--max-stopword-ratio", type=float, default=0.85)
    parser.add_argument("--drop-prefix-queries", action="store_true")
    parser.add_argument("--min-relevant-docs", type=int, default=1)

    parser.add_argument("--relabel-mode", choices=["keep", "cross-only"], default="cross-only")
    parser.add_argument("--cross-model", default="cross-encoder/ms-marco-MiniLM-L-6-v2")
    parser.add_argument("--cross-content-chars", type=int, default=512)
    parser.add_argument("--labeling-strategy", choices=["fixed", "quantile"], default="fixed")
    args = parser.parse_args()

    random.seed(args.seed)

    with open(args.input, "r", encoding="utf-8") as f:
        raw_data = json.load(f)

    data, dropped_invalid = sanitize_dataset(raw_data)
    sampled = sample_data(data, args.max_queries, args.sample, args.seed)

    if args.relabel_mode == "cross-only":
        sampled = relabel_with_cross_encoder(
            sampled,
            model_name=args.cross_model,
            content_chars=args.cross_content_chars,
            labeling_strategy=args.labeling_strategy,
        )

    cleaned, dropped_quality = filter_quality(
        sampled,
        min_query_terms=args.min_query_terms,
        min_query_chars=args.min_query_chars,
        max_stopword_ratio=args.max_stopword_ratio,
        drop_prefix_queries=args.drop_prefix_queries,
        min_relevant_docs=args.min_relevant_docs,
    )

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(cleaned, f, indent=2, ensure_ascii=False)

    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "input": args.input,
        "output": args.output,
        "seed": args.seed,
        "raw_count": len(raw_data),
        "dropped_invalid": dropped_invalid,
        "sampled_count": len(sampled),
        "clean_count": len(cleaned),
        "dropped_quality_count": len(dropped_quality),
        "label_distribution": build_label_distribution(cleaned),
        "quality_args": {
            "min_query_terms": args.min_query_terms,
            "min_query_chars": args.min_query_chars,
            "max_stopword_ratio": args.max_stopword_ratio,
            "drop_prefix_queries": args.drop_prefix_queries,
            "min_relevant_docs": args.min_relevant_docs,
        },
        "relabel_mode": args.relabel_mode,
        "labeling_strategy": args.labeling_strategy,
        "dropped_quality_examples": dropped_quality[:30],
    }
    report_path = f"{args.output}.report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print("Clean data build completed.")
    print(f"- sampled: {len(sampled)}")
    print(f"- clean: {len(cleaned)}")
    print(f"- dropped quality: {len(dropped_quality)}")
    print(f"- output: {args.output}")
    print(f"- report: {report_path}")


if __name__ == "__main__":
    main()
