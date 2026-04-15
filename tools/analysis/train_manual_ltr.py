import argparse
import csv
import json
import os
import random
import sys
from collections import defaultdict
from datetime import datetime, timezone


sys.path.insert(
    0,
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
)

from engine.retrieval import search_documents_with_fallback  # noqa: E402
from ranking.feature_extractor import FeatureExtractor  # noqa: E402
from ranking.ranker import LTRRanker  # noqa: E402


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


def document_from_hit(hit):
    source = dict(hit.get("_source") or {})
    source["id"] = str(hit.get("_id") or source.get("id") or "")
    source["es_score"] = float(hit.get("_score", source.get("es_score", 0.0)) or 0.0)
    return source


def build_training_data(app_state, judgments, candidate_size, relax_threshold):
    out = []
    missing_positive = 0
    for query, label_map in judgments.items():
        hits, retrieval_strategy = search_documents_with_fallback(
            app_state.es,
            query,
            size=candidate_size,
            hl=False,
            relax_threshold=relax_threshold,
            index_name=app_state.runtime_config.index_name,
        )
        documents = []
        labels = []
        seen = set()
        for hit in hits:
            doc_id = str(hit.get("_id") or "")
            if doc_id not in label_map:
                continue
            seen.add(doc_id)
            documents.append(document_from_hit(hit))
            labels.append(int(label_map[doc_id]))

        missing_positive += sum(
            1
            for doc_id, label in label_map.items()
            if int(label) > 0 and doc_id not in seen
        )
        if documents:
            out.append(
                {
                    "query": query,
                    "documents": documents,
                    "relevance_labels": labels,
                    "meta": {
                        "source": "manual_pooled_judgments",
                        "retrieval_strategy": retrieval_strategy,
                    },
                }
            )
    return out, {"missing_positive_labels": missing_positive}


def split_data(data, seed, train_ratio, val_ratio):
    rng = random.Random(seed)
    indices = list(range(len(data)))
    rng.shuffle(indices)
    n = len(indices)
    n_train = max(1, int(n * train_ratio))
    n_val = max(0, int(n * val_ratio))
    if n_train + n_val >= n:
        n_val = max(0, n - n_train - 1)
    train = [data[i] for i in indices[:n_train]]
    val = [data[i] for i in indices[n_train:n_train + n_val]]
    test = [data[i] for i in indices[n_train + n_val:]]
    return train, val, test, {
        "seed": seed,
        "train_indices": indices[:n_train],
        "val_indices": indices[n_train:n_train + n_val],
        "test_indices": indices[n_train + n_val:],
    }


def label_distribution(data):
    counts = defaultdict(int)
    for item in data:
        for label in item["relevance_labels"]:
            counts[int(label)] += 1
    return dict(sorted(counts.items()))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--judgments", default="reports/manual_relevance_pool_20260411/judgment_pool.csv")
    parser.add_argument("--label-column", default="manual_label")
    parser.add_argument("--min-labels-per-query", type=int, default=10)
    parser.add_argument("--candidate-size", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--n-estimators", type=int, default=80)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--max-depth", type=int, default=4)
    parser.add_argument("--out-dir", default="reports/manual_relevance_pool_20260411/manual_ltr_training")
    parser.add_argument("--model-output", default="models/ltr_model_manual.pkl")
    args = parser.parse_args()

    from app import app_state  # noqa: E402

    judgments = load_judgments(args.judgments, args.label_column, args.min_labels_per_query)
    if not judgments:
        raise RuntimeError("no usable manual judgments")

    os.makedirs(args.out_dir, exist_ok=True)
    data, diagnostics = build_training_data(
        app_state,
        judgments,
        candidate_size=args.candidate_size,
        relax_threshold=app_state.runtime_config.recall_relax_threshold,
    )
    if len(data) < 5:
        raise RuntimeError(f"not enough LTR training queries: {len(data)}")

    train_set, val_set, test_set, split = split_data(
        data,
        seed=args.seed,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
    )

    data_path = os.path.join(args.out_dir, "manual_ltr_training_data.json")
    with open(data_path, "w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, ensure_ascii=False)

    extractor = FeatureExtractor()
    ranker = LTRRanker(extractor)
    ranker.train(
        training_data=train_set,
        validation_data=val_set if val_set else None,
        n_estimators=args.n_estimators,
        learning_rate=args.learning_rate,
        max_depth=args.max_depth,
        random_state=args.seed,
        n_jobs=1,
        deterministic=True,
    )
    ranker.save_model(args.model_output)

    summary = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "judgments": args.judgments,
        "model_output": args.model_output,
        "training_data": data_path,
        "query_count": len(data),
        "document_count": sum(len(item["documents"]) for item in data),
        "label_distribution": label_distribution(data),
        "diagnostics": diagnostics,
        "split": split,
        "params": {
            "candidate_size": args.candidate_size,
            "n_estimators": args.n_estimators,
            "learning_rate": args.learning_rate,
            "max_depth": args.max_depth,
            "seed": args.seed,
        },
    }
    summary_path = os.path.join(args.out_dir, "summary.json")
    with open(summary_path, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)

    print(f"training_data: {data_path}")
    print(f"model: {args.model_output}")
    print(f"summary: {summary_path}")
    print(f"queries: {summary['query_count']}")
    print(f"documents: {summary['document_count']}")
    print(f"label_distribution: {summary['label_distribution']}")


if __name__ == "__main__":
    main()
