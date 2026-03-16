import argparse
import os
import sys

sys.path.insert(
    0,
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
)

from webapp.bootstrap import create_es_client
from webapp.config import load_config


def format_ratio(part, total):
    if total <= 0:
        return "0.00%"
    return f"{(part / total) * 100.0:.2f}%"


def compute_coverage_stats(total_docs, content_docs, title_docs, both_docs):
    total_docs = int(total_docs or 0)
    content_docs = int(content_docs or 0)
    title_docs = int(title_docs or 0)
    both_docs = int(both_docs or 0)
    return {
        "total_docs": total_docs,
        "content_emb_docs": content_docs,
        "title_emb_docs": title_docs,
        "both_emb_docs": both_docs,
        "missing_content_emb_docs": max(0, total_docs - content_docs),
        "missing_title_emb_docs": max(0, total_docs - title_docs),
        "missing_any_emb_docs": max(0, total_docs - both_docs),
        "content_emb_ratio": format_ratio(content_docs, total_docs),
        "title_emb_ratio": format_ratio(title_docs, total_docs),
        "both_emb_ratio": format_ratio(both_docs, total_docs),
    }


def count_exists(es, index, field_name):
    return es.count(index=index, query={"exists": {"field": field_name}})["count"]


def count_both_exists(es, index, left_field, right_field):
    return es.count(
        index=index,
        query={
            "bool": {
                "must": [
                    {"exists": {"field": left_field}},
                    {"exists": {"field": right_field}},
                ]
            }
        },
    )["count"]


def analyze(index):
    config = load_config()
    es = create_es_client(config)
    total_docs = es.count(index=index, query={"match_all": {}})["count"]
    content_docs = count_exists(es, index, "content_emb")
    title_docs = count_exists(es, index, "title_emb")
    both_docs = count_both_exists(es, index, "content_emb", "title_emb")
    return compute_coverage_stats(total_docs, content_docs, title_docs, both_docs)


def main():
    config = load_config()
    parser = argparse.ArgumentParser(description="Analyze embedding coverage inside Elasticsearch index")
    parser.add_argument("--index", default=config.index_name)
    args = parser.parse_args()

    try:
        stats = analyze(args.index)
    except Exception as exc:
        print(f"Failed to analyze embedding coverage for index '{args.index}': {exc}")
        print(
            "Hint: ensure Elasticsearch is running and set SEARCHENGINE_ES_HOST/PORT "
            "to a reachable endpoint if you are not inside docker-compose."
        )
        raise SystemExit(1)

    print(f"Index: {args.index}")
    print(f"Total docs: {stats['total_docs']}")
    print(
        f"content_emb: {stats['content_emb_docs']} "
        f"({stats['content_emb_ratio']}) | missing={stats['missing_content_emb_docs']}"
    )
    print(
        f"title_emb: {stats['title_emb_docs']} "
        f"({stats['title_emb_ratio']}) | missing={stats['missing_title_emb_docs']}"
    )
    print(
        f"both fields: {stats['both_emb_docs']} "
        f"({stats['both_emb_ratio']}) | missing_any={stats['missing_any_emb_docs']}"
    )


if __name__ == "__main__":
    main()
