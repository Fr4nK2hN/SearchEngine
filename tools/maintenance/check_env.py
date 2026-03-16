#!/usr/bin/env python3

import argparse
import importlib.util
import os
import platform
import sys

sys.path.insert(
    0,
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
)

from webapp.bootstrap import create_es_client
from webapp.config import load_config


CORE_IMPORTS = [
    ("flask", "Flask"),
    ("elasticsearch", "elasticsearch"),
    ("sentence_transformers", "sentence-transformers"),
    ("torch", "torch"),
    ("lightgbm", "lightgbm"),
    ("sklearn", "scikit-learn"),
    ("nltk", "nltk"),
    ("pythonjsonlogger", "python-json-logger"),
]


def check_imports():
    rows = []
    for module_name, label in CORE_IMPORTS:
        rows.append((label, importlib.util.find_spec(module_name) is not None))
    return rows


def check_files(config):
    candidates = [
        ("processed_corpus", "data/msmarco_100k_processed.json"),
        ("raw_corpus", "data/msmarco_100k.json"),
        ("idf_dict", "models/idf_dict.json"),
        ("ltr_model", config.ltr_model_path),
        ("query_router_model", config.query_router_model_path),
    ]
    return [(label, path, os.path.exists(path)) for label, path in candidates]


def check_elasticsearch(config):
    try:
        es = create_es_client(config)
        return es.ping(), None
    except Exception as exc:
        return False, str(exc)


def main():
    parser = argparse.ArgumentParser(description="Check local SearchEngine runtime prerequisites")
    parser.add_argument("--check-es", action="store_true", help="Ping Elasticsearch as part of the check")
    args = parser.parse_args()

    config = load_config()
    failures = []

    print(f"Python: {platform.python_version()}")
    print(f"Configured Elasticsearch: {config.es_scheme}://{config.es_host}:{config.es_port}")
    print(f"Configured index: {config.index_name}")

    print("\nImports:")
    for label, ok in check_imports():
        print(f"  [{'OK' if ok else 'MISS'}] {label}")
        if not ok:
            failures.append(f"missing import: {label}")

    print("\nFiles:")
    for label, path, ok in check_files(config):
        print(f"  [{'OK' if ok else 'MISS'}] {label}: {path}")
        if label in {"processed_corpus", "idf_dict", "ltr_model", "query_router_model"} and not ok:
            failures.append(f"missing file: {path}")

    if args.check_es:
        ok, error = check_elasticsearch(config)
        if ok:
            print("\nElasticsearch:\n  [OK] ping succeeded")
        else:
            print("\nElasticsearch:")
            print(f"  [MISS] ping failed: {error or 'ping returned false'}")
            failures.append("elasticsearch ping failed")

    if failures:
        print("\nEnvironment check failed:")
        for item in failures:
            print(f"  - {item}")
        raise SystemExit(1)

    print("\nEnvironment check passed.")


if __name__ == "__main__":
    main()
