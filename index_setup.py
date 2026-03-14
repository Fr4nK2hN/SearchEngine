import hashlib
import json
import os


INDEX_NAME = "documents"
INDEX_SCHEMA_VERSION = 2
DOCUMENT_BUILD_SPEC = {
    "processed": {
        "include_fields": [
            "title",
            "content",
            "content_full",
            "keywords",
            "related_queries",
            "combined_text",
        ],
        "optional_fields": ["quality"],
        "content_full_fallback": "content",
        "combined_text_fallback": "title + content",
    },
    "raw": {
        "include_fields": [
            "title",
            "content",
            "content_full",
            "related_queries",
            "combined_text",
        ],
        "content_full_fallback": "content",
        "combined_text_fallback": "title + content + related_queries",
    },
}

PROCESSED_DATA_PATHS = [
    "data/msmarco_100k_processed.json",
    "data/msmarco_docs_processed.json",
]
RAW_DATA_PATHS = [
    "data/msmarco_100k.json",
    "data/msmarco_docs.json",
]


def get_index_properties():
    return {
        "title": {
            "type": "text",
            "analyzer": "standard",
            "fields": {"keyword": {"type": "keyword"}},
        },
        "content": {"type": "text", "analyzer": "standard"},
        "content_full": {"type": "text", "analyzer": "standard"},
        "keywords": {"type": "keyword"},
        "related_queries": {"type": "text", "analyzer": "standard"},
        "combined_text": {"type": "text", "analyzer": "standard"},
        "quality": {"type": "keyword"},
    }


def resolve_dataset():
    for path in PROCESSED_DATA_PATHS:
        if os.path.exists(path):
            return {"path": path, "processed": True}
    for path in RAW_DATA_PATHS:
        if os.path.exists(path):
            return {"path": path, "processed": False}
    raise FileNotFoundError(
        "No dataset found. Please download or preprocess MS MARCO first."
    )


def load_documents(dataset_path):
    with open(dataset_path, "r", encoding="utf-8") as f:
        return json.load(f)


def compute_file_sha256(path, chunk_size=1024 * 1024):
    hasher = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            hasher.update(chunk)
    return hasher.hexdigest()


def compute_index_fingerprint(dataset_path):
    payload = {
        "schema_version": INDEX_SCHEMA_VERSION,
        "properties": get_index_properties(),
        "document_build_spec": DOCUMENT_BUILD_SPEC,
        "dataset_sha256": compute_file_sha256(dataset_path),
    }
    encoded = json.dumps(payload, sort_keys=True, ensure_ascii=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def build_index_mappings(dataset_path):
    fingerprint = compute_index_fingerprint(dataset_path)
    meta = {
        "index_fingerprint": fingerprint,
        "schema_version": INDEX_SCHEMA_VERSION,
        "dataset_path": os.path.basename(dataset_path),
    }
    mappings = {
        "_meta": meta,
        "properties": get_index_properties(),
    }
    return mappings, meta


def get_expected_index_spec():
    dataset = resolve_dataset()
    mappings, meta = build_index_mappings(dataset["path"])
    return dataset, mappings, meta


def get_current_index_meta(es_client, index_name=INDEX_NAME):
    mapping = es_client.indices.get_mapping(index=index_name)
    return ((mapping.get(index_name) or {}).get("mappings") or {}).get("_meta") or {}


def index_meta_matches(current_meta, expected_meta):
    return current_meta.get("index_fingerprint") == expected_meta.get("index_fingerprint")


def build_source_document(doc, processed):
    if processed:
        src = {
            "title": doc.get("title", ""),
            "content": doc.get("content", ""),
            "content_full": doc.get("content_full", doc.get("content", "")),
            "keywords": doc.get("keywords", []),
            "related_queries": doc.get("related_queries", []),
            "combined_text": doc.get(
                "combined_text",
                f"{doc.get('title', '')} {doc.get('content', '')}",
            ),
        }
        if "quality" in doc:
            src["quality"] = doc.get("quality")
        return src

    combined_text = f"{doc.get('title', '')} {doc.get('content', '')}"
    if doc.get("related_queries"):
        combined_text += " " + " ".join(doc["related_queries"])
    return {
        "title": doc.get("title", ""),
        "content": doc.get("content", ""),
        "content_full": doc.get("content", ""),
        "related_queries": doc.get("related_queries", []),
        "combined_text": combined_text,
    }
