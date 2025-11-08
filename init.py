import time
from elasticsearch import Elasticsearch
from elasticsearch.helpers import bulk
import json

# Give Elasticsearch time to start
time.sleep(30)

es = Elasticsearch([{'host': 'elasticsearch', 'port': 9200, 'scheme': 'http'}])

def create_index_and_bulk_index():
    """
    Create (or recreate) the 'documents' index and bulk index data.
    Prefer using processed dataset if available to match app.py mappings.
    """
    index_name = "documents"
    if es.indices.exists(index=index_name):
        print(f"Index '{index_name}' exists. Deleting...")
        es.indices.delete(index=index_name)

    print(f"Creating index '{index_name}' with unified mappings...")
    mappings = {
        "properties": {
            "title": {
                "type": "text",
                "analyzer": "standard",
                "fields": {"keyword": {"type": "keyword"}}
            },
            "content": {"type": "text", "analyzer": "standard"},
            "content_full": {"type": "text", "analyzer": "standard"},
            "keywords": {"type": "keyword"},
            "related_queries": {"type": "text", "analyzer": "standard"},
            "combined_text": {"type": "text", "analyzer": "standard"},
            "quality": {"type": "keyword"}
        }
    }
    es.indices.create(index=index_name, mappings=mappings)
    print(f"Index '{index_name}' created.")

    # Prefer processed data, fallback to raw (兼容 100k 命名)
    data_paths_processed = [
        "data/msmarco_100k_processed.json",
        "data/msmarco_docs_processed.json"
    ]
    data_paths_raw = [
        "data/msmarco_100k.json",
        "data/msmarco_docs.json"
    ]

    documents = None
    processed = False
    for p in data_paths_processed:
        try:
            with open(p, "r", encoding="utf-8") as f:
                documents = json.load(f)
            print(f"✓ Using processed dataset: {p}")
            processed = True
            break
        except FileNotFoundError:
            continue
    if documents is None:
        for p in data_paths_raw:
            try:
                with open(p, "r", encoding="utf-8") as f:
                    documents = json.load(f)
                print(f"⚠ Processed dataset not found, using raw dataset: {p}")
                processed = False
                break
            except FileNotFoundError:
                continue
    if documents is None:
        raise FileNotFoundError("No dataset found. Please download or preprocess MS MARCO first.")

    actions = []
    for doc in documents:
        if processed:
            src = {
                "title": doc.get("title", ""),
                "content": doc.get("content", ""),
                "content_full": doc.get("content_full", doc.get("content", "")),
                "keywords": doc.get("keywords", []),
                "related_queries": doc.get("related_queries", []),
                "combined_text": doc.get("combined_text", f"{doc.get('title','')} {doc.get('content','')}")
            }
            if "quality" in doc:
                src["quality"] = doc.get("quality")
        else:
            combined_text = f"{doc.get('title','')} {doc.get('content','')}"
            if doc.get('related_queries'):
                combined_text += " " + " ".join(doc['related_queries'])
            src = {
                "title": doc.get("title", ""),
                "content": doc.get("content", ""),
                "content_full": doc.get("content", ""),
                "related_queries": doc.get("related_queries", []),
                "combined_text": combined_text
            }

        actions.append({
            "_index": index_name,
            "_id": doc.get("id", ""),
            "_source": src
        })

    bulk(es, actions, refresh=True)
    print(f"Bulk indexing completed. Indexed {len(actions)} documents.")

if __name__ == '__main__':
    create_index_and_bulk_index()