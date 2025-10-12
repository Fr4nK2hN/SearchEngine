import time
from elasticsearch import Elasticsearch
from elasticsearch.helpers import bulk
import json

# Give Elasticsearch time to start
time.sleep(30)

es = Elasticsearch([{'host': 'elasticsearch', 'port': 9200, 'scheme': 'http'}])

def create_index_and_bulk_index():
    """
    Creates an Elasticsearch index and bulk indexes the MS MARCO documents if the index doesn't exist.
    """
    index_name = "documents"
    if es.indices.exists(index=index_name):
        print(f"Index '{index_name}' exists. Deleting...")
        es.indices.delete(index=index_name)

    print(f"Creating index '{index_name}'...")
    mappings = {
        "properties": {
            "title": {
                "type": "text",
                "analyzer": "standard",
                "fields": {
                    "keyword": {"type": "keyword"}
                }
            },
            "content": {
                "type": "text",
                "analyzer": "standard"
            },
            "related_queries": {
                "type": "text",
                "analyzer": "standard"
            },
            "combined_text": {
                "type": "text",
                "analyzer": "standard"
            }
        }
    }
    es.indices.create(index=index_name, mappings=mappings)
    print(f"Index '{index_name}' created.")

    with open("data/msmarco_docs.json", "r", encoding="utf-8") as f:
        documents = json.load(f)

    actions = []
    for doc in documents:
        # 创建组合文本字段以提高搜索效果
        combined_text = f"{doc['title']} {doc['content']}"
        if doc.get('related_queries'):
            combined_text += " " + " ".join(doc['related_queries'])
        
        actions.append({
            "_index": index_name,
            "_id": doc["id"],
            "_source": {
                "title": doc["title"],
                "content": doc["content"],
                "related_queries": doc.get("related_queries", []),
                "combined_text": combined_text
            }
        })

    bulk(es, actions)
    print(f"Bulk indexing completed. Indexed {len(actions)} documents.")

if __name__ == '__main__':
    create_index_and_bulk_index()