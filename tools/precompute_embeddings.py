import argparse
from elasticsearch import Elasticsearch
from elasticsearch.helpers import bulk
from sentence_transformers import SentenceTransformer


def precompute(index: str, limit: int, batch_size: int):
    es = Elasticsearch([{'host': 'elasticsearch', 'port': 9200, 'scheme': 'http'}])
    model = SentenceTransformer('all-MiniLM-L6-v2')

    fetched = 0
    actions = []
    res = es.search(index=index, size=500, query={"match_all": {}}, scroll='2m')
    scroll_id = res.get('_scroll_id')

    def flush():
        nonlocal actions
        if actions:
            bulk(es, actions)
            actions = []

    while True:
        hits = res['hits']['hits']
        if not hits:
            break
        for h in hits:
            if limit and fetched >= limit:
                break
            _id = h['_id']
            src = h.get('_source') or {}
            title = src.get('title') or ''
            content = (src.get('content_full') or src.get('content') or '')
            content_short = content[:512]
            title_emb = model.encode(title)
            content_emb = model.encode(content_short)
            actions.append({
                '_op_type': 'update',
                '_index': index,
                '_id': _id,
                'doc': {
                    'title_emb': list(map(float, title_emb)),
                    'content_emb': list(map(float, content_emb))
                }
            })
            fetched += 1
            if len(actions) >= batch_size:
                flush()
        if limit and fetched >= limit:
            break
        res = es.scroll(scroll_id=scroll_id, scroll='2m')
        scroll_id = res.get('_scroll_id')

    flush()
    print(f"Precomputed embeddings for {fetched} documents in index '{index}'")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--index', default='documents')
    parser.add_argument('--limit', type=int, default=5000)
    parser.add_argument('--batch', type=int, default=64)
    args = parser.parse_args()
    precompute(args.index, args.limit, args.batch)

