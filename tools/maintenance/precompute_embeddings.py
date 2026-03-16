import argparse
import os
import sys

from elasticsearch.helpers import bulk
from sentence_transformers import SentenceTransformer

sys.path.insert(
    0,
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
)

from webapp.bootstrap import create_es_client
from webapp.config import load_config


def has_embedding(source, field_name):
    value = (source or {}).get(field_name)
    return isinstance(value, (list, tuple)) and len(value) > 0


def build_embedding_work_items(hits, overwrite=False):
    work_items = []
    for hit in hits:
        source = hit.get("_source") or {}
        title = source.get("title") or ""
        content = (source.get("content_full") or source.get("content") or "")[:512]
        needs_title = bool(title) and (overwrite or not has_embedding(source, "title_emb"))
        needs_content = overwrite or not has_embedding(source, "content_emb")
        if not needs_title and not needs_content:
            continue
        work_items.append(
            {
                "id": hit["_id"],
                "title": title,
                "content": content,
                "needs_title": needs_title,
                "needs_content": needs_content,
            }
        )
    return work_items


def build_update_actions(index, work_items, title_embeddings, content_embeddings):
    actions = []
    title_iter = iter(title_embeddings)
    content_iter = iter(content_embeddings)

    for item in work_items:
        doc = {}
        if item["needs_title"]:
            doc["title_emb"] = list(map(float, next(title_iter)))
        if item["needs_content"]:
            doc["content_emb"] = list(map(float, next(content_iter)))
        if doc:
            actions.append(
                {
                    "_op_type": "update",
                    "_index": index,
                    "_id": item["id"],
                    "doc": doc,
                }
            )

    return actions


def encode_embeddings(model, work_items, batch_size):
    title_texts = [item["title"] for item in work_items if item["needs_title"]]
    content_texts = [item["content"] for item in work_items if item["needs_content"]]

    title_embeddings = []
    content_embeddings = []
    if title_texts:
        title_embeddings = model.encode(
            title_texts,
            batch_size=batch_size,
            show_progress_bar=False,
        )
    if content_texts:
        content_embeddings = model.encode(
            content_texts,
            batch_size=batch_size,
            show_progress_bar=False,
        )
    return title_embeddings, content_embeddings


def precompute(index, limit, batch_size, scan_size=500, overwrite=False):
    config = load_config()
    es = create_es_client(config)

    fetched = 0
    updated = 0
    skipped = 0

    response = es.search(index=index, size=scan_size, query={"match_all": {}}, scroll="2m")
    scroll_id = response.get("_scroll_id")
    model = SentenceTransformer(config.bi_encoder_model_name)

    while True:
        hits = response["hits"]["hits"]
        if not hits:
            break

        if limit:
            remaining = max(0, limit - fetched)
            hits = hits[:remaining]
        if not hits:
            break

        work_items = build_embedding_work_items(hits, overwrite=overwrite)
        skipped += len(hits) - len(work_items)

        if work_items:
            title_embeddings, content_embeddings = encode_embeddings(model, work_items, batch_size)
            actions = build_update_actions(index, work_items, title_embeddings, content_embeddings)
            if actions:
                bulk(es, actions)
                updated += len(actions)

        fetched += len(hits)
        if limit and fetched >= limit:
            break

        response = es.scroll(scroll_id=scroll_id, scroll="2m")
        scroll_id = response.get("_scroll_id")

    print(
        f"Embedding backfill finished for index '{index}': "
        f"fetched={fetched}, updated={updated}, skipped={skipped}, overwrite={overwrite}"
    )


def main():
    config = load_config()
    parser = argparse.ArgumentParser(description="Backfill bi-encoder embeddings into Elasticsearch documents")
    parser.add_argument("--index", default=config.index_name)
    parser.add_argument("--limit", type=int, default=5000)
    parser.add_argument("--batch", type=int, default=64, help="SentenceTransformer encode batch size")
    parser.add_argument("--scan-size", type=int, default=500, help="Elasticsearch scroll page size")
    parser.add_argument("--overwrite", action="store_true", help="Recompute embeddings even if fields already exist")
    args = parser.parse_args()
    try:
        precompute(
            index=args.index,
            limit=args.limit,
            batch_size=args.batch,
            scan_size=args.scan_size,
            overwrite=args.overwrite,
        )
    except Exception as exc:
        print(f"Failed to backfill embeddings for index '{args.index}': {exc}")
        print(
            "Hint: ensure Elasticsearch is running, set SEARCHENGINE_ES_HOST/PORT "
            "to a reachable endpoint if you are not inside docker-compose, and make sure "
            "the bi-encoder model has been downloaded locally."
        )
        raise SystemExit(1)


if __name__ == "__main__":
    main()
