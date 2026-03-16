def normalize_doc_ids(raw_ids):
    normalized = []
    for doc_id in raw_ids:
        if doc_id is None:
            continue
        text = str(doc_id).strip()
        if text:
            normalized.append(text)
    return normalized


def fetch_documents_by_ids(es, index_name, raw_ids):
    doc_ids = normalize_doc_ids(raw_ids)
    if not doc_ids:
        return []

    response = es.mget(index=index_name, body={"ids": doc_ids})
    docs_by_id = {}
    for item in response.get("docs", []):
        if item.get("found"):
            docs_by_id[str(item.get("_id"))] = {
                "_id": item.get("_id"),
                "_source": item.get("_source", {}),
            }

    return [docs_by_id[doc_id] for doc_id in doc_ids if doc_id in docs_by_id]
