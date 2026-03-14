DEFAULT_RECALL_RELAX_THRESHOLD = 5


def build_es_query(query, relaxed=False, hl=False, size=50):
    """
    Build the shared Elasticsearch retrieval query used by both online search
    and offline training data generation.
    """
    tokens = [t for t in (query or "").strip().split() if t]
    is_short_single_term = len(tokens) == 1 and len(tokens[0]) <= 4
    fields = ["title^4", "content^2.5", "combined_text^1.5", "related_queries^2"]

    multi_match = {
        "query": query,
        "fields": fields,
        "type": "best_fields",
        "operator": "and" if len(tokens) >= 2 else "or",
    }
    if relaxed:
        multi_match["operator"] = "or"
        if len(tokens) >= 2:
            multi_match["minimum_should_match"] = "60%"
        if not is_short_single_term:
            multi_match["fuzziness"] = "AUTO"

    content_phrase = {
        "query": query,
        "slop": 2 if relaxed else 1,
        "boost": 1.2 if relaxed else 2.0,
    }
    title_phrase = {
        "query": query,
        "slop": 1,
        "boost": 1.8 if relaxed else 3.0,
    }
    related_query_match = {
        "query": query,
        "boost": 1.4 if relaxed else 2.2,
    }

    es_query = {
        "query": {
            "bool": {
                "should": [
                    {"multi_match": {**multi_match, "boost": 2.0 if relaxed else 4.0}},
                    {"match_phrase": {"title": title_phrase}},
                    {"match_phrase": {"content": content_phrase}},
                    {"match": {"related_queries": related_query_match}},
                ],
                "minimum_should_match": 1,
            }
        },
        "size": size,
    }
    if hl:
        es_query["highlight"] = {"fields": {"title": {}, "content": {}}}
    return es_query


def _normalize_relax_threshold(relax_threshold):
    try:
        return max(1, int(relax_threshold))
    except (TypeError, ValueError):
        return DEFAULT_RECALL_RELAX_THRESHOLD


def search_documents_with_fallback(
    es_client,
    query,
    size=50,
    hl=False,
    relax_threshold=DEFAULT_RECALL_RELAX_THRESHOLD,
    index_name="documents",
):
    """
    Run strict retrieval first, then relaxed retrieval when the strict result
    set is too small. Returns (hits, retrieval_strategy).
    """
    strict_query = build_es_query(query, relaxed=False, hl=hl, size=size)
    strict_response = es_client.search(index=index_name, body=strict_query)
    results = strict_response["hits"]["hits"]
    retrieval_strategy = "strict"

    if len(results) < _normalize_relax_threshold(relax_threshold):
        relaxed_query = build_es_query(query, relaxed=True, hl=hl, size=size)
        relaxed_response = es_client.search(index=index_name, body=relaxed_query)
        relaxed_results = relaxed_response["hits"]["hits"]
        if len(relaxed_results) > len(results):
            results = relaxed_results
            retrieval_strategy = "relaxed_fallback"

    return results, retrieval_strategy
