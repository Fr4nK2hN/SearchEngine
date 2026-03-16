def log_client_events(logger, events):
    if not events:
        return {"status": "error", "message": "No data provided"}, 400

    if isinstance(events, dict):
        events = [events]
    if not isinstance(events, list):
        return {"status": "error", "message": "Invalid payload format"}, 400

    for event in events:
        if not isinstance(event, dict):
            continue
        logger.info("client_event", extra=event)

    return {"status": "success"}, 200


def build_click_log_entry(data):
    return {
        "event": "result_click_confirmed",
        "sessionId": data.get("session_id"),
        "search_id": data.get("search_id"),
        "doc_id": data.get("doc_id"),
        "rank": data.get("rank"),
        "query": data.get("query"),
    }
