import json
import math
import os


def parse_log_line_events(line):
    """从一行 JSON 日志中提取事件对象（兼容旧/新日志格式）。"""
    line = (line or "").strip()
    if not line:
        return []

    try:
        record = json.loads(line)
    except json.JSONDecodeError:
        return []

    if isinstance(record, list):
        return [item for item in record if isinstance(item, dict)]
    if not isinstance(record, dict):
        return []

    events = []

    message = record.get("message")
    if isinstance(message, str):
        message_str = message.strip()
        if message_str[:1] in ("{", "["):
            try:
                parsed = json.loads(message_str)
                if isinstance(parsed, dict):
                    events.append(parsed)
                elif isinstance(parsed, list):
                    events.extend(item for item in parsed if isinstance(item, dict))
            except json.JSONDecodeError:
                pass

    event_keys = {
        "type", "event", "searchId", "search_id", "sessionId",
        "query", "rankingMethod", "results_count", "result_ids",
    }
    if any(key in record for key in event_keys):
        events.append(record)

    return events


def load_events_from_log(log_file, limit=None):
    """读取并解析事件日志。limit 为只取最后 N 行。"""
    if not os.path.exists(log_file):
        return []

    with open(log_file, "r", encoding="utf-8") as handle:
        lines = handle.readlines()

    if limit and limit > 0:
        lines = lines[-limit:]

    events = []
    for line in lines:
        events.extend(parse_log_line_events(line))
    return events


def _to_positive_int(value):
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return None
    return parsed if parsed > 0 else None


def _to_non_negative_float(value):
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(parsed) or parsed < 0:
        return None
    return parsed


def _percentile(values, p):
    if not values:
        return 0.0
    arr = sorted(float(v) for v in values)
    if len(arr) == 1:
        return arr[0]
    pos = (len(arr) - 1) * (p / 100.0)
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return arr[lo]
    weight = pos - lo
    return arr[lo] * (1 - weight) + arr[hi] * weight


def build_event_summary(events_data):
    """构建导出/仪表盘共用的事件汇总。"""
    session_ids = {
        event.get("sessionId")
        for event in events_data
        if isinstance(event, dict) and event.get("sessionId")
    }

    summary = {
        "total_sessions": len(session_ids),
        "total_events": len(events_data),
        "event_types": {},
        "query_stats": {
            "total_queries": 0,
            "unique_queries": set(),
            "abandoned_queries": 0,
        },
        "interaction_stats": {
            "total_clicks": 0,
            "confirmed_clicks": 0,
            "total_scrolls": 0,
            "average_session_duration": 0,
        },
        "feedback_stats": {
            "total_searches": 0,
            "searches_with_click": 0,
            "ctr": 0.0,
            "ctr_at_1": 0.0,
            "ctr_at_3": 0.0,
            "ctr_at_10": 0.0,
            "avg_click_rank": 0.0,
            "median_click_rank": 0.0,
            "clicks_per_query": 0.0,
            "abandonment_rate": 0.0,
        },
        "latency_stats": {
            "sample_count": 0,
            "avg_total_ms": 0.0,
            "p95_total_ms": 0.0,
            "avg_retrieval_ms": 0.0,
            "avg_feature_ms": 0.0,
            "avg_inference_ms": 0.0,
            "by_ranking_method": {},
        },
        "adaptive_stats": {
            "total_routed": 0,
            "easy_count": 0,
            "hard_count": 0,
            "hard_rate": 0.0,
            "avg_confidence": 0.0,
            "model_routed": 0,
            "heuristic_routed": 0,
        },
    }

    session_durations = []
    completed_search_ids = []
    click_ranks_by_search = {}
    all_click_ranks = []
    latency_total = []
    latency_retrieval = []
    latency_feature = []
    latency_inference = []
    latency_by_method = {}
    adaptive_confidence = []

    for event in events_data:
        if not isinstance(event, dict):
            continue

        event_type = event.get("type") or event.get("event") or "unknown"
        summary["event_types"][event_type] = summary["event_types"].get(event_type, 0) + 1

        total_ms = _to_non_negative_float(event.get("total_ms"))
        if total_ms is not None:
            latency_total.append(total_ms)
            method = str(event.get("rankingMethod") or "unknown")
            latency_by_method.setdefault(method, []).append(total_ms)

            retrieval_ms = _to_non_negative_float(event.get("retrieval_ms"))
            feature_ms = _to_non_negative_float(event.get("feature_ms"))
            inference_ms = _to_non_negative_float(event.get("inference_ms"))
            if retrieval_ms is not None:
                latency_retrieval.append(retrieval_ms)
            if feature_ms is not None:
                latency_feature.append(feature_ms)
            if inference_ms is not None:
                latency_inference.append(inference_ms)

        route_label = str(event.get("route_label") or "").strip().lower()
        if route_label in ("easy", "hard"):
            summary["adaptive_stats"]["total_routed"] += 1
            if route_label == "easy":
                summary["adaptive_stats"]["easy_count"] += 1
            else:
                summary["adaptive_stats"]["hard_count"] += 1

            route_confidence = _to_non_negative_float(event.get("route_confidence"))
            if route_confidence is not None:
                adaptive_confidence.append(min(1.0, route_confidence))

            route_source = str(event.get("route_source") or "").strip().lower()
            if route_source == "model":
                summary["adaptive_stats"]["model_routed"] += 1
            elif route_source == "heuristic":
                summary["adaptive_stats"]["heuristic_routed"] += 1

        if event_type == "query_submitted" and event.get("sessionId"):
            summary["query_stats"]["total_queries"] += 1
            summary["query_stats"]["unique_queries"].add(event.get("query", ""))

        elif event_type == "query_abandoned":
            summary["query_stats"]["abandoned_queries"] += 1

        elif event_type == "search_completed" and event.get("sessionId"):
            summary["feedback_stats"]["total_searches"] += 1
            search_id = event.get("searchId") or event.get("search_id")
            if search_id:
                completed_search_ids.append(str(search_id))

        elif event_type == "result_clicked":
            summary["interaction_stats"]["total_clicks"] += 1

            rank = _to_positive_int(event.get("rank"))
            if rank is not None:
                all_click_ranks.append(rank)

                search_id = event.get("searchId") or event.get("search_id")
                if search_id:
                    sid = str(search_id)
                    click_ranks_by_search.setdefault(sid, []).append(rank)

        elif event_type == "result_click_confirmed":
            summary["interaction_stats"]["confirmed_clicks"] += 1

        elif event_type == "scroll_action":
            summary["interaction_stats"]["total_scrolls"] += 1

        elif event_type == "session_end":
            duration = event.get("totalDuration", 0)
            if isinstance(duration, (int, float)) and duration > 0:
                session_durations.append(duration)

    summary["query_stats"]["unique_queries"] = len(summary["query_stats"]["unique_queries"])

    if session_durations:
        summary["interaction_stats"]["average_session_duration"] = (
            sum(session_durations) / len(session_durations)
        )

    total_queries = summary["query_stats"]["total_queries"]
    total_clicks = summary["interaction_stats"]["total_clicks"]
    abandoned_queries = summary["query_stats"]["abandoned_queries"]

    if total_queries > 0:
        summary["feedback_stats"]["clicks_per_query"] = total_clicks / total_queries
        summary["feedback_stats"]["abandonment_rate"] = abandoned_queries / total_queries

    if completed_search_ids:
        clicked = 0
        clicked_at_1 = 0
        clicked_at_3 = 0
        clicked_at_10 = 0

        for search_id in completed_search_ids:
            ranks = click_ranks_by_search.get(search_id, [])
            if not ranks:
                continue

            clicked += 1
            min_rank = min(ranks)
            if min_rank <= 1:
                clicked_at_1 += 1
            if min_rank <= 3:
                clicked_at_3 += 1
            if min_rank <= 10:
                clicked_at_10 += 1

        denom = len(completed_search_ids)
        summary["feedback_stats"]["searches_with_click"] = clicked
        summary["feedback_stats"]["ctr"] = clicked / denom
        summary["feedback_stats"]["ctr_at_1"] = clicked_at_1 / denom
        summary["feedback_stats"]["ctr_at_3"] = clicked_at_3 / denom
        summary["feedback_stats"]["ctr_at_10"] = clicked_at_10 / denom

    if all_click_ranks:
        ranks_sorted = sorted(all_click_ranks)
        n = len(ranks_sorted)
        summary["feedback_stats"]["avg_click_rank"] = sum(ranks_sorted) / n
        if n % 2 == 1:
            summary["feedback_stats"]["median_click_rank"] = float(ranks_sorted[n // 2])
        else:
            mid = n // 2
            summary["feedback_stats"]["median_click_rank"] = (
                ranks_sorted[mid - 1] + ranks_sorted[mid]
            ) / 2.0

    if latency_total:
        summary["latency_stats"]["sample_count"] = len(latency_total)
        summary["latency_stats"]["avg_total_ms"] = sum(latency_total) / len(latency_total)
        summary["latency_stats"]["p95_total_ms"] = _percentile(latency_total, 95)
    if latency_retrieval:
        summary["latency_stats"]["avg_retrieval_ms"] = (
            sum(latency_retrieval) / len(latency_retrieval)
        )
    if latency_feature:
        summary["latency_stats"]["avg_feature_ms"] = sum(latency_feature) / len(latency_feature)
    if latency_inference:
        summary["latency_stats"]["avg_inference_ms"] = (
            sum(latency_inference) / len(latency_inference)
        )
    for method, vals in latency_by_method.items():
        summary["latency_stats"]["by_ranking_method"][method] = {
            "count": len(vals),
            "avg_total_ms": sum(vals) / len(vals),
            "p95_total_ms": _percentile(vals, 95),
        }

    routed = summary["adaptive_stats"]["total_routed"]
    if routed > 0:
        summary["adaptive_stats"]["hard_rate"] = summary["adaptive_stats"]["hard_count"] / routed
        if adaptive_confidence:
            summary["adaptive_stats"]["avg_confidence"] = (
                sum(adaptive_confidence) / len(adaptive_confidence)
            )

    return summary
