#!/usr/bin/env python3
"""
Build LTR-style training data from user feedback events.

Expected inputs:
- export_data.json from /export_data (contains raw_events)
- optional replay trace JSON from replay_feedback_simulation.py

Output format:
[
  {
    "query": "...",
    "documents": [...],
    "relevance_labels": [...]
  }
]
"""

import argparse
import json
import os
from collections import defaultdict
from datetime import datetime, timezone
from urllib.parse import urlencode
from urllib.request import urlopen


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def normalize_events(export_obj):
    if isinstance(export_obj, dict):
        if isinstance(export_obj.get("raw_events"), list):
            return [e for e in export_obj["raw_events"] if isinstance(e, dict)]
        if isinstance(export_obj.get("events"), list):
            return [e for e in export_obj["events"] if isinstance(e, dict)]
    return []


def event_type(event):
    return event.get("type") or event.get("event") or "unknown"


def make_doc_from_result(hit):
    source = hit.get("_source", {}) if isinstance(hit, dict) else {}
    return {
        "id": hit.get("_id"),
        "title": source.get("title", ""),
        "content": source.get("content", ""),
        "related_queries": source.get("related_queries", []),
        "es_score": float(hit.get("_score", 0.0) or 0.0),
    }


def click_rank_to_label(rank):
    if rank <= 1:
        return 4
    if rank <= 3:
        return 3
    if rank <= 5:
        return 2
    return 1


def _to_positive_int(value):
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return None
    return parsed if parsed > 0 else None


def _pick_first(*values):
    for value in values:
        if value is None:
            continue
        if isinstance(value, str) and not value.strip():
            continue
        return value
    return None


def fetch_search_results(base_url, query, mode, rerank_top_n=None, timeout=20):
    params = {"q": query, "mode": mode}
    top_n = _to_positive_int(rerank_top_n)
    if top_n is not None:
        params["rerank_top_n"] = top_n
    params = urlencode(params)
    with urlopen(base_url.rstrip("/") + "/search?" + params, timeout=timeout) as resp:
        payload = json.loads(resp.read().decode("utf-8"))
    return payload.get("results", [])


def fetch_documents_by_ids(base_url, doc_ids, doc_scores=None, timeout=20):
    params = urlencode({"id": [str(doc_id) for doc_id in doc_ids]}, doseq=True)
    with urlopen(base_url.rstrip("/") + "/documents_by_ids?" + params, timeout=timeout) as resp:
        payload = json.loads(resp.read().decode("utf-8"))

    documents = payload.get("documents", [])
    if not isinstance(documents, list):
        return []

    score_map = {}
    if isinstance(doc_scores, list):
        for idx, doc_id in enumerate(doc_ids):
            if idx >= len(doc_scores):
                break
            try:
                score_map[str(doc_id)] = float(doc_scores[idx])
            except (TypeError, ValueError):
                continue

    ordered = []
    for hit in documents:
        if not isinstance(hit, dict):
            continue
        hit_id = str(hit.get("_id"))
        source = hit.get("_source", {}) if isinstance(hit.get("_source"), dict) else {}
        ordered.append(
            {
                "_id": hit_id,
                "_score": score_map.get(hit_id, 0.0),
                "_source": source,
            }
        )
    return ordered


def _rebuild_from_trace_hits(trace_hits, result_ids, result_scores):
    hits_by_id = {}
    for hit in trace_hits or []:
        if isinstance(hit, dict) and hit.get("_id") is not None:
            hits_by_id[str(hit.get("_id"))] = hit

    score_map = {}
    if isinstance(result_scores, list):
        for idx, doc_id in enumerate(result_ids):
            if idx >= len(result_scores):
                break
            try:
                score_map[str(doc_id)] = float(result_scores[idx])
            except (TypeError, ValueError):
                continue

    ordered = []
    for doc_id in result_ids:
        hit = hits_by_id.get(str(doc_id))
        if not hit:
            continue
        rebuilt = dict(hit)
        rebuilt["_score"] = score_map.get(str(doc_id), float(hit.get("_score", 0.0) or 0.0))
        ordered.append(rebuilt)
    return ordered


def build_from_feedback(
    events,
    trace_by_search_id,
    base_url,
    candidate_source,
    fallback_mode,
    use_confirmed_only,
    min_clicks_per_search,
):
    searches = {}
    serp_results_by_search_id = {}
    confirmed_clicks = defaultdict(list)
    raw_clicks = defaultdict(list)

    for ev in events:
        t = event_type(ev)
        sid = ev.get("searchId") or ev.get("search_id")

        if sid:
            key = str(sid)
            search = searches.setdefault(
                key,
                {
                    "search_id": key,
                    "query": None,
                    "requested_mode": None,
                    "route_selected_mode": None,
                    "route_label": None,
                    "route_guardrail": None,
                    "route_rerank_top_n": None,
                    "route_source": None,
                    "result_ids": None,
                    "result_scores": None,
                    "session_id": None,
                },
            )
            search["query"] = _pick_first(search.get("query"), ev.get("query"))
            search["requested_mode"] = _pick_first(search.get("requested_mode"), ev.get("mode"))
            search["route_selected_mode"] = _pick_first(
                search.get("route_selected_mode"),
                ev.get("route_selected_mode"),
                ev.get("selected_mode"),
            )
            search["route_label"] = _pick_first(search.get("route_label"), ev.get("route_label"))
            search["route_guardrail"] = _pick_first(
                search.get("route_guardrail"),
                ev.get("route_guardrail"),
            )
            search["route_source"] = _pick_first(
                search.get("route_source"),
                ev.get("route_source"),
            )
            search["route_rerank_top_n"] = _pick_first(
                search.get("route_rerank_top_n"),
                _to_positive_int(ev.get("route_rerank_top_n")),
                _to_positive_int(ev.get("rerank_top_n")),
            )
            result_ids = ev.get("result_ids")
            if not isinstance(result_ids, list):
                result_ids = ev.get("results")
            if isinstance(result_ids, list) and result_ids:
                search["result_ids"] = result_ids

            result_scores = ev.get("result_scores")
            if isinstance(result_scores, list) and result_scores:
                search["result_scores"] = result_scores
            search["session_id"] = _pick_first(
                search.get("session_id"),
                ev.get("sessionId"),
                ev.get("session_id"),
            )

        if t == "search_completed":
            if sid:
                searches[str(sid)]["session_id"] = _pick_first(
                    searches[str(sid)].get("session_id"),
                    ev.get("sessionId"),
                )
        elif t == "serp_impression":
            if sid and isinstance(ev.get("result_docs"), list):
                serp_results_by_search_id[str(sid)] = ev.get("result_docs")
        elif t == "result_click_confirmed":
            if sid:
                confirmed_clicks[str(sid)].append(ev)
        elif t == "result_clicked":
            if sid:
                raw_clicks[str(sid)].append(ev)

    samples = []
    dropped = defaultdict(int)

    for sid, search in searches.items():
        query = search.get("query")
        requested_mode = search.get("requested_mode") or fallback_mode
        route_selected_mode = search.get("route_selected_mode")
        replay_mode = requested_mode
        if requested_mode == "adaptive" and route_selected_mode:
            replay_mode = route_selected_mode
        if not query:
            dropped["missing_query"] += 1
            continue

        click_events = confirmed_clicks.get(sid, [])
        if not click_events and not use_confirmed_only:
            click_events = raw_clicks.get(sid, [])

        if len(click_events) < min_clicks_per_search:
            dropped["not_enough_clicks"] += 1
            continue

        results = None
        actual_candidate_source = None

        result_ids = search.get("result_ids") if isinstance(search.get("result_ids"), list) else None
        result_scores = search.get("result_scores") if isinstance(search.get("result_scores"), list) else None
        if result_ids:
            trace_hits = trace_by_search_id.get(sid)
            if isinstance(trace_hits, list):
                results = _rebuild_from_trace_hits(trace_hits, result_ids, result_scores)
                actual_candidate_source = "server_result_ids_trace" if results else None
            if results is None and base_url:
                try:
                    results = fetch_documents_by_ids(base_url, result_ids, result_scores)
                    actual_candidate_source = "server_result_ids_api" if results else None
                except Exception:
                    results = None
                    actual_candidate_source = None

        if results is None:
            results = serp_results_by_search_id.get(sid)
            actual_candidate_source = "serp_event" if results is not None else None

        if results is None and candidate_source == "trace":
            results = trace_by_search_id.get(sid)
            actual_candidate_source = "trace" if results is not None else None
        if results is None:
            if not base_url:
                dropped["missing_results_no_base_url"] += 1
                continue
            try:
                results = fetch_search_results(
                    base_url,
                    query,
                    mode=replay_mode,
                    rerank_top_n=search.get("route_rerank_top_n"),
                )
                actual_candidate_source = "search"
            except Exception:
                dropped["fetch_failed"] += 1
                continue

        if not results:
            dropped["empty_results"] += 1
            continue

        documents = [make_doc_from_result(hit) for hit in results]
        labels = [0] * len(documents)
        doc_idx = {str(doc.get("id")): i for i, doc in enumerate(documents)}

        for clk in click_events:
            doc_id = clk.get("doc_id") or clk.get("docId")
            rank = clk.get("rank")
            if not doc_id:
                continue
            key = str(doc_id)
            if key not in doc_idx:
                continue
            try:
                rank_value = int(rank)
                if rank_value <= 0:
                    rank_value = 10
            except Exception:
                rank_value = 10
            idx = doc_idx[key]
            labels[idx] = max(labels[idx], click_rank_to_label(rank_value))

        if max(labels) <= 0:
            dropped["click_not_in_candidates"] += 1
            continue

        samples.append(
            {
                "query": query,
                "documents": documents,
                "relevance_labels": labels,
                "meta": {
                    "search_id": sid,
                    "mode": replay_mode,
                    "requested_mode": requested_mode,
                    "route_selected_mode": route_selected_mode,
                    "route_label": search.get("route_label"),
                    "route_guardrail": search.get("route_guardrail"),
                    "route_rerank_top_n": search.get("route_rerank_top_n"),
                    "route_source": search.get("route_source"),
                    "candidate_source": actual_candidate_source,
                    "session_id": search.get("session_id"),
                    "click_count": len(click_events),
                },
            }
        )

    return samples, dropped, {
        "search_count": len(searches),
        "searches_with_confirmed_click": sum(1 for sid in searches if confirmed_clicks.get(sid)),
        "searches_with_raw_click": sum(1 for sid in searches if raw_clicks.get(sid)),
    }


def main():
    parser = argparse.ArgumentParser(description="Build feedback-based LTR dataset")
    parser.add_argument("--export-json", required=True, help="Path to export_data.json")
    parser.add_argument("--trace-json", default="", help="Optional replay trace JSON")
    parser.add_argument("--output", default="data/ltr_training_data_feedback.json")
    parser.add_argument("--report", default="", help="Optional report output path")
    parser.add_argument(
        "--use-confirmed-only",
        action="store_true",
        help="Use only result_click_confirmed events as supervision",
    )
    parser.add_argument("--min-clicks-per-search", type=int, default=1)
    parser.add_argument(
        "--fallback-mode",
        default="baseline",
        help="Mode used when fetching results by query",
    )
    parser.add_argument(
        "--candidate-source",
        choices=["trace", "search"],
        default="trace",
        help="Use replay trace results or re-fetch candidates from /search",
    )
    parser.add_argument(
        "--base-url",
        default="http://127.0.0.1:5000",
        help="Base URL for fallback /search fetching",
    )
    args = parser.parse_args()

    export_obj = load_json(args.export_json)
    events = normalize_events(export_obj)

    trace_by_search_id = {}
    if args.trace_json:
        trace_obj = load_json(args.trace_json)
        traces = trace_obj.get("search_traces", [])
        for item in traces:
            if not isinstance(item, dict):
                continue
            sid = item.get("search_id")
            results = item.get("results")
            if sid and isinstance(results, list):
                trace_by_search_id[str(sid)] = results

    samples, dropped, stats = build_from_feedback(
        events=events,
        trace_by_search_id=trace_by_search_id,
        base_url=args.base_url,
        candidate_source=args.candidate_source,
        fallback_mode=args.fallback_mode,
        use_confirmed_only=args.use_confirmed_only,
        min_clicks_per_search=max(1, int(args.min_clicks_per_search)),
    )

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(samples, f, indent=2, ensure_ascii=False)

    label_dist = defaultdict(int)
    for item in samples:
        for v in item.get("relevance_labels", []):
            label_dist[int(v)] += 1

    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "export_json": args.export_json,
        "trace_json": args.trace_json or None,
        "output": args.output,
        "input_event_count": len(events),
        "search_stats": stats,
        "sample_count": len(samples),
        "label_distribution": dict(sorted(label_dist.items())),
        "dropped_counts": dict(sorted(dropped.items())),
        "args": {
            "use_confirmed_only": bool(args.use_confirmed_only),
            "min_clicks_per_search": int(args.min_clicks_per_search),
            "fallback_mode": args.fallback_mode,
            "candidate_source": args.candidate_source,
            "base_url": args.base_url,
        },
    }

    report_path = args.report or f"{args.output}.report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print("Feedback dataset build completed.")
    print(f"- samples: {len(samples)}")
    print(f"- labels: {dict(sorted(label_dist.items()))}")
    print(f"- output: {args.output}")
    print(f"- report: {report_path}")


if __name__ == "__main__":
    main()
