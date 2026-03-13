#!/usr/bin/env python3
"""
Replay a deterministic user feedback session against the running app API.
This is used by the defense demo script to produce stable behavioral metrics.
"""

import argparse
import json
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import Request, urlopen


def iso_now():
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def http_get_json(base_url, path, params=None, timeout=20):
    query = ("?" + urlencode(params)) if params else ""
    with urlopen(base_url + path + query, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def http_post_json(base_url, path, payload, timeout=20):
    data = json.dumps(payload).encode("utf-8")
    req = Request(
        base_url + path,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urlopen(req, timeout=timeout) as resp:
        body = resp.read().decode("utf-8")
        return resp.status, (json.loads(body) if body else {})


def load_queries(path, num_queries):
    fallback = [
        "what is rba",
        "python language",
        "how long do you need for sydney",
        "was ronald reagan a democrat",
    ]
    if not path:
        return fallback[:num_queries]

    p = Path(path)
    if not p.exists():
        return fallback[:num_queries]

    try:
        data = json.loads(p.read_text(encoding="utf-8"))
        if isinstance(data, list):
            queries = []
            for item in data:
                if isinstance(item, dict):
                    q = str(item.get("query", "")).strip()
                else:
                    q = str(item).strip()
                if q:
                    queries.append(q)
            if queries:
                return queries[:num_queries]
        if isinstance(data, dict):
            baseline = data.get("baseline_queries")
            if isinstance(baseline, list):
                queries = []
                for item in baseline:
                    if isinstance(item, dict):
                        q = str(item.get("query", "")).strip()
                    else:
                        q = str(item).strip()
                    if q:
                        queries.append(q)
                if queries:
                    return queries[:num_queries]
    except Exception:
        pass
    return fallback[:num_queries]


def pick_rank_candidates(result_count, index, max_clicks):
    """Generate stable but non-trivial click ranks."""
    if result_count <= 0:
        return []
    candidate = [1, 2, 3, 5]
    rotated = candidate[index % len(candidate):] + candidate[: index % len(candidate)]
    ranks = []
    for r in rotated:
        if r <= result_count and r not in ranks:
            ranks.append(r)
        if len(ranks) >= max_clicks:
            break
    if not ranks and result_count > 0:
        ranks = [1]
    return ranks


def main():
    parser = argparse.ArgumentParser(description="Replay deterministic feedback events")
    parser.add_argument("--base-url", default="http://127.0.0.1:5000")
    parser.add_argument("--queries-file", default="data/sample_queries.json")
    parser.add_argument("--num-queries", type=int, default=8)
    parser.add_argument(
        "--mode-cycle",
        default="ltr,cross_encoder,hybrid,baseline",
        help="Comma separated ranking modes to cycle through",
    )
    parser.add_argument(
        "--click-policy",
        choices=["all", "alternate", "none"],
        default="alternate",
        help="Whether each search gets a click",
    )
    parser.add_argument(
        "--max-clicks-per-query",
        type=int,
        default=1,
        help="Maximum click events per clicked query",
    )
    parser.add_argument(
        "--trace-path",
        default="",
        help="Optional path to save detailed replay trace JSON",
    )
    parser.add_argument("--sleep-ms", type=int, default=80)
    args = parser.parse_args()

    base_url = args.base_url.rstrip("/")
    modes = [m.strip() for m in args.mode_cycle.split(",") if m.strip()]
    if not modes:
        raise ValueError("mode_cycle is empty")

    queries = load_queries(args.queries_file, args.num_queries)
    if not queries:
        raise ValueError("no queries available")

    session_id = f"defense-{int(time.time())}-{uuid.uuid4().hex[:8]}"
    events = []
    click_count = 0
    search_count = 0
    start_ms = int(time.time() * 1000)
    search_traces = []

    for i, query in enumerate(queries):
        mode = modes[i % len(modes)]
        submit_ts = iso_now()
        events.append(
            {
                "type": "query_submitted",
                "sessionId": session_id,
                "query": query,
                "mode": mode,
                "timestamp": submit_ts,
            }
        )

        t0 = int(time.time() * 1000)
        search = http_get_json(base_url, "/search", {"q": query, "mode": mode})
        t1 = int(time.time() * 1000)
        results = search.get("results", [])
        search_id = search.get("search_id")
        search_traces.append(
            {
                "query": query,
                "mode": mode,
                "search_id": search_id,
                "result_count": len(results),
                "results": results,
            }
        )

        search_count += 1
        events.append(
            {
                "type": "search_completed",
                "sessionId": session_id,
                "searchId": search_id,
                "query": query,
                "mode": mode,
                "resultCount": len(results),
                "searchDuration": max(0, t1 - t0),
                "timestamp": iso_now(),
            }
        )

        should_click = (
            args.click_policy == "all"
            or (args.click_policy == "alternate" and i % 2 == 0)
        )

        if should_click and results:
            max_clicks = max(1, int(args.max_clicks_per_query))
            ranks = pick_rank_candidates(len(results), i, max_clicks)
            for rank in ranks:
                clicked = results[rank - 1]
                doc_id = clicked.get("_id")

                events.append(
                    {
                        "type": "result_clicked",
                        "sessionId": session_id,
                        "searchId": search_id,
                        "query": query,
                        "docId": doc_id,
                        "rank": rank,
                        "score": clicked.get("_score"),
                        "timestamp": iso_now(),
                    }
                )
                http_post_json(
                    base_url,
                    "/track_click",
                    {
                        "session_id": session_id,
                        "search_id": search_id,
                        "doc_id": doc_id,
                        "rank": rank,
                        "query": query,
                    },
                )
                click_count += 1
        else:
            events.append(
                {
                    "type": "query_abandoned",
                    "sessionId": session_id,
                    "query": query,
                    "timeSpent": max(0, int(time.time() * 1000) - t0),
                    "timestamp": iso_now(),
                }
            )

        if args.sleep_ms > 0:
            time.sleep(args.sleep_ms / 1000.0)

    events.append(
        {
            "type": "session_end",
            "sessionId": session_id,
            "totalDuration": max(0, int(time.time() * 1000) - start_ms),
            "finalQueryCount": len(queries),
            "finalClickCount": click_count,
            "timestamp": iso_now(),
        }
    )

    status, _ = http_post_json(base_url, "/log", events)
    if status != 200:
        raise RuntimeError(f"/log failed with status {status}")

    summary = {
        "session_id": session_id,
        "queries_sent": len(queries),
        "searches": search_count,
        "clicks": click_count,
        "events_sent": len(events),
    }

    if args.trace_path:
        trace_payload = {
            **summary,
            "base_url": base_url,
            "generated_at": iso_now(),
            "search_traces": search_traces,
            "events": events,
        }
        path = Path(args.trace_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(trace_payload, ensure_ascii=False, indent=2), encoding="utf-8")
        summary["trace_path"] = str(path)

    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
