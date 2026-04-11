import argparse
import csv
import json
import os
import time
from collections import OrderedDict
from datetime import datetime, timezone
from urllib.parse import urlencode
from urllib.request import urlopen


DEFAULT_MODES = ["baseline", "ltr", "adaptive", "hybrid", "cross_encoder"]


def normalize_query(text):
    return " ".join(str(text or "").split())


def load_queries(path, max_queries=None):
    with open(path, "r", encoding="utf-8") as handle:
        raw = json.load(handle)
    queries = []
    seen = set()
    previous_labels = {}

    for item in raw:
        query = normalize_query(item.get("query"))
        if not query or query in seen:
            continue
        seen.add(query)
        queries.append(query)

        label_map = {}
        docs = item.get("documents") or []
        labels = item.get("relevance_labels") or []
        for doc, label in zip(docs, labels):
            if not isinstance(doc, dict):
                continue
            doc_id = doc.get("id") or doc.get("_id")
            if doc_id is None:
                continue
            try:
                label_map[str(doc_id)] = int(label)
            except (TypeError, ValueError):
                label_map[str(doc_id)] = ""
        previous_labels[query] = label_map

        if max_queries and len(queries) >= max_queries:
            break

    return queries, previous_labels


def fetch_search(base_url, query, mode, top_n, timeout):
    params = urlencode(
        {
            "q": query,
            "mode": mode,
            "session_id": f"pool-{mode}-{int(time.time() * 1000)}",
        }
    )
    url = base_url.rstrip("/") + "/search?" + params
    t0 = time.perf_counter()
    with urlopen(url, timeout=timeout) as response:
        payload = json.loads(response.read().decode("utf-8"))
    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    return payload.get("results", [])[:top_n], elapsed_ms


def compact_text(text, limit):
    text = " ".join(str(text or "").split())
    if len(text) <= limit:
        return text
    return text[: limit - 3] + "..."


def hit_field(hit, name):
    if hit.get(name):
        return hit.get(name)
    source = hit.get("_source")
    if isinstance(source, dict):
        return source.get(name)
    return None


def merge_candidate(pool, hit, mode, rank, query, previous_label):
    doc_id = str(hit.get("_id") or hit.get("id") or "")
    if not doc_id:
        return

    if doc_id not in pool:
        title = hit_field(hit, "title")
        snippet = hit_field(hit, "content") or hit_field(hit, "content_full") or hit_field(hit, "combined_text")
        pool[doc_id] = {
            "query": query,
            "doc_id": doc_id,
            "title": compact_text(title, 180),
            "snippet": compact_text(snippet, 600),
            "source_modes": [],
            "best_rank": rank,
            "rank_baseline": "",
            "rank_ltr": "",
            "rank_adaptive": "",
            "rank_hybrid": "",
            "rank_cross_encoder": "",
            "previous_label": previous_label.get(doc_id, ""),
            "manual_label": "",
            "notes": "",
        }

    row = pool[doc_id]
    if mode not in row["source_modes"]:
        row["source_modes"].append(mode)
    row["best_rank"] = min(int(row["best_rank"]), rank)
    rank_key = f"rank_{mode}"
    if rank_key in row and (row[rank_key] == "" or int(row[rank_key]) > rank):
        row[rank_key] = rank


def build_pool(queries, previous_labels, base_url, modes, top_n, timeout):
    rows = []
    timings = []

    for qidx, query in enumerate(queries, start=1):
        pool = OrderedDict()
        for mode in modes:
            results, elapsed_ms = fetch_search(base_url, query, mode, top_n, timeout)
            timings.append({"query": query, "mode": mode, "elapsed_ms": elapsed_ms})
            for rank, hit in enumerate(results, start=1):
                merge_candidate(
                    pool=pool,
                    hit=hit,
                    mode=mode,
                    rank=rank,
                    query=query,
                    previous_label=previous_labels.get(query, {}),
                )

        for local_idx, row in enumerate(sorted(pool.values(), key=lambda x: (int(x["best_rank"]), x["doc_id"])), start=1):
            row["query_id"] = qidx
            row["pool_rank"] = local_idx
            row["source_modes"] = ",".join(row["source_modes"])
            rows.append(row)

    return rows, timings


def write_csv(path, rows):
    fieldnames = [
        "query_id",
        "query",
        "pool_rank",
        "doc_id",
        "title",
        "snippet",
        "source_modes",
        "best_rank",
        "rank_baseline",
        "rank_ltr",
        "rank_adaptive",
        "rank_hybrid",
        "rank_cross_encoder",
        "previous_label",
        "manual_label",
        "notes",
    ]
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_readme(path, args, rows, timings, query_count):
    judged_from_old = sum(1 for row in rows if row.get("previous_label") not in ("", None))
    avg_pool_size = (len(rows) / query_count) if query_count else 0.0
    lines = [
        "# Manual Relevance Judgment Pool",
        "",
        f"- Generated at: `{datetime.now(timezone.utc).isoformat()}`",
        f"- Query source: `{args.data}`",
        f"- Search API: `{args.base_url}`",
        f"- Modes: `{args.modes}`",
        f"- Top-N per mode: `{args.top_n}`",
        f"- Query count: `{query_count}`",
        f"- Candidate rows: `{len(rows)}`",
        f"- Average pooled candidates per query: `{avg_pool_size:.2f}`",
        f"- Rows with previous weak labels: `{judged_from_old}`",
        "",
        "## How to Label",
        "",
        "Fill `manual_label` for every row. Use a small graded scale:",
        "",
        "| Label | Meaning |",
        "| ---: | --- |",
        "| 0 | Irrelevant or misleading |",
        "| 1 | Marginally related, only weak keyword overlap |",
        "| 2 | Partially relevant, useful but incomplete |",
        "| 3 | Highly relevant and directly answers the query |",
        "",
        "Do not copy `previous_label` blindly. It is only a weak reference from the earlier dataset and should be corrected when it is noisy.",
        "",
        "After labeling, run:",
        "",
        "```bash",
        "docker exec app python tools/analysis/evaluate_manual_judgments.py \\",
        f"  --judgments {args.out_dir}/judgment_pool.csv \\",
        f"  --out-dir {args.out_dir}/eval_after_labeling",
        "```",
        "",
        "The evaluator skips incomplete queries by default. If you label a smaller subset, label all rows for those selected queries instead of labeling scattered rows.",
        "",
        "## Why This Solves the Low NDCG Problem",
        "",
        "- The pool merges Top-N results from multiple rankers, so the judged set covers more documents than the old 10-doc candidate list.",
        "- Manual labels increase reliable positive examples and reduce weak-label noise.",
        "- End-to-end NDCG becomes more meaningful because returned Top-10 documents are likely to exist in the judged pool.",
    ]
    with open(path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/ltr_training_data_feedback_combined.json")
    parser.add_argument("--base-url", default="http://127.0.0.1:5000")
    parser.add_argument("--modes", default=",".join(DEFAULT_MODES))
    parser.add_argument("--top-n", type=int, default=10)
    parser.add_argument("--max-queries", type=int, default=0)
    parser.add_argument("--timeout", type=float, default=60.0)
    parser.add_argument("--out-dir", default="reports/manual_relevance_pool_20260411")
    args = parser.parse_args()

    modes = [x.strip() for x in args.modes.split(",") if x.strip()]
    max_queries = args.max_queries if args.max_queries > 0 else None
    queries, previous_labels = load_queries(args.data, max_queries=max_queries)
    if not queries:
        raise RuntimeError(f"no queries loaded from {args.data}")

    os.makedirs(args.out_dir, exist_ok=True)
    rows, timings = build_pool(
        queries=queries,
        previous_labels=previous_labels,
        base_url=args.base_url,
        modes=modes,
        top_n=args.top_n,
        timeout=args.timeout,
    )

    csv_path = os.path.join(args.out_dir, "judgment_pool.csv")
    json_path = os.path.join(args.out_dir, "judgment_pool.json")
    timing_path = os.path.join(args.out_dir, "pool_request_timings.json")
    readme_path = os.path.join(args.out_dir, "README.md")

    write_csv(csv_path, rows)
    with open(json_path, "w", encoding="utf-8") as handle:
        json.dump(rows, handle, indent=2, ensure_ascii=False)
    with open(timing_path, "w", encoding="utf-8") as handle:
        json.dump(timings, handle, indent=2, ensure_ascii=False)
    write_readme(readme_path, args, rows, timings, len(queries))

    print(f"judgment_pool_csv: {csv_path}")
    print(f"judgment_pool_json: {json_path}")
    print(f"readme: {readme_path}")
    print(f"queries: {len(queries)}")
    print(f"candidate_rows: {len(rows)}")
    print(f"avg_candidates_per_query: {len(rows) / len(queries):.2f}")


if __name__ == "__main__":
    main()
