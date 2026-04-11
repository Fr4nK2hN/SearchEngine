import argparse
import csv
import json
import math
import os
import time
from datetime import datetime, timezone
from urllib.parse import urlencode
from urllib.request import urlopen


DEFAULT_MODES = ["baseline", "ltr", "adaptive", "hybrid", "cross_encoder"]


def load_dataset(path):
    with open(path, "r", encoding="utf-8") as handle:
        raw = json.load(handle)
    items = []
    for item in raw:
        query = item.get("query")
        docs = item.get("documents")
        labels = item.get("relevance_labels")
        if not isinstance(query, str) or not isinstance(docs, list) or not isinstance(labels, list):
            continue
        label_map = {}
        for doc, label in zip(docs, labels):
            if not isinstance(doc, dict):
                continue
            doc_id = doc.get("id") or doc.get("_id")
            if doc_id is None:
                continue
            try:
                label_map[str(doc_id)] = int(label)
            except (TypeError, ValueError):
                label_map[str(doc_id)] = 0
        if label_map:
            items.append({"query": " ".join(query.split()), "label_map": label_map})
    return items


def dcg_at_k(labels, k):
    total = 0.0
    for idx, rel in enumerate(labels[:k]):
        total += (2.0 ** float(rel) - 1.0) / math.log2(idx + 2.0)
    return total


def ndcg_at_k(returned_labels, missing_labels, k=10):
    dcg = dcg_at_k(returned_labels, k)
    ideal = sorted(list(returned_labels) + list(missing_labels), reverse=True)
    idcg = dcg_at_k(ideal, k)
    return (dcg / idcg) if idcg > 0 else 0.0


def percentile(values, p):
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
    return arr[lo] * (hi - pos) + arr[hi] * (pos - lo)


def fetch_search(base_url, query, mode, timeout):
    params = urlencode({"q": query, "mode": mode, "session_id": f"eval-{int(time.time())}"})
    url = base_url.rstrip("/") + "/search?" + params
    t0 = time.perf_counter()
    with urlopen(url, timeout=timeout) as response:
        payload = json.loads(response.read().decode("utf-8"))
    t1 = time.perf_counter()
    return payload, (t1 - t0) * 1000.0


def evaluate(items, base_url, modes, top_k, timeout):
    rows = []
    per_query = []

    for mode in modes:
        ndcgs = []
        latencies = []
        result_counts = []
        route_counts = {}

        for item in items:
            payload, latency_ms = fetch_search(base_url, item["query"], mode, timeout)
            results = payload.get("results", [])
            returned_ids = [str(hit.get("_id")) for hit in results[:top_k] if hit.get("_id") is not None]
            returned_set = set(returned_ids)
            returned_labels = [int(item["label_map"].get(doc_id, 0)) for doc_id in returned_ids]
            missing_labels = [
                int(label)
                for doc_id, label in item["label_map"].items()
                if int(label) > 0 and doc_id not in returned_set
            ]
            score = ndcg_at_k(returned_labels, missing_labels, top_k)
            ndcgs.append(score)
            latencies.append(latency_ms)
            result_counts.append(len(results))

            routing = payload.get("routing") or {}
            route_key = routing.get("selected_mode") or routing.get("route_label") or mode
            route_counts[route_key] = route_counts.get(route_key, 0) + 1

            per_query.append(
                {
                    "mode": mode,
                    "query": item["query"],
                    "ndcg@10": score,
                    "latency_ms": latency_ms,
                    "result_count": len(results),
                    "route": route_key,
                }
            )

        rows.append(
            {
                "mode": mode,
                "query_count": len(items),
                "ndcg@10": sum(ndcgs) / len(ndcgs) if ndcgs else 0.0,
                "latency_avg_ms": sum(latencies) / len(latencies) if latencies else 0.0,
                "latency_p50_ms": percentile(latencies, 50),
                "latency_p95_ms": percentile(latencies, 95),
                "result_count_avg": sum(result_counts) / len(result_counts) if result_counts else 0.0,
                "route_counts": route_counts,
            }
        )
    return rows, per_query


def write_csv(path, rows):
    if not rows:
        return
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_svg(path, rows):
    width = 1200
    height = 680
    left = 120
    top = 120
    chart_w = 940
    chart_h = 190
    gap = 260
    max_latency = max((row["latency_avg_ms"] for row in rows), default=1.0)
    if max_latency <= 0:
        max_latency = 1.0
    colors = {
        "baseline": "#0b6e4f",
        "ltr": "#7c3aed",
        "adaptive": "#2563eb",
        "hybrid": "#d97706",
        "cross_encoder": "#e11d48",
    }
    bar_w = 110
    step = chart_w / max(1, len(rows))

    def bar_group(metric, y0, max_v, label):
        parts = []
        parts.append(f'<text x="{left}" y="{y0 - 38}" class="section">{label}</text>')
        for frac in [0.0, 0.25, 0.5, 0.75, 1.0]:
            y = y0 + chart_h - chart_h * frac
            parts.append(f'<line x1="{left}" y1="{y:.1f}" x2="{left + chart_w}" y2="{y:.1f}" class="grid"/>')
            parts.append(f'<text x="{left - 16}" y="{y + 5:.1f}" text-anchor="end" class="axis">{max_v * frac:.2f}</text>')
        for idx, row in enumerate(rows):
            x = left + step * idx + (step - bar_w) / 2
            value = float(row[metric])
            h = chart_h * (value / max_v) if max_v > 0 else 0
            y = y0 + chart_h - h
            color = colors.get(row["mode"], "#475569")
            parts.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_w}" height="{h:.1f}" rx="10" fill="{color}"/>')
            value_text = f"{value:.4f}" if metric == "ndcg@10" else f"{value:.1f}"
            parts.append(f'<text x="{x + bar_w / 2:.1f}" y="{y - 10:.1f}" text-anchor="middle" class="value">{value_text}</text>')
            mode_label = row["mode"].replace("_", "-")
            parts.append(f'<text x="{x + bar_w / 2:.1f}" y="{y0 + chart_h + 30}" text-anchor="middle" class="axis">{mode_label}</text>')
        return "".join(parts)

    svg = f'''<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
<style>
text {{ font-family: Arial, Helvetica, sans-serif; fill: #1f2937; }}
.title {{ font-size: 26px; font-weight: 700; }}
.subtitle {{ font-size: 14px; fill: #64748b; }}
.section {{ font-size: 18px; font-weight: 700; }}
.axis {{ font-size: 12px; fill: #64748b; }}
.value {{ font-size: 13px; font-weight: 700; }}
.grid {{ stroke: #e2e8f0; stroke-width: 1; }}
</style>
<rect x="0" y="0" width="{width}" height="{height}" fill="none"/>
<text x="{width / 2}" y="44" text-anchor="middle" class="title">Latest Search Quality and Latency Evaluation</text>
<text x="{width / 2}" y="70" text-anchor="middle" class="subtitle">Metric: NDCG@10 and average client response time</text>
{bar_group("ndcg@10", top, 1.0, "NDCG@10")}
{bar_group("latency_avg_ms", top + gap, max_latency * 1.1, "Average Response Time (ms)")}
</svg>'''
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(svg)


def write_markdown(path, rows, args):
    lines = [
        f"# Latest Search Quality and Latency Evaluation",
        "",
        f"- Generated at: `{datetime.now(timezone.utc).isoformat()}`",
        f"- Dataset: `{args.data}`",
        f"- Base URL: `{args.base_url}`",
        f"- Top-K: `{args.top_k}`",
        f"- Query count: `{rows[0]['query_count'] if rows else 0}`",
        "",
        "| Mode | NDCG@10 | Avg Response (ms) | P50 (ms) | P95 (ms) | Avg Results |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            f"| `{row['mode']}` | {row['ndcg@10']:.4f} | {row['latency_avg_ms']:.2f} | "
            f"{row['latency_p50_ms']:.2f} | {row['latency_p95_ms']:.2f} | {row['result_count_avg']:.2f} |"
        )
    lines.extend(
        [
            "",
            "## Notes",
            "",
            "- NDCG@10 is computed against the feedback-labeled dataset by matching returned document IDs.",
            "- Response time is measured as client-side HTTP elapsed time for the current `/search` API.",
            "- The chart is generated as an SVG for direct use in slides.",
        ]
    )
    with open(path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/ltr_training_data_feedback_combined.json")
    parser.add_argument("--base-url", default="http://127.0.0.1:5000")
    parser.add_argument("--modes", default=",".join(DEFAULT_MODES))
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--timeout", type=float, default=30.0)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--out-dir", default="reports/current_quality_latency_eval")
    args = parser.parse_args()

    modes = [x.strip() for x in args.modes.split(",") if x.strip()]
    items = load_dataset(args.data)
    if not items:
        raise RuntimeError(f"no valid evaluation items loaded from {args.data}")

    os.makedirs(args.out_dir, exist_ok=True)

    for _ in range(max(0, args.warmup)):
        for mode in modes:
            fetch_search(args.base_url, items[0]["query"], mode, args.timeout)

    rows, per_query = evaluate(items, args.base_url, modes, args.top_k, args.timeout)

    summary_path = os.path.join(args.out_dir, "summary.json")
    csv_path = os.path.join(args.out_dir, "per_query.csv")
    svg_path = os.path.join(args.out_dir, "quality_latency_chart.svg")
    md_path = os.path.join(args.out_dir, "report.md")

    with open(summary_path, "w", encoding="utf-8") as handle:
        json.dump({"generated_at": datetime.now(timezone.utc).isoformat(), "rows": rows}, handle, indent=2, ensure_ascii=False)
    write_csv(csv_path, per_query)
    write_svg(svg_path, rows)
    write_markdown(md_path, rows, args)

    print(f"summary: {summary_path}")
    print(f"per_query: {csv_path}")
    print(f"chart: {svg_path}")
    print(f"report: {md_path}")
    for row in rows:
        print(
            f"{row['mode']}: ndcg@10={row['ndcg@10']:.4f}, "
            f"avg_ms={row['latency_avg_ms']:.2f}, p95_ms={row['latency_p95_ms']:.2f}"
        )


if __name__ == "__main__":
    main()
