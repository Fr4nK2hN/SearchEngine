import argparse
import csv
import json
import math
import os
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from urllib.parse import urlencode
from urllib.request import urlopen


DEFAULT_MODES = ["baseline", "ltr", "adaptive", "hybrid", "cross_encoder"]

sys.path.insert(
    0,
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
)


def parse_label(value):
    if value is None:
        return None
    value = str(value).strip()
    if value == "":
        return None
    try:
        label = int(float(value))
    except ValueError:
        return None
    return max(0, min(3, label))


def load_judgments(path, label_column, min_labels_per_query):
    judgments = defaultdict(dict)
    rows = []
    with open(path, "r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            query = " ".join(str(row.get("query") or "").split())
            doc_id = str(row.get("doc_id") or "").strip()
            if not query or not doc_id:
                continue
            label = parse_label(row.get(label_column))
            rows.append({**row, "parsed_label": label})
            if label is not None:
                judgments[query][doc_id] = label

    skipped_queries = {
        query: len(label_map)
        for query, label_map in judgments.items()
        if len(label_map) < min_labels_per_query
    }
    filtered = {
        query: label_map
        for query, label_map in judgments.items()
        if len(label_map) >= min_labels_per_query
    }
    return filtered, rows, skipped_queries


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


def mrr_at_k(returned_labels, k=10):
    for idx, rel in enumerate(returned_labels[:k], start=1):
        if rel > 0:
            return 1.0 / idx
    return 0.0


def recall_at_k(returned_labels, missing_labels, k=10):
    all_labels = list(returned_labels) + list(missing_labels)
    total_positive = sum(1 for label in all_labels if label > 0)
    if total_positive <= 0:
        return 0.0
    return sum(1 for label in returned_labels[:k] if label > 0) / total_positive


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


def fetch_search(base_url, query, mode, timeout, client=None, rerank_top_n=None):
    params = {"q": query, "mode": mode, "session_id": f"manual-eval-{int(time.time() * 1000)}"}
    if rerank_top_n is not None:
        params["rerank_top_n"] = str(rerank_top_n)
    t0 = time.perf_counter()
    if client is not None:
        response = client.get("/search", query_string=params)
        payload = response.get_json()
        if response.status_code >= 400:
            raise RuntimeError(f"local /search failed: {response.status_code} {payload}")
    else:
        url = base_url.rstrip("/") + "/search?" + urlencode(params)
        with urlopen(url, timeout=timeout) as response:
            payload = json.loads(response.read().decode("utf-8"))
    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    return payload, elapsed_ms


def evaluate(judgments, base_url, modes, top_k, timeout, client=None, rerank_top_n=None):
    rows = []
    per_query = []
    queries = sorted(judgments.keys())

    for mode in modes:
        ndcgs = []
        mrrs = []
        recalls = []
        latencies = []
        unjudged_counts = []
        route_counts = {}

        for query in queries:
            label_map = judgments[query]
            payload, elapsed_ms = fetch_search(
                base_url,
                query,
                mode,
                timeout,
                client=client,
                rerank_top_n=rerank_top_n,
            )
            hits = payload.get("results", [])[:top_k]
            returned_ids = [str(hit.get("_id")) for hit in hits if hit.get("_id") is not None]
            returned_set = set(returned_ids)
            returned_labels = [int(label_map.get(doc_id, 0)) for doc_id in returned_ids]
            missing_labels = [
                int(label)
                for doc_id, label in label_map.items()
                if int(label) > 0 and doc_id not in returned_set
            ]
            unjudged = sum(1 for doc_id in returned_ids if doc_id not in label_map)

            ndcg = ndcg_at_k(returned_labels, missing_labels, top_k)
            mrr = mrr_at_k(returned_labels, top_k)
            recall = recall_at_k(returned_labels, missing_labels, top_k)
            ndcgs.append(ndcg)
            mrrs.append(mrr)
            recalls.append(recall)
            latencies.append(elapsed_ms)
            unjudged_counts.append(unjudged)

            routing = payload.get("routing") or {}
            route_key = routing.get("selected_mode") or routing.get("route_label") or mode
            route_counts[route_key] = route_counts.get(route_key, 0) + 1

            per_query.append(
                {
                    "mode": mode,
                    "query": query,
                    "ndcg@10": ndcg,
                    "mrr@10": mrr,
                    "recall@10": recall,
                    "latency_ms": elapsed_ms,
                    "unjudged_returned": unjudged,
                    "route": route_key,
                }
            )

        rows.append(
            {
                "mode": mode,
                "query_count": len(queries),
                "ndcg@10": sum(ndcgs) / len(ndcgs) if ndcgs else 0.0,
                "mrr@10": sum(mrrs) / len(mrrs) if mrrs else 0.0,
                "recall@10": sum(recalls) / len(recalls) if recalls else 0.0,
                "latency_avg_ms": sum(latencies) / len(latencies) if latencies else 0.0,
                "latency_p50_ms": percentile(latencies, 50),
                "latency_p95_ms": percentile(latencies, 95),
                "unjudged_returned_avg": sum(unjudged_counts) / len(unjudged_counts) if unjudged_counts else 0.0,
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
    max_latency = max(max_latency, 1.0)
    colors = {
        "baseline": "#0b6e4f",
        "ltr": "#7c3aed",
        "adaptive": "#2563eb",
        "hybrid": "#d97706",
        "cross_encoder": "#e11d48",
    }
    bar_w = 110
    step = chart_w / max(1, len(rows))

    def group(metric, y0, max_v, label, value_fmt):
        out = [f'<text x="{left}" y="{y0 - 38}" class="section">{label}</text>']
        for frac in [0.0, 0.25, 0.5, 0.75, 1.0]:
            y = y0 + chart_h - chart_h * frac
            out.append(f'<line x1="{left}" y1="{y:.1f}" x2="{left + chart_w}" y2="{y:.1f}" class="grid"/>')
            out.append(f'<text x="{left - 16}" y="{y + 5:.1f}" text-anchor="end" class="axis">{max_v * frac:.2f}</text>')
        for idx, row in enumerate(rows):
            x = left + step * idx + (step - bar_w) / 2
            value = float(row[metric])
            h = chart_h * (value / max_v) if max_v > 0 else 0.0
            y = y0 + chart_h - h
            color = colors.get(row["mode"], "#475569")
            out.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_w}" height="{h:.1f}" rx="10" fill="{color}"/>')
            out.append(f'<text x="{x + bar_w / 2:.1f}" y="{y - 10:.1f}" text-anchor="middle" class="value">{value_fmt(value)}</text>')
            out.append(f'<text x="{x + bar_w / 2:.1f}" y="{y0 + chart_h + 30}" text-anchor="middle" class="axis">{row["mode"].replace("_", "-")}</text>')
        return "".join(out)

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
<text x="{width / 2}" y="44" text-anchor="middle" class="title">Manual Judgment Evaluation</text>
<text x="{width / 2}" y="70" text-anchor="middle" class="subtitle">NDCG@10 and average response time based on pooled human labels</text>
{group("ndcg@10", top, 1.0, "NDCG@10", lambda v: f"{v:.4f}")}
{group("latency_avg_ms", top + gap, max_latency * 1.1, "Average Response Time (ms)", lambda v: f"{v:.1f}")}
</svg>'''
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(svg)


def write_report(path, rows, args, label_stats):
    lines = [
        "# Manual Judgment Evaluation Report",
        "",
        f"- Generated at: `{datetime.now(timezone.utc).isoformat()}`",
        f"- Judgments: `{args.judgments}`",
        f"- Label column: `{args.label_column}`",
        f"- Base URL: `{args.base_url}`",
        f"- Top-K: `{args.top_k}`",
        f"- Labeled rows: `{label_stats['labeled_rows']}`",
        f"- Unlabeled rows ignored in ground truth: `{label_stats['unlabeled_rows']}`",
        f"- Queries with labels: `{label_stats['query_count']}`",
        f"- Skipped incomplete queries: `{label_stats['skipped_query_count']}`",
        "",
        "| Mode | NDCG@10 | MRR@10 | Recall@10 | Avg Response (ms) | P95 (ms) | Avg Unjudged Returned |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            f"| `{row['mode']}` | {row['ndcg@10']:.4f} | {row['mrr@10']:.4f} | {row['recall@10']:.4f} | "
            f"{row['latency_avg_ms']:.2f} | {row['latency_p95_ms']:.2f} | {row['unjudged_returned_avg']:.2f} |"
        )
    lines.extend(
        [
            "",
            "## Notes",
            "",
            "- Labels should be manually reviewed; weak historical labels should not be treated as final ground truth.",
            "- `Avg Unjudged Returned` should be close to 0 if the pooled judgment set covers all evaluated modes well.",
            "- Returned documents outside the judgment pool are treated as 0 relevance, so high unjudged coverage still biases the result downward.",
        ]
    )
    with open(path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--judgments", default="reports/manual_relevance_pool_20260411/judgment_pool.csv")
    parser.add_argument("--label-column", default="manual_label")
    parser.add_argument("--base-url", default="http://127.0.0.1:5000")
    parser.add_argument("--modes", default=",".join(DEFAULT_MODES))
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--min-labels-per-query", type=int, default=10)
    parser.add_argument("--timeout", type=float, default=60.0)
    parser.add_argument("--rerank-top-n", type=int, default=0)
    parser.add_argument("--use-flask-test-client", action="store_true")
    parser.add_argument("--out-dir", default="reports/manual_relevance_pool_20260411/eval_after_labeling")
    args = parser.parse_args()

    modes = [x.strip() for x in args.modes.split(",") if x.strip()]
    judgments, judgment_rows, skipped_queries = load_judgments(
        args.judgments,
        args.label_column,
        args.min_labels_per_query,
    )
    if not judgments:
        raise RuntimeError(
            f"no usable labels found in {args.judgments}; fill `{args.label_column}` first "
            f"and ensure each evaluated query has at least {args.min_labels_per_query} labels"
        )

    os.makedirs(args.out_dir, exist_ok=True)
    client = None
    if args.use_flask_test_client:
        from app import app as flask_app

        client = flask_app.test_client()

    rows, per_query = evaluate(
        judgments,
        args.base_url,
        modes,
        args.top_k,
        args.timeout,
        client=client,
        rerank_top_n=args.rerank_top_n if args.rerank_top_n > 0 else None,
    )
    label_stats = {
        "labeled_rows": sum(1 for row in judgment_rows if row["parsed_label"] is not None),
        "unlabeled_rows": sum(1 for row in judgment_rows if row["parsed_label"] is None),
        "query_count": len(judgments),
        "skipped_query_count": len(skipped_queries),
        "skipped_queries": skipped_queries,
    }

    summary_path = os.path.join(args.out_dir, "summary.json")
    per_query_path = os.path.join(args.out_dir, "per_query.csv")
    chart_path = os.path.join(args.out_dir, "manual_ndcg_latency_chart.svg")
    report_path = os.path.join(args.out_dir, "report.md")

    with open(summary_path, "w", encoding="utf-8") as handle:
        json.dump(
            {
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "label_stats": label_stats,
                "rows": rows,
            },
            handle,
            indent=2,
            ensure_ascii=False,
        )
    write_csv(per_query_path, per_query)
    write_svg(chart_path, rows)
    write_report(report_path, rows, args, label_stats)

    print(f"summary: {summary_path}")
    print(f"per_query: {per_query_path}")
    print(f"chart: {chart_path}")
    print(f"report: {report_path}")
    for row in rows:
        print(
            f"{row['mode']}: ndcg@10={row['ndcg@10']:.4f}, "
            f"mrr@10={row['mrr@10']:.4f}, recall@10={row['recall@10']:.4f}, "
            f"avg_ms={row['latency_avg_ms']:.2f}, unjudged_avg={row['unjudged_returned_avg']:.2f}"
        )


if __name__ == "__main__":
    main()
