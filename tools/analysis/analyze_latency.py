import argparse
import json
import math
import os

import matplotlib.pyplot as plt


EMBEDDING_METRIC_NAMES = (
    "candidate_docs",
    "precomputed_content_emb_count",
    "precomputed_title_emb_count",
    "encoded_content_emb_count",
    "encoded_title_emb_count",
)


def parse_line(line):
    try:
        return json.loads(line)
    except Exception:
        stripped = line.strip()
        start = stripped.find("{")
        if start >= 0:
            try:
                return json.loads(stripped[start:])
            except Exception:
                return None
        return None


def p99(values):
    if not values:
        return 0.0
    arr = sorted(values)
    idx = max(0, int(math.ceil(0.99 * len(arr)) - 1))
    return float(arr[idx])


def percentile(values, q):
    if not values:
        return 0.0
    arr = sorted(values)
    idx = max(0, int(math.ceil((q / 100.0) * len(arr)) - 1))
    return float(arr[idx])


def ratio_text(numerator, denominator):
    if denominator <= 0:
        return "N/A"
    return f"{(numerator / denominator) * 100.0:.2f}%"


def build_group():
    group = {
        "total": [],
        "retrieval": [],
        "feature": [],
        "inference": [],
    }
    for metric_name in EMBEDDING_METRIC_NAMES:
        group[metric_name] = []
    return group


def collect_groups(lines):
    groups = {}
    for line in lines:
        record = parse_line(line)
        if not isinstance(record, dict):
            continue
        if record.get("message") != "Search successful" and not record.get("rankingMethod"):
            continue
        key = record.get("rankingMethod", "Unknown")
        group = groups.setdefault(key, build_group())
        for field_name, bucket in (
            ("total_ms", "total"),
            ("retrieval_ms", "retrieval"),
            ("feature_ms", "feature"),
            ("inference_ms", "inference"),
        ):
            value = record.get(field_name)
            if isinstance(value, (int, float)):
                group[bucket].append(float(value))
        for metric_name in EMBEDDING_METRIC_NAMES:
            value = record.get(metric_name)
            if isinstance(value, (int, float)):
                group[metric_name].append(int(value))
    return groups


def summarize_embedding_metrics(group):
    candidate_total = sum(group["candidate_docs"])
    content_total = sum(group["precomputed_content_emb_count"]) + sum(group["encoded_content_emb_count"])
    title_total = sum(group["precomputed_title_emb_count"]) + sum(group["encoded_title_emb_count"])
    return {
        "avg_candidate_docs": (
            candidate_total / len(group["candidate_docs"]) if group["candidate_docs"] else 0.0
        ),
        "content_hit_ratio": ratio_text(sum(group["precomputed_content_emb_count"]), content_total),
        "title_hit_ratio": ratio_text(sum(group["precomputed_title_emb_count"]), title_total),
        "avg_online_content_fill": (
            sum(group["encoded_content_emb_count"]) / len(group["encoded_content_emb_count"])
            if group["encoded_content_emb_count"]
            else 0.0
        ),
        "avg_online_title_fill": (
            sum(group["encoded_title_emb_count"]) / len(group["encoded_title_emb_count"])
            if group["encoded_title_emb_count"]
            else 0.0
        ),
        "has_embedding_metrics": bool(group["candidate_docs"]),
    }


def print_latency_table(groups):
    print("| 模型策略 | 平均检索 (ms) | 特征 (ms) | 推理 (ms) | 平均总耗时 (ms) | P50 (ms) | P90 (ms) | P95 (ms) | P99 (ms) |")
    print("| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |")
    rows = []
    for name, group in groups.items():
        avg_ret = sum(group["retrieval"]) / len(group["retrieval"]) if group["retrieval"] else 0.0
        avg_fe = sum(group["feature"]) / len(group["feature"]) if group["feature"] else 0.0
        avg_inf = sum(group["inference"]) / len(group["inference"]) if group["inference"] else 0.0
        avg_tot = sum(group["total"]) / len(group["total"]) if group["total"] else 0.0
        p50 = percentile(group["total"], 50)
        p90 = percentile(group["total"], 90)
        p95 = percentile(group["total"], 95)
        p99_val = p99(group["total"])
        print(
            f"| {name} | {avg_ret:.2f} | {avg_fe:.2f} | {avg_inf:.2f} | "
            f"{avg_tot:.2f} | {p50:.2f} | {p90:.2f} | {p95:.2f} | {p99_val:.2f} |"
        )
        rows.append((name, avg_ret, avg_fe, avg_inf, avg_tot, p50, p90, p95, p99_val))
    return rows


def print_embedding_table(groups):
    summaries = [
        (name, summarize_embedding_metrics(group))
        for name, group in groups.items()
    ]
    summaries = [item for item in summaries if item[1]["has_embedding_metrics"]]
    if not summaries:
        return
    print()
    print("| 模型策略 | 平均候选文档数 | content_emb 命中率 | title_emb 命中率 | 平均在线补齐 content | 平均在线补齐 title |")
    print("| :--- | :--- | :--- | :--- | :--- | :--- |")
    for name, summary in summaries:
        print(
            f"| {name} | {summary['avg_candidate_docs']:.2f} | "
            f"{summary['content_hit_ratio']} | {summary['title_hit_ratio']} | "
            f"{summary['avg_online_content_fill']:.2f} | {summary['avg_online_title_fill']:.2f} |"
        )


def plot_latency_charts(plot_dir, names, avg_ret_list, avg_fe_list, avg_inf_list, avg_tot_list, p50_list, p90_list, p95_list, p99_list):
    os.makedirs(plot_dir, exist_ok=True)

    plt.figure(figsize=(8, 4))
    x = range(len(names))
    plt.bar(x, avg_tot_list, color=["#4e79a7", "#f28e2b", "#59a14f", "#e15759"][: len(names)])
    plt.xticks(x, names, rotation=20, ha="right")
    plt.ylabel("Average Total Latency (ms)")
    plt.title("Average Total Latency by Strategy")
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, "latency_avg.png"), dpi=200)
    plt.close()

    plt.figure(figsize=(8, 4))
    plt.bar(x, avg_ret_list, label="Retrieval", color="#4e79a7")
    plt.bar(x, avg_fe_list, bottom=avg_ret_list, label="Feature", color="#f28e2b")
    bottom_inf = [ret + feat for ret, feat in zip(avg_ret_list, avg_fe_list)]
    plt.bar(x, avg_inf_list, bottom=bottom_inf, label="Inference", color="#59a14f")
    plt.xticks(x, names, rotation=20, ha="right")
    plt.ylabel("Average Stage Latency (ms)")
    plt.title("Average Latency by Stage (Stacked)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, "latency_stacked.png"), dpi=200)
    plt.close()

    plt.figure(figsize=(10, 5))
    width = 0.18
    plt.bar([i - 1.5 * width for i in x], p50_list, width=width, label="P50", color="#76b7b2")
    plt.bar([i - 0.5 * width for i in x], p90_list, width=width, label="P90", color="#59a14f")
    plt.bar([i + 0.5 * width for i in x], p95_list, width=width, label="P95", color="#edc948")
    plt.bar([i + 1.5 * width for i in x], p99_list, width=width, label="P99", color="#e15759")
    plt.xticks(x, names, rotation=20, ha="right")
    plt.ylabel("Total Latency Percentiles (ms)")
    plt.title("Latency Percentiles by Strategy")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, "latency_percentiles.png"), dpi=200)
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", default="logs/events.log")
    parser.add_argument("--last", type=int, default=0, help="仅统计最后N行")
    parser.add_argument("--plot_dir", default="", help="输出图表目录（为空则不生成图）")
    args = parser.parse_args()
    path = args.path
    if not os.path.exists(path):
        print(f"日志文件不存在: {path}")
        return

    with open(path, "r", encoding="utf-8") as handle:
        lines = handle.readlines()
    if args.last and args.last > 0:
        lines = lines[-args.last:]

    groups = collect_groups(lines)
    rows = print_latency_table(groups)
    print_embedding_table(groups)

    if args.plot_dir and rows:
        names = [row[0] for row in rows]
        avg_ret_list = [row[1] for row in rows]
        avg_fe_list = [row[2] for row in rows]
        avg_inf_list = [row[3] for row in rows]
        avg_tot_list = [row[4] for row in rows]
        p50_list = [row[5] for row in rows]
        p90_list = [row[6] for row in rows]
        p95_list = [row[7] for row in rows]
        p99_list = [row[8] for row in rows]
        plot_latency_charts(
            args.plot_dir,
            names,
            avg_ret_list,
            avg_fe_list,
            avg_inf_list,
            avg_tot_list,
            p50_list,
            p90_list,
            p95_list,
            p99_list,
        )


if __name__ == "__main__":
    main()
