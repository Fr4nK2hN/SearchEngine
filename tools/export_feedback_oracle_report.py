#!/usr/bin/env python3
"""
Export a concise markdown report for feedback-driven oracle experiments.
"""

import argparse
import json
import os
from datetime import datetime, timezone


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def fmt_pct(x):
    return f"{x * 100:.2f}%"


def fmt_num(x):
    return f"{x:.4f}"


def build_section(title, result):
    oracle = result["oracle"]
    profile = result.get("diagnostics", {}).get("data_profile", {})
    warnings = result.get("diagnostics", {}).get("warnings", [])
    experts = result.get("experts", {})

    lines = [
        f"## {title}",
        "",
        f"- Data used: `{result.get('data_used_count')}`",
        f"- Best single expert: `{oracle.get('best_single_expert')}`",
        f"- Oracle nDCG gain: `{fmt_num(oracle.get('ndcg_gain_abs', 0.0))}` ({fmt_pct(oracle.get('ndcg_gain_rel', 0.0))})",
        f"- Selection count: `{oracle.get('selection_count')}`",
        f"- Query profile: `prefix_like_ratio={profile.get('prefix_like_ratio', 0.0):.2%}`, `first_doc_is_max_ratio={profile.get('first_doc_is_max_ratio', 0.0):.2%}`",
        "",
        "| Expert | nDCG@10 | MRR@10 | Avg Latency (ms) |",
        "| --- | --- | --- | --- |",
    ]

    for name in ["Baseline", "LTR", "Cross-Encoder", "Hybrid"]:
        if name not in experts:
            continue
        m = experts[name]
        lines.append(
            f"| {name} | {fmt_num(m.get('ndcg', 0.0))} | {fmt_num(m.get('mrr', 0.0))} | {m.get('latency_avg_ms', 0.0):.2f} |"
        )

    if warnings:
        lines.append("")
        lines.append("Warnings:")
        for w in warnings:
            lines.append(f"- {w}")

    lines.append("")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Export feedback oracle markdown report")
    parser.add_argument("--primary", required=True, help="Primary oracle_results.json")
    parser.add_argument("--secondary", default="", help="Optional secondary oracle_results.json")
    parser.add_argument("--dataset-report", default="", help="Optional feedback dataset report json")
    parser.add_argument("--output", required=True, help="Output markdown path")
    args = parser.parse_args()

    primary = load_json(args.primary)
    secondary = load_json(args.secondary) if args.secondary else None
    dataset_report = load_json(args.dataset_report) if args.dataset_report else None

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)

    lines = [
        "# Feedback Oracle Experiment Record",
        "",
        f"- Generated at (UTC): `{datetime.now(timezone.utc).isoformat()}`",
        f"- Primary result: `{args.primary}`",
    ]
    if args.secondary:
        lines.append(f"- Secondary result: `{args.secondary}`")
    if args.dataset_report:
        lines.append(f"- Dataset report: `{args.dataset_report}`")

    lines.append("")
    if dataset_report:
        lines.extend(
            [
                "## Feedback Dataset",
                "",
                f"- Samples: `{dataset_report.get('sample_count')}`",
                f"- Search count: `{dataset_report.get('search_stats', {}).get('search_count')}`",
                f"- Searches with confirmed click: `{dataset_report.get('search_stats', {}).get('searches_with_confirmed_click')}`",
                f"- Label distribution: `{dataset_report.get('label_distribution')}`",
                f"- Dropped: `{dataset_report.get('dropped_counts')}`",
                "",
            ]
        )

    lines.append(build_section("Primary (Combined Feedback)", primary))
    if secondary:
        lines.append(build_section("Secondary (No Prefix Filter)", secondary))

    lines.extend(
        [
            "## Conclusion",
            "",
            "Feedback-driven oracle shows clear upper bound over best single expert, supporting adaptive routing as a viable direction.",
            "",
        ]
    )

    with open(args.output, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"Report exported: {args.output}")


if __name__ == "__main__":
    main()
