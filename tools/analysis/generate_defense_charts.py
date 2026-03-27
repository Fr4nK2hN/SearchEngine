#!/usr/bin/env python3
"""
Generate polished defense/demo charts from existing project reports.

Data sources:
- reports/performance_report_20260316_adaptive_balanced.md
- reports/adaptive_tradeoff_scans_20260316_live/adaptive_tradeoff_scans.json

Outputs:
- PNG charts under reports/defense_charts_YYYYMMDD/
- README.md aggregating the generated figures
"""

from __future__ import annotations

import argparse
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "axes.titlesize": 17,
        "axes.titleweight": "bold",
        "axes.labelsize": 11.5,
        "axes.edgecolor": "#C9C4B8",
        "axes.linewidth": 1.0,
        "axes.facecolor": "#FFFDF8",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.facecolor": "#F6F1E8",
        "savefig.facecolor": "#F6F1E8",
        "grid.color": "#DED7C8",
        "grid.alpha": 0.45,
        "grid.linestyle": "--",
        "legend.frameon": False,
        "legend.fontsize": 10,
        "xtick.color": "#3A342B",
        "ytick.color": "#3A342B",
        "axes.labelcolor": "#3A342B",
        "text.color": "#2B251C",
    }
)


PALETTE = {
    "baseline": "#C05A4A",
    "old_adaptive": "#867A6A",
    "balanced": "#0F766E",
    "quality": "#D97706",
    "low_latency": "#6B7280",
    "ltr": "#2563EB",
    "cross_encoder": "#B45309",
    "feature": "#7C3AED",
    "inference": "#CA8A04",
    "retrieval": "#334155",
    "other": "#9CA3AF",
}


def clean_text(value: str) -> str:
    return (
        value.replace("`", "")
        .replace("**", "")
        .replace("（", "(")
        .replace("）", ")")
        .strip()
    )


def to_number(value: str) -> float:
    text = clean_text(str(value)).replace(",", "").replace("%", "")
    text = text.strip()
    if not text or text == "-":
        return float("nan")
    if "/" in text:
        left, _, _right = text.partition("/")
        return float(left.strip())
    return float(text)


def parse_markdown_table_block(lines: list[str]) -> list[dict[str, str]]:
    headers = [clean_text(cell) for cell in lines[0].strip().strip("|").split("|")]
    rows: list[dict[str, str]] = []
    for line in lines[2:]:
        cells = [clean_text(cell) for cell in line.strip().strip("|").split("|")]
        if len(cells) != len(headers):
            continue
        rows.append(dict(zip(headers, cells)))
    return rows


def extract_tables(markdown_text: str) -> list[list[dict[str, str]]]:
    lines = markdown_text.splitlines()
    tables: list[list[dict[str, str]]] = []
    idx = 0
    while idx < len(lines):
        if (
            lines[idx].startswith("|")
            and idx + 1 < len(lines)
            and re.match(r"^\|\s*-", lines[idx + 1])
        ):
            block = [lines[idx], lines[idx + 1]]
            idx += 2
            while idx < len(lines) and lines[idx].startswith("|"):
                block.append(lines[idx])
                idx += 1
            tables.append(parse_markdown_table_block(block))
            continue
        idx += 1
    return tables


@dataclass
class ReportData:
    coverage_rows: list[dict[str, str]]
    offline_rows: list[dict[str, str]]
    online_rows: list[dict[str, str]]
    stage_rows: list[dict[str, str]]
    route_rows: list[dict[str, str]]
    tradeoff_scan: dict


def load_report_data(perf_report: Path, tradeoff_json: Path) -> ReportData:
    tables = extract_tables(perf_report.read_text(encoding="utf-8"))
    coverage_rows: list[dict[str, str]] = []
    offline_rows: list[dict[str, str]] = []
    online_rows: list[dict[str, str]] = []
    stage_rows: list[dict[str, str]] = []
    route_rows: list[dict[str, str]] = []

    for table in tables:
        if not table:
            continue
        headers = tuple(table[0].keys())
        first_label = next(iter(table[0].values()))
        if headers == ("指标", "数值") and first_label == "Total docs":
            coverage_rows = table
        elif headers == ("配置", "threshold", "CE top-k", "nDCG@10", "MRR@10", "Recall@10", "Avg Latency (ms)", "P95 (ms)"):
            offline_rows = table
        elif headers == ("模式", "Server Avg (ms)", "Server P50", "Server P95", "Server P99", "Client Avg (ms)", "Client P50", "Client P95", "Client P99"):
            online_rows = table
        elif headers == ("指标", "数值") and first_label == "Retrieval Avg (ms)":
            stage_rows = table
        elif headers == ("路由", "查询数", "占比"):
            route_rows = table

    tradeoff_scan = json.loads(tradeoff_json.read_text(encoding="utf-8"))
    return ReportData(
        coverage_rows=coverage_rows,
        offline_rows=offline_rows,
        online_rows=online_rows,
        stage_rows=stage_rows,
        route_rows=route_rows,
        tradeoff_scan=tradeoff_scan,
    )


def figure_path(out_dir: Path, index: int, slug: str) -> Path:
    return out_dir / f"{index:02d}_{slug}.png"


def add_figure_title(fig, title: str, subtitle: str | None = None):
    fig.text(
        0.08,
        0.982,
        title,
        ha="left",
        va="top",
        fontsize=20,
        fontweight="bold",
        color="#2B251C",
    )
    if subtitle:
        fig.text(
            0.08,
            0.936,
            subtitle,
            ha="left",
            va="top",
            fontsize=10.5,
            color="#6B6256",
        )


def add_panel_title(ax, title: str, subtitle: str | None = None):
    ax.set_title(title, loc="left", pad=28, fontsize=13.5)
    if subtitle:
        ax.text(
            0.0,
            1.035,
            subtitle,
            transform=ax.transAxes,
            fontsize=9.5,
            color="#6B6256",
            ha="left",
            va="bottom",
        )


def finalize_figure(fig, *, reserve_header=False):
    top = 0.88 if reserve_header else 0.96
    fig.tight_layout(rect=(0.0, 0.0, 1.0, top))


def save_quality_latency_chart(data: ReportData, out_path: Path):
    label_to_key = {
        "Baseline": "baseline",
        "旧默认 Adaptive": "old_adaptive",
        "新默认 Adaptive(平衡点)": "balanced",
        "质量优先 Adaptive": "quality",
        "极低延迟 Adaptive": "low_latency",
        "新默认 Adaptive（平衡点）": "balanced",
    }
    label_to_display = {
        "Baseline": "Baseline",
        "旧默认 Adaptive": "Old adaptive",
        "新默认 Adaptive(平衡点)": "Balanced default",
        "新默认 Adaptive（平衡点）": "Balanced default",
        "质量优先 Adaptive": "Quality-first adaptive",
        "极低延迟 Adaptive": "Low-latency adaptive",
    }
    rows = []
    for row in data.offline_rows:
        label = clean_text(row["配置"])
        rows.append(
            {
                "label": label,
                "display_label": label_to_display.get(label, label),
                "latency": to_number(row["Avg Latency (ms)"]),
                "ndcg": to_number(row["nDCG@10"]),
                "recall": to_number(row["Recall@10"]),
                "color": PALETTE[label_to_key.get(label, "low_latency")],
            }
        )

    fig, ax = plt.subplots(figsize=(10.5, 6.7))
    add_figure_title(
        fig,
        "Offline Quality vs. Latency",
        "Candidate configs from the balanced-adaptive evaluation report",
    )
    ax.grid(True, axis="both")

    for item in rows:
        size = 520 + item["recall"] * 2400
        edge = "#1F2937" if "新默认" in item["label"] else "#F8F5EC"
        linewidth = 2.6 if "新默认" in item["label"] else 1.4
        ax.scatter(
            item["latency"],
            item["ndcg"],
            s=size,
            color=item["color"],
            alpha=0.92,
            edgecolors=edge,
            linewidths=linewidth,
            zorder=3,
        )
        dx = 1.3 if item["latency"] < 40 else 1.8
        dy = 0.0015 if "Baseline" not in item["label"] else -0.003
        ax.text(
            item["latency"] + dx,
            item["ndcg"] + dy,
            item["display_label"],
            fontsize=10,
            ha="left",
            va="center",
        )

    ax.annotate(
        "Balanced default\nbest overall trade-off",
        xy=(27.7, 0.1903),
        xytext=(44, 0.186),
        arrowprops={"arrowstyle": "->", "color": "#0F766E", "lw": 1.8},
        bbox={"boxstyle": "round,pad=0.4", "facecolor": "#E7F8F5", "edgecolor": "#0F766E"},
        fontsize=10,
    )

    ax.set_xlabel("Average latency (ms)")
    ax.set_ylabel("nDCG@10")
    ax.set_xlim(10, 78)
    ax.set_ylim(0.14, 0.20)
    finalize_figure(fig, reserve_header=True)
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def save_online_latency_compare(data: ReportData, out_path: Path):
    baseline = next(row for row in data.online_rows if row["模式"] == "Baseline")
    adaptive = next(row for row in data.online_rows if "Adaptive" in row["模式"])

    categories = ["Server Avg", "Server P95", "Client Avg", "Client P95"]
    baseline_values = [
        to_number(baseline["Server Avg (ms)"]),
        to_number(baseline["Server P95"]),
        to_number(baseline["Client Avg (ms)"]),
        to_number(baseline["Client P95"]),
    ]
    adaptive_values = [
        to_number(adaptive["Server Avg (ms)"]),
        to_number(adaptive["Server P95"]),
        to_number(adaptive["Client Avg (ms)"]),
        to_number(adaptive["Client P95"]),
    ]

    y = np.arange(len(categories))
    fig, ax = plt.subplots(figsize=(10.8, 6.2))
    add_figure_title(
        fig,
        "Baseline vs. Adaptive (Live Requests)",
        "Same 42 queries through the real /search endpoint",
    )
    ax.grid(True, axis="x")

    for idx, cat in enumerate(categories):
        ax.plot(
            [baseline_values[idx], adaptive_values[idx]],
            [idx, idx],
            color="#D6D0C4",
            linewidth=3.0,
            zorder=1,
        )
        ax.scatter(
            baseline_values[idx],
            idx,
            s=170,
            color=PALETTE["baseline"],
            edgecolors="#FFFDF8",
            linewidths=1.2,
            zorder=3,
            label="Baseline" if idx == 0 else None,
        )
        ax.scatter(
            adaptive_values[idx],
            idx,
            s=200,
            color=PALETTE["balanced"],
            edgecolors="#FFFDF8",
            linewidths=1.2,
            zorder=3,
            label="Adaptive (balanced default)" if idx == 0 else None,
        )
        ax.text(baseline_values[idx] - 1.6, idx + 0.18, f"{baseline_values[idx]:.1f}", ha="right", fontsize=9)
        ax.text(adaptive_values[idx] + 1.6, idx - 0.18, f"{adaptive_values[idx]:.1f}", ha="left", fontsize=9)

    ax.set_yticks(y)
    ax.set_yticklabels(categories)
    ax.set_xlabel("Latency (ms)")
    ax.set_xlim(0, 100)
    ax.legend(loc="lower right")
    finalize_figure(fig, reserve_header=True)
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def save_stage_breakdown(data: ReportData, out_path: Path):
    stage = {row["指标"]: to_number(row["数值"]) for row in data.stage_rows}
    baseline = next(row for row in data.online_rows if row["模式"] == "Baseline")
    adaptive = next(row for row in data.online_rows if "Adaptive" in row["模式"])
    baseline_server = to_number(baseline["Server Avg (ms)"])
    adaptive_server = to_number(adaptive["Server Avg (ms)"])
    adaptive_other = max(0.0, adaptive_server - stage["Retrieval Avg (ms)"] - stage["Feature Avg (ms)"] - stage["Inference Avg (ms)"])

    labels = ["Baseline", "Adaptive"]
    retrieval = [baseline_server, stage["Retrieval Avg (ms)"]]
    feature = [0.0, stage["Feature Avg (ms)"]]
    inference = [0.0, stage["Inference Avg (ms)"]]
    other = [0.0, adaptive_other]

    fig, ax = plt.subplots(figsize=(10.4, 5.8))
    add_figure_title(
        fig,
        "Adaptive Server-Time Breakdown",
        "The balanced default pays for quality mainly through reranking",
    )
    ax.grid(True, axis="x")
    y = np.arange(len(labels))
    left = np.zeros(len(labels))
    segments = [
        ("Retrieval", retrieval, PALETTE["retrieval"]),
        ("Feature", feature, PALETTE["feature"]),
        ("Inference", inference, PALETTE["inference"]),
        ("Other", other, PALETTE["other"]),
    ]
    for name, values, color in segments:
        ax.barh(y, values, left=left, height=0.52, color=color, label=name)
        for idx, value in enumerate(values):
            if value >= 2.0:
                ax.text(left[idx] + value / 2.0, idx, f"{value:.1f}", ha="center", va="center", fontsize=9, color="white")
        left += values

    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_xlabel("Average server time (ms)")
    ax.legend(ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.08))
    finalize_figure(fig, reserve_header=True)
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def save_route_donut(data: ReportData, out_path: Path):
    labels = [clean_text(row["路由"]) for row in data.route_rows]
    values = [to_number(row["查询数"]) for row in data.route_rows]
    colors = [
        "#0F766E",
        "#C2410C",
        "#64748B",
        "#B45309",
        "#7C3AED",
    ][: len(labels)]

    fig, ax = plt.subplots(figsize=(8.8, 6.6))
    add_figure_title(
        fig,
        "Adaptive Route Mix",
        "Most queries stay on the faster paths; only 23.8% hit Cross-Encoder",
    )
    wedges, _ = ax.pie(
        values,
        startangle=110,
        colors=colors,
        wedgeprops={"width": 0.36, "edgecolor": "#FFFDF8", "linewidth": 2},
    )
    total = sum(values)
    for wedge, label, value in zip(wedges, labels, values):
        angle = (wedge.theta2 + wedge.theta1) / 2.0
        x = math.cos(math.radians(angle))
        y = math.sin(math.radians(angle))
        ax.text(
            1.17 * x,
            1.17 * y,
            f"{label}\n{int(value)} ({value / total:.1%})",
            ha="center",
            va="center",
            fontsize=9,
        )
    ax.text(0, 0.05, "42\nqueries", ha="center", va="center", fontsize=24, fontweight="bold")
    ax.text(0, -0.17, "live benchmark", ha="center", va="center", fontsize=10, color="#6B6256")
    ax.set_aspect("equal")
    finalize_figure(fig, reserve_header=True)
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def save_tradeoff_scan_panel(data: ReportData, out_path: Path):
    hard_rows = data.tradeoff_scan["hard_rate_scan"]
    ce_rows = data.tradeoff_scan["ce_depth_scan"]

    hard_x = [row["actual_hard_rate"] * 100.0 if "actual_hard_rate" in row else row["hard_rate"] * 100.0 for row in hard_rows]
    hard_ndcg = [row["ndcg@10"] for row in hard_rows]
    hard_latency = [row["latency_ms_avg"] for row in hard_rows]

    ce_x_labels = [row["ce_top_n_label"] for row in ce_rows]
    ce_x = np.arange(len(ce_rows))
    ce_ndcg = [row["ndcg@10"] for row in ce_rows]
    ce_latency = [row["latency_ms_avg"] for row in ce_rows]

    fig, axes = plt.subplots(1, 2, figsize=(14.2, 5.8))
    add_panel_title(
        axes[0],
        "Hard-Rate Sweep",
        "More hard queries improve quality, but increase latency quickly",
    )
    axes[0].grid(True, axis="both")
    ax0b = axes[0].twinx()
    axes[0].plot(hard_x, hard_ndcg, color=PALETTE["balanced"], marker="o", linewidth=2.8, label="nDCG@10")
    ax0b.plot(hard_x, hard_latency, color=PALETTE["quality"], marker="s", linewidth=2.4, label="Avg latency")
    axes[0].set_xlabel("Actual hard-query rate (%)")
    axes[0].set_ylabel("nDCG@10", color=PALETTE["balanced"])
    ax0b.set_ylabel("Latency (ms)", color=PALETTE["quality"])
    axes[0].scatter([31.0], [0.1903], s=160, color="#111827", zorder=5)
    axes[0].annotate(
        "balanced point",
        xy=(31.0, 0.1903),
        xytext=(35.5, 0.181),
        arrowprops={"arrowstyle": "->", "lw": 1.6, "color": "#111827"},
        fontsize=9,
    )

    add_panel_title(
        axes[1],
        "CE Depth Sweep",
        "Top-5 keeps the same quality as the current policy in this scan",
    )
    axes[1].grid(True, axis="y")
    bars = axes[1].bar(ce_x, ce_latency, color="#E7D6BE", edgecolor="#C49A6C", linewidth=1.0)
    axes[1].set_xticks(ce_x)
    axes[1].set_xticklabels(ce_x_labels)
    axes[1].set_xlabel("CE top-N")
    axes[1].set_ylabel("Latency (ms)")
    ax1b = axes[1].twinx()
    ax1b.plot(ce_x, ce_ndcg, color=PALETTE["balanced"], marker="o", linewidth=2.8)
    ax1b.set_ylabel("nDCG@10", color=PALETTE["balanced"])
    for rect, latency in zip(bars, ce_latency):
        axes[1].text(rect.get_x() + rect.get_width() / 2.0, rect.get_height() + 1.6, f"{latency:.1f}", ha="center", fontsize=8.5)

    finalize_figure(fig, reserve_header=False)
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def save_embedding_coverage_chart(data: ReportData, out_path: Path):
    coverage = {row["指标"]: row["数值"] for row in data.coverage_rows}
    metrics = [
        ("Content embeddings", to_number(coverage["content_emb 覆盖率"])),
        ("Title embeddings", to_number(coverage["title_emb 覆盖率"])),
        ("Both fields", to_number(coverage["双字段同时覆盖"])),
    ]

    fig, ax = plt.subplots(figsize=(9.6, 5.6))
    add_figure_title(
        fig,
        "Embedding Coverage",
        "The document side is fully precomputed, so online LTR avoids vector backfill",
    )
    ax.grid(False)
    y = np.arange(len(metrics))
    values = [value for _label, value in metrics]
    labels = [label for label, _value in metrics]
    ax.barh(y, [100] * len(values), color="#E6E1D5", height=0.5)
    ax.barh(y, values, color=PALETTE["balanced"], height=0.5)
    for idx, value in enumerate(values):
        ax.text(value - 1.5, idx, f"{value:.0f}%", ha="right", va="center", color="white", fontweight="bold")
    ax.set_xlim(0, 100)
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.set_xlabel("Coverage (%)")
    ax.text(
        0.99,
        -0.20,
        "Online content fill = 0\nOnline title fill = 0",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=10,
        bbox={"boxstyle": "round,pad=0.35", "facecolor": "#F5F9F8", "edgecolor": "#B5D8D4"},
    )
    finalize_figure(fig, reserve_header=True)
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def build_markdown(out_dir: Path, chart_paths: list[Path], data: ReportData, perf_report: Path, tradeoff_json: Path) -> str:
    return f"""# 搜索引擎项目阶段性汇报（非正式演示版）

这份材料用于向老师汇报当前项目进展。  
图表基于项目中已有的真实报告自动生成，没有手工改数。

数据来源：
- `{perf_report.as_posix()}`
- `{tradeoff_json.as_posix()}`

## 这次汇报的重点

- 当前 adaptive 默认参数已经调整到一个更合理的平衡点。
- 这个配置相对 baseline，排序质量更好，在线延迟也还可以接受。
- 这个配置也优于之前的默认 adaptive：更快，也更准。
- 当前主要的尾延迟来自少量 `hard -> cross_encoder` 查询。
- 文档 embedding 已经 100% 预计算，线上没有额外补算。

## 什么是 Adaptive

在这个项目里，adaptive 可以理解为“按查询难度选择不同排序策略”。

- 如果查询比较简单，就走更轻的排序路径
- 如果查询比较复杂，就走更强的重排序路径
- 这样做的目的，是在效果和性能之间取得更好的平衡

我现在这版 adaptive 的基本思路是：

- easy query：优先走 `baseline` 或 `ltr`
- hard query：走 `cross_encoder`

也就是说，adaptive 不是给所有查询都使用最重的模型，而是只把更高的计算成本留给更难的查询。

如果直接全部使用重模型，效果可能更高，但延迟也会明显变大。  
如果全部使用 baseline，速度会更快，但排序质量会下降。  
adaptive 的意义，就是在这两者之间找到一个更适合实际系统运行的折中点。

## 1. 总结果：离线质量和时延的关系

![Offline trade-off](./{chart_paths[0].name})

- 绿色高亮点就是我现在采用的默认配置：`threshold=0.6062`，`hard top-k cap=5`
- 这个点不是最低延迟，也不是最高质量，而是整体更平衡的位置
- baseline 更快，但排序质量更低

## 2. 真实接口测试：Baseline 和 Adaptive 的在线对比

![Online latency comparison](./{chart_paths[1].name})

- 我现在比较的是同一批 `42` 条 query 走真实 `/search` 接口得到的结果，不是手工估算
- adaptive 的目标不是一定比 baseline 更快，而是在合理时延内换取更好的排序质量

## 3. Adaptive 的时间主要花在哪里

![Stage breakdown](./{chart_paths[2].name})

- 现在最大的额外开销主要不是 embedding 在线计算
- 文档 embedding 已经提前离线算好了
- 当前真正拉高延迟的，是少量 hard query 进入重排序路径后的推理成本

## 4. 为什么 Adaptive 没有慢得像全量 Cross-Encoder

![Route distribution](./{chart_paths[3].name})

- 不是所有 query 都走最重的模型
- 只有一部分 query 会走 `hard -> cross_encoder`
- 大部分 query 还是停留在更轻的路径上

所以 adaptive 的核心思路是：把重计算留给更难的查询。

## 5. 我为什么把默认参数定在这里

![Tradeoff scan panel](./{chart_paths[4].name})

- 左图展示 hard-rate 提高后，质量会上升，但延迟也会上升
- 右图展示 CE top-N 提高后，延迟上升更明显，但质量收益并不总是成比例
- 所以我最后选的是一个更适合演示和实际运行的平衡点
- 这个参数不是凭感觉设定的，而是根据扫描结果选出来的

## 6. 关于预计算，我准备怎么解释

![Embedding coverage](./{chart_paths[5].name})

- 这里预计算的是文档侧静态特征，不是提前知道查询答案
- 这和倒排索引、IDF 预计算、本地缓存本质上是同一类工程优化
- 它的目的，是把可以离线完成的计算提前做掉，从而降低真实查询时延

## 当前总结

- 目前这个搜索引擎原型已经能完整跑通从检索到重排序再到前端展示的主流程
- 我现在做的重点已经从“能不能跑”转到“怎么把效果和性能做平衡”
- 当前 adaptive 默认策略已经比 baseline 更准，同时在线时延仍然可控
- 接下来我还可以继续优化 hard query 路径的推理成本，并补充更完整的展示材料和实验分析
"""


def main():
    parser = argparse.ArgumentParser(description="Generate defense/demo charts from existing reports.")
    parser.add_argument(
        "--perf-report",
        default="reports/performance_report_20260316_adaptive_balanced.md",
        help="Markdown performance report to parse.",
    )
    parser.add_argument(
        "--tradeoff-json",
        default="reports/adaptive_tradeoff_scans_20260316_live/adaptive_tradeoff_scans.json",
        help="Adaptive tradeoff scan JSON to parse.",
    )
    parser.add_argument(
        "--out-dir",
        default="reports/defense_charts_20260324",
        help="Directory where charts and README.md will be written.",
    )
    args = parser.parse_args()

    perf_report = Path(args.perf_report)
    tradeoff_json = Path(args.tradeoff_json)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    data = load_report_data(perf_report, tradeoff_json)
    if not data.offline_rows or not data.online_rows or not data.route_rows:
        raise RuntimeError("Failed to parse required tables from performance report.")

    chart_paths = [
        figure_path(out_dir, 1, "offline_quality_latency_tradeoff"),
        figure_path(out_dir, 2, "live_latency_baseline_vs_adaptive"),
        figure_path(out_dir, 3, "adaptive_stage_breakdown"),
        figure_path(out_dir, 4, "adaptive_route_distribution"),
        figure_path(out_dir, 5, "parameter_scan_panel"),
        figure_path(out_dir, 6, "embedding_coverage"),
    ]

    save_quality_latency_chart(data, chart_paths[0])
    save_online_latency_compare(data, chart_paths[1])
    save_stage_breakdown(data, chart_paths[2])
    save_route_donut(data, chart_paths[3])
    save_tradeoff_scan_panel(data, chart_paths[4])
    save_embedding_coverage_chart(data, chart_paths[5])

    readme_path = out_dir / "README.md"
    readme_path.write_text(
        build_markdown(out_dir, chart_paths, data, perf_report, tradeoff_json),
        encoding="utf-8",
    )

    print(f"Generated {len(chart_paths)} charts in {out_dir}")
    print(f"- markdown: {readme_path}")
    for path in chart_paths:
        print(f"- chart: {path}")


if __name__ == "__main__":
    main()
