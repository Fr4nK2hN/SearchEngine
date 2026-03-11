import os
import json
import math
import argparse
import matplotlib.pyplot as plt

def parse_line(line):
    try:
        return json.loads(line)
    except Exception:
        s = line.strip()
        i = s.find('{')
        if i >= 0:
            try:
                return json.loads(s[i:])
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

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', default='logs/events.log')
    parser.add_argument('--last', type=int, default=0, help='仅统计最后N行')
    parser.add_argument('--plot_dir', default='', help='输出图表目录（为空则不生成图）')
    args = parser.parse_args()
    path = args.path
    if not os.path.exists(path):
        print('日志文件不存在: logs/events.log')
        return
    groups = {}
    with open(path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    if args.last and args.last > 0:
        lines = lines[-args.last:]
    for line in lines:
        record = parse_line(line)
        if not isinstance(record, dict):
            continue
        if record.get('message') != 'Search successful' and not record.get('rankingMethod'):
            continue
        key = record.get('rankingMethod', 'Unknown')
        g = groups.setdefault(key, {'total': [], 'retrieval': [], 'feature': [], 'inference': []})
        t = record.get('total_ms')
        r = record.get('retrieval_ms')
        fms = record.get('feature_ms')
        ims = record.get('inference_ms')
        if isinstance(t, (int, float)):
            g['total'].append(float(t))
        if isinstance(r, (int, float)):
            g['retrieval'].append(float(r))
        if isinstance(fms, (int, float)):
            g['feature'].append(float(fms))
        if isinstance(ims, (int, float)):
            g['inference'].append(float(ims))
    print('| 模型策略 | 平均检索 (ms) | 特征 (ms) | 推理 (ms) | 平均总耗时 (ms) | P50 (ms) | P90 (ms) | P95 (ms) | P99 (ms) |')
    print('| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |')
    names = []
    avg_ret_list = []
    avg_fe_list = []
    avg_inf_list = []
    avg_tot_list = []
    p50_list = []
    p90_list = []
    p95_list = []
    p99_list = []
    for name, g in groups.items():
        avg_ret = sum(g['retrieval']) / len(g['retrieval']) if g['retrieval'] else 0.0
        avg_fe = sum(g['feature']) / len(g['feature']) if g['feature'] else 0.0
        avg_inf = sum(g['inference']) / len(g['inference']) if g['inference'] else 0.0
        avg_tot = sum(g['total']) / len(g['total']) if g['total'] else 0.0
        p50 = percentile(g['total'], 50)
        p90 = percentile(g['total'], 90)
        p95 = percentile(g['total'], 95)
        p99_val = p99(g['total'])
        print(f'| {name} | {avg_ret:.2f} | {avg_fe:.2f} | {avg_inf:.2f} | {avg_tot:.2f} | {p50:.2f} | {p90:.2f} | {p95:.2f} | {p99_val:.2f} |')
        names.append(name)
        avg_ret_list.append(avg_ret)
        avg_fe_list.append(avg_fe)
        avg_inf_list.append(avg_inf)
        avg_tot_list.append(avg_tot)
        p50_list.append(p50)
        p90_list.append(p90)
        p95_list.append(p95)
        p99_list.append(p99_val)

    if args.plot_dir:
        os.makedirs(args.plot_dir, exist_ok=True)
        # 1) 平均总耗时柱状图
        plt.figure(figsize=(8, 4))
        x = range(len(names))
        plt.bar(x, avg_tot_list, color=['#4e79a7', '#f28e2b', '#59a14f', '#e15759'][:len(names)])
        plt.xticks(x, names, rotation=20, ha='right')
        plt.ylabel('Average Total Latency (ms)')
        plt.title('Average Total Latency by Strategy')
        plt.tight_layout()
        path_avg = os.path.join(args.plot_dir, 'latency_avg.png')
        plt.savefig(path_avg, dpi=200)
        plt.close()

        # 2) 分段耗时堆叠图（平均）
        plt.figure(figsize=(8, 4))
        x = range(len(names))
        plt.bar(x, avg_ret_list, label='Retrieval', color='#4e79a7')
        plt.bar(x, avg_fe_list, bottom=avg_ret_list, label='Feature', color='#f28e2b')
        bottom_inf = [r + f for r, f in zip(avg_ret_list, avg_fe_list)]
        plt.bar(x, avg_inf_list, bottom=bottom_inf, label='Inference', color='#59a14f')
        plt.xticks(x, names, rotation=20, ha='right')
        plt.ylabel('Average Stage Latency (ms)')
        plt.title('Average Latency by Stage (Stacked)')
        plt.legend()
        plt.tight_layout()
        path_stack = os.path.join(args.plot_dir, 'latency_stacked.png')
        plt.savefig(path_stack, dpi=200)
        plt.close()

        # 3) 分位数柱状图（P50/P90/P95/P99）
        plt.figure(figsize=(10, 5))
        width = 0.18
        x = range(len(names))
        plt.bar([i - 1.5*width for i in x], p50_list, width=width, label='P50', color='#76b7b2')
        plt.bar([i - 0.5*width for i in x], p90_list, width=width, label='P90', color='#59a14f')
        plt.bar([i + 0.5*width for i in x], p95_list, width=width, label='P95', color='#edc948')
        plt.bar([i + 1.5*width for i in x], p99_list, width=width, label='P99', color='#e15759')
        plt.xticks(x, names, rotation=20, ha='right')
        plt.ylabel('Total Latency Percentiles (ms)')
        plt.title('Latency Percentiles by Strategy')
        plt.legend()
        plt.tight_layout()
        path_pct = os.path.join(args.plot_dir, 'latency_percentiles.png')
        plt.savefig(path_pct, dpi=200)
        plt.close()

if __name__ == '__main__':
    main()
