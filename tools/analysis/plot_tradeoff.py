import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os

# 数据来源：docs/performance_report.md
# NDCG@10 (Accuracy)
ndcg_data = {
    'Baseline': 0.7472,
    'LTR': 0.9927,
    'Cross-Encoder': 0.9569,
    'Hybrid': 0.9810
}

# Average Total Latency (ms)
latency_data = {
    'Baseline': 116.80,
    'LTR': 413.73,
    'Cross-Encoder': 557.28,
    'Hybrid': 390.24
}

# 颜色映射 (与 analyze_latency.py 保持一致)
colors = {
    'Baseline': '#e15759',      # Red
    'LTR': '#4e79a7',           # Blue
    'Cross-Encoder': '#f28e2b', # Orange
    'Hybrid': '#59a14f'         # Green
}

# 标记映射
markers = {
    'Baseline': 'o',
    'LTR': '^',
    'Cross-Encoder': 's',
    'Hybrid': 'D'
}

def plot_tradeoff():
    # 确保输出目录存在
    output_dir = 'models'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    plt.figure(figsize=(8, 6))
    
    # 绘制散点
    for name in ndcg_data.keys():
        x = latency_data[name]
        y = ndcg_data[name]
        plt.scatter(x, y, c=colors[name], s=200, label=name, marker=markers[name], edgecolors='k', zorder=10)
        
        # 添加标签
        offset_y = 0.005
        if name == 'Baseline':
            offset_y = -0.015
        plt.text(x, y + offset_y, name, fontsize=11, ha='center', fontweight='bold')

    # 设置坐标轴范围 (留出余量)
    plt.xlim(0, 700)
    plt.ylim(0.6, 1.05)
    
    # 坐标轴标签
    plt.xlabel('Average Latency (ms)  [Lower is Better]', fontsize=12, fontweight='bold')
    plt.ylabel('NDCG@10  [Higher is Better]', fontsize=12, fontweight='bold')
    plt.title('Accuracy vs. Latency Trade-off', fontsize=14, fontweight='bold')
    
    # 绘制 "Sweet Spot" 区域 (左上角)
    # rect = patches.Rectangle((0, 0.95), 450, 0.1, linewidth=1, edgecolor='none', facecolor='green', alpha=0.1)
    # plt.gca().add_patch(rect)
    # plt.text(200, 1.02, 'Sweet Spot (High Acc & Low Latency)', color='green', fontsize=10, ha='center')

    # 绘制箭头指向更优方向
    plt.arrow(650, 0.65, -100, 0, head_width=0.02, head_length=20, fc='gray', ec='gray', alpha=0.5)
    plt.text(600, 0.68, 'Faster', color='gray', ha='center')
    
    plt.arrow(50, 0.8, 0, 0.1, head_width=15, head_length=0.02, fc='gray', ec='gray', alpha=0.5)
    plt.text(80, 0.85, 'Better', color='gray', va='center', rotation=90)

    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    
    output_path = os.path.join(output_dir, 'tradeoff_plot.png')
    plt.savefig(output_path, dpi=300)
    print(f"Trade-off plot saved to {output_path}")


def main():
    plot_tradeoff()


if __name__ == '__main__':
    main()
