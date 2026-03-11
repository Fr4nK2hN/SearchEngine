"""
从预处理后的语料库计算 IDF (Inverse Document Frequency) 字典

用法:
    python tools/build_idf.py
    python tools/build_idf.py --input data/msmarco_100k_processed.json --output models/idf_dict.json

算法:
    IDF(t) = log(N / (df(t) + 1))
    其中 N 为文档总数, df(t) 为包含词 t 的文档数
"""

import os
import sys
import json
import math
import argparse
import re
from collections import Counter

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def tokenize_simple(text):
    """简单分词: 小写化 + 提取字母数字串 + 去停用词"""
    stop_words = {
        'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
        'of', 'with', 'by', 'from', 'as', 'is', 'was', 'are', 'were', 'be',
        'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will',
        'would', 'could', 'should', 'may', 'might', 'can', 'this', 'that',
        'these', 'those', 'it', 'its', 'they', 'their', 'them', 'not', 'no',
        'so', 'if', 'then', 'than', 'too', 'very', 'just',
    }
    tokens = re.findall(r'\b[a-z]{2,}\b', text.lower())
    return [t for t in tokens if t not in stop_words]


def build_idf(input_path, output_path):
    """从语料库文件构建 IDF 字典"""
    print(f"正在读取语料库: {input_path}")
    with open(input_path, 'r', encoding='utf-8') as f:
        documents = json.load(f)

    N = len(documents)
    print(f"文档总数: {N}")

    # 统计每个词出现在多少个文档中 (Document Frequency)
    df_counter = Counter()
    for i, doc in enumerate(documents):
        if (i + 1) % 10000 == 0:
            print(f"  处理进度: {i + 1}/{N}")
        # 合并 title + content 提取词项
        text = (doc.get('title', '') + ' ' + doc.get('content', '') +
                ' ' + doc.get('content_full', ''))
        doc_terms = set(tokenize_simple(text))
        df_counter.update(doc_terms)

    print(f"唯一词项数: {len(df_counter)}")

    # 计算 IDF: log(N / (df + 1))
    idf_dict = {}
    for term, df in df_counter.items():
        idf_dict[term] = round(math.log(N / (df + 1)), 4)

    # 保存
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(idf_dict, f, ensure_ascii=False)

    # 统计信息
    idf_values = sorted(idf_dict.values())
    print(f"\n✓ IDF 字典已保存到 {output_path}")
    print(f"  词项总数: {len(idf_dict)}")
    print(f"  IDF 范围: {idf_values[0]:.4f} ~ {idf_values[-1]:.4f}")
    print(f"  IDF 中位数: {idf_values[len(idf_values) // 2]:.4f}")

    # 打印一些示例
    print("\n  高频词 (低 IDF):")
    low_idf = sorted(idf_dict.items(), key=lambda x: x[1])[:10]
    for term, idf in low_idf:
        print(f"    {term:20s}  IDF={idf:.4f}  DF={df_counter[term]}")

    print("\n  低频词 (高 IDF):")
    high_idf = sorted(idf_dict.items(), key=lambda x: x[1], reverse=True)[:10]
    for term, idf in high_idf:
        print(f"    {term:20s}  IDF={idf:.4f}  DF={df_counter[term]}")

    return idf_dict


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='从语料库构建 IDF 字典')
    parser.add_argument('--input', type=str, default=None,
                        help='输入语料库 JSON 文件路径')
    parser.add_argument('--output', type=str, default='models/idf_dict.json',
                        help='输出 IDF 字典路径 (默认: models/idf_dict.json)')
    args = parser.parse_args()

    # 自动检测输入文件
    if args.input is None:
        candidates = [
            'data/msmarco_100k_processed.json',
            'data/msmarco_docs_processed.json',
            'data/msmarco_100k.json',
            'data/msmarco_docs.json',
        ]
        for c in candidates:
            if os.path.exists(c):
                args.input = c
                break
        if args.input is None:
            print("✗ 未找到语料库文件，请用 --input 指定路径")
            sys.exit(1)

    build_idf(args.input, args.output)
