import json
import re
from typing import List, Dict


class DataPreprocessor:
    """数据预处理器"""
    
    def __init__(self):
        self.max_title_length = 100
        self.max_content_length = 500
        self.min_content_length = 50

    def _read_first_non_ws_char(self, file_path: str) -> str:
        """
        读取文件的第一个非空白字符，用于判断是否是 JSON 数组或 JSON Lines。
        """
        with open(file_path, 'r', encoding='utf-8') as f:
            while True:
                ch = f.read(1)
                if not ch:
                    return ''
                if not ch.isspace():
                    return ch

    def _iter_jsonl(self, file_path: str):
        """
        逐行解析 JSON Lines 文件。
        """
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    yield json.loads(line)
                except json.JSONDecodeError:
                    # 跳过无法解析的行，避免中断整个流程
                    continue

    def _detect_msmarco_entry(self, entry: Dict) -> bool:
        """
        粗略检测条目是否为 MS MARCO QA 格式（包含 query / passages）。
        """
        if not isinstance(entry, dict):
            return False
        if 'query' in entry and isinstance(entry.get('passages'), dict):
            passages = entry.get('passages') or {}
            return 'passage_text' in passages
        return False

    def _convert_msmarco_entry_to_doc(self, entry: Dict) -> Dict:
        """
        将 MS MARCO 条目转换为本项目的文档结构：
        {id, title, content, related_queries, source_url}
        - 使用 query 作为标题
        - 使用被标记为 is_selected==1 的 passage 作为内容，若不存在则取第一个或前两个连接
        """
        query = entry.get('query', '')
        qid = entry.get('query_id', '')
        passages = entry.get('passages') or {}
        texts = passages.get('passage_text') or []
        is_selected = passages.get('is_selected') or []
        urls = passages.get('url') or []

        selected_idx = None
        if isinstance(is_selected, list) and 1 in is_selected:
            try:
                selected_idx = is_selected.index(1)
            except ValueError:
                selected_idx = None

        main_text = ''
        if selected_idx is not None and isinstance(texts, list) and 0 <= selected_idx < len(texts):
            main_text = texts[selected_idx]
        else:
            # 没有标注选中段落时，尽量组合前两个段落
            if isinstance(texts, list) and texts:
                main_text = ' '.join(texts[:2])

        doc = {
            'id': str(qid) if qid is not None else '',
            'title': query or '',
            'content': main_text or '',
            'related_queries': [query] if query else []
        }
        # 记录来源 URL（若有选中段落则取其 URL，否则取第一个）
        if isinstance(urls, list) and urls:
            if selected_idx is not None and 0 <= selected_idx < len(urls):
                doc['source_url'] = urls[selected_idx]
            else:
                doc['source_url'] = urls[0]
        return doc
        
    def clean_text(self, text: str) -> str:
        """
        清洗文本
        - 移除多余空格
        - 移除特殊字符
        - 规范化标点符号
        """
        if not text:
            return ""
        
        # 移除多余空格和换行
        text = re.sub(r'\s+', ' ', text)
        
        # 移除 HTML 标签（如果有）
        text = re.sub(r'<[^>]+>', '', text)
        
        # 规范化引号（将中文/弯引号替换为英文直引号）
        text = text.replace('“', '"').replace('”', '"')
        text = text.replace('‘', "'").replace('’', "'")
        
        # 移除多余的标点符号
        text = re.sub(r'\.{2,}', '...', text)
        
        # 移除开头和结尾的空格
        text = text.strip()
        
        return text
    
    def extract_title(self, text: str, max_length: int = 100) -> str:
        """
        从文本中提取标题
        如果没有明显标题，从内容开头提取
        """
        text = self.clean_text(text)
        
        if not text:
            return "Untitled Document"
        
        # 尝试提取第一句话作为标题
        sentences = re.split(r'[.!?]', text)
        if sentences and sentences[0]:
            title = sentences[0].strip()
            
            # 如果第一句话太长，截断
            if len(title) > max_length:
                title = title[:max_length-3] + "..."
            
            return title
        
        # 如果没有句子，直接截断
        if len(text) > max_length:
            return text[:max_length-3] + "..."
        
        return text
    
    def summarize_content(self, text: str, max_length: int = 500) -> str:
        """
        总结内容，保留最相关的部分
        """
        text = self.clean_text(text)
        
        if not text:
            return ""
        
        # 如果内容太短，直接返回
        if len(text) <= max_length:
            return text
        
        # 尝试保留完整句子
        sentences = re.split(r'(?<=[.!?])\s+', text)
        
        summary = ""
        for sentence in sentences:
            if len(summary) + len(sentence) <= max_length:
                summary += sentence + " "
            else:
                break
        
        summary = summary.strip()
        
        # 如果没有获取到任何句子，强制截断
        if not summary:
            summary = text[:max_length-3] + "..."
        elif len(summary) < max_length and len(text) > len(summary):
            summary += "..."
        
        return summary
    
    def extract_keywords(self, text: str, top_k: int = 5) -> List[str]:
        """
        提取关键词（简单版本）
        """
        text = self.clean_text(text.lower())
        
        # 移除停用词
        stop_words = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
            'of', 'with', 'by', 'from', 'as', 'is', 'was', 'are', 'were', 'be',
            'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will',
            'would', 'could', 'should', 'may', 'might', 'can', 'this', 'that',
            'these', 'those', 'it', 'its', 'they', 'their', 'them'
        }
        
        # 提取单词
        words = re.findall(r'\b[a-z]{3,}\b', text)
        
        # 过滤停用词
        words = [w for w in words if w not in stop_words]
        
        # 统计词频
        from collections import Counter
        word_freq = Counter(words)
        
        # 返回最常见的词
        return [word for word, _ in word_freq.most_common(top_k)]
    
    def process_document(self, doc: Dict) -> Dict:
        """
        处理单个文档
        """
        processed = {
            'id': doc.get('id', ''),
            'original_title': doc.get('title', ''),
            'original_content': doc.get('content', '')
        }
        
        # 清洗和提取标题
        title = doc.get('title', '')
        if not title or len(title.strip()) < 5:
            # 如果标题太短或不存在，从内容中提取
            title = self.extract_title(doc.get('content', ''))
        else:
            title = self.clean_text(title)
            if len(title) > self.max_title_length:
                title = title[:self.max_title_length-3] + "..."
        
        processed['title'] = title
        
        # 清洗和总结内容
        content = self.clean_text(doc.get('content', ''))
        
        # 如果内容太短，标记为低质量
        if len(content) < self.min_content_length:
            processed['quality'] = 'low'
        else:
            processed['quality'] = 'good'
        
        # 总结内容
        processed['content'] = self.summarize_content(content, self.max_content_length)
        processed['content_full'] = content  # 保留完整内容
        
        # 提取关键词
        processed['keywords'] = self.extract_keywords(content)
        
        # 保留相关查询
        processed['related_queries'] = doc.get('related_queries', [])
        
        # 生成组合文本用于搜索
        combined_text = f"{processed['title']} {processed['content']}"
        if processed['keywords']:
            combined_text += " " + " ".join(processed['keywords'])
        if processed['related_queries']:
            combined_text += " " + " ".join(processed['related_queries'])
        
        processed['combined_text'] = combined_text
        
        return processed
    
    def process_dataset(self, input_file: str, output_file: str, limit: int | None = None):
        """
        处理整个数据集
        """
        print(f"正在读取数据: {input_file}")
        # 支持 JSON 数组与 JSON Lines，两种格式自动检测；
        # 若检测为 MS MARCO 条目，则进行字段映射。
        first_char = self._read_first_non_ws_char(input_file)
        documents: List[Dict] = []

        if first_char == '[':
            # JSON 数组
            with open(input_file, 'r', encoding='utf-8') as f:
                raw = json.load(f)
            if isinstance(raw, list) and raw:
                # 检测第一项是否为 MS MARCO 格式
                if self._detect_msmarco_entry(raw[0]):
                    iterable = raw if limit is None else raw[:limit]
                    for entry in iterable:
                        documents.append(self._convert_msmarco_entry_to_doc(entry))
                else:
                    # 已经是文档格式，直接使用
                    documents = raw if limit is None else raw[:limit]
        else:
            # 视为 JSON Lines
            count = 0
            for entry in self._iter_jsonl(input_file):
                if self._detect_msmarco_entry(entry):
                    documents.append(self._convert_msmarco_entry_to_doc(entry))
                elif isinstance(entry, dict):
                    documents.append(entry)
                count += 1
                if limit is not None and count >= limit:
                    break

        print(f"总共 {len(documents)} 个文档")
        
        processed_docs = []
        low_quality_count = 0
        
        for i, doc in enumerate(documents):
            if (i + 1) % 100 == 0:
                print(f"处理进度: {i + 1}/{len(documents)}")
            
            processed = self.process_document(doc)
            
            if processed['quality'] == 'low':
                low_quality_count += 1
            
            processed_docs.append(processed)
        
        print(f"\n处理完成:")
        print(f"  - 总文档数: {len(processed_docs)}")
        print(f"  - 低质量文档: {low_quality_count}")
        print(f"  - 高质量文档: {len(processed_docs) - low_quality_count}")
        
        # 保存处理后的数据
        print(f"\n保存到: {output_file}")
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(processed_docs, f, indent=2, ensure_ascii=False)
        
        print("✓ 数据预处理完成!")
        
        return processed_docs
    
    def generate_sample_queries(self, documents: List[Dict], num_queries: int = 20) -> List[str]:
        """
        从文档中生成示例查询
        """
        queries = []
        
        for doc in documents[:num_queries * 2]:
            # 从标题生成查询
            title_words = doc['title'].split()[:5]
            if len(title_words) >= 2:
                queries.append(' '.join(title_words[:3]))
            
            # 从关键词生成查询
            if doc.get('keywords'):
                queries.append(' '.join(doc['keywords'][:2]))
            
            # 从相关查询中选择
            if doc.get('related_queries'):
                queries.extend(doc['related_queries'][:1])
        
        # 去重
        queries = list(set(queries))[:num_queries]
        
        return queries


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='预处理 MS MARCO 数据')
    parser.add_argument('--input', type=str, default='data/msmarco_100k.json',
                       help='输入文件路径')
    parser.add_argument('--output', type=str, default='data/msmarco_100k_processed.json',
                       help='输出文件路径')
    parser.add_argument('--max-title', type=int, default=100,
                       help='最大标题长度')
    parser.add_argument('--max-content', type=int, default=500,
                       help='最大内容长度')
    parser.add_argument('--generate-queries', action='store_true',
                       help='生成示例查询')
    parser.add_argument('--limit', type=int, default=None,
                        help='仅处理前 N 条数据用于快速验证')
    
    args = parser.parse_args()
    
    # 创建预处理器
    preprocessor = DataPreprocessor()
    preprocessor.max_title_length = args.max_title
    preprocessor.max_content_length = args.max_content
    
    # 处理数据
    processed_docs = preprocessor.process_dataset(args.input, args.output, limit=args.limit)
    
    # 生成示例查询
    if args.generate_queries:
        print("\n生成示例查询...")
        queries = preprocessor.generate_sample_queries(processed_docs, num_queries=20)
        
        queries_file = 'data/sample_queries.json'
        with open(queries_file, 'w', encoding='utf-8') as f:
            json.dump(queries, f, indent=2, ensure_ascii=False)
        
        print(f"✓ 示例查询已保存到: {queries_file}")
        print("\n示例查询:")
        for i, q in enumerate(queries[:10], 1):
            print(f"  {i}. {q}")


if __name__ == '__main__':
    main()