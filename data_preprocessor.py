import ir_datasets
import json
from collections import defaultdict
from tqdm import tqdm


def preprocess_msmarco():
    # 加载 MS MARCO 段落开发集
    dataset = ir_datasets.load("msmarco-passage/dev")

    # 构建 query_id -> query_text 映射
    query_text_map = {}
    for q in tqdm(dataset.queries_iter(), desc="加载查询 (queries)"):
        query_text_map[q.query_id] = q.text

    # 从 qrels 构建 doc_id -> [query_id] 映射，并收集相关段落 ID
    doc_to_query_ids = defaultdict(list)
    for qrel in tqdm(dataset.qrels_iter(), desc="加载相关性判断 (qrels)"):
        if getattr(qrel, "relevance", 1) > 0:
            doc_to_query_ids[qrel.doc_id].append(qrel.query_id)

    relevant_passage_ids = set(doc_to_query_ids.keys())

    # 使用 docs_store 快速获取段落内容
    docstore = dataset.docs_store()

    documents = []
    for pid in tqdm(relevant_passage_ids, desc="获取段落正文 (docs_store)"):
        doc = docstore.get(pid)
        if doc is None:
            continue
        text = getattr(doc, "text", str(doc))

        # 将 title 设为段落前缀以模拟真实搜索体验，保留相关查询信息
        qids = doc_to_query_ids.get(pid, [])
        related_queries = [query_text_map.get(qid, "") for qid in qids if query_text_map.get(qid, "")]
        
        # 使用段落前120字符作为标题，更贴近真实搜索引擎
        title = text[:120].strip()
        if not title.endswith('.') and not title.endswith('!') and not title.endswith('?'):
            title += "..."

        documents.append({
            "id": pid,
            "title": title,
            "content": text,
            "related_queries": related_queries  # 保留相关查询信息供后续使用
        })

    print(f"相关段落数量: {len(relevant_passage_ids)}，索引文档数: {len(documents)}")

    # 保存为供 Elasticsearch 索引的 JSON 文件
    with open("data/msmarco_docs.json", "w", encoding="utf-8") as f:
        json.dump(documents, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    preprocess_msmarco()