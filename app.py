from flask import Flask, request, jsonify, render_template
from elasticsearch import Elasticsearch
from elasticsearch.helpers import bulk
import json
import logging
import os
import math
from sentence_transformers import CrossEncoder
from pythonjsonlogger import jsonlogger
import uuid
import time

# 导入 LTR 相关模块
from ranking.feature_extractor import FeatureExtractor
from ranking.ranker import LTRRanker
from ranking.query_router import QueryRouter

app = Flask(__name__)
es = Elasticsearch([{'host': 'elasticsearch', 'port': 9200, 'scheme': 'http'}])

# 加载 Cross-Encoder 模型
cross_encoder_model = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')

# ========== 初始化 LTR 组件 ==========
print("Initializing LTR components...")
feature_extractor = FeatureExtractor()
ltr_ranker = LTRRanker(feature_extractor)

# 加载训练好的 LTR 模型（如果存在）
LTR_MODEL_PATH = 'models/ltr_model.pkl'
if os.path.exists(LTR_MODEL_PATH):
    try:
        ltr_ranker.load_model(LTR_MODEL_PATH)
        print(f"✓ LTR model loaded from {LTR_MODEL_PATH}")
    except Exception as e:
        print(f"✗ Failed to load LTR model: {e}")
        ltr_ranker = None
else:
    print(f"✗ LTR model not found at {LTR_MODEL_PATH}")
    ltr_ranker = None

# ========== 初始化 Query Router ==========
ROUTER_MODEL_PATH = 'models/query_router.pkl'
query_router = QueryRouter(
    model_path=ROUTER_MODEL_PATH,
    default_easy_mode='ltr',
    default_hard_mode='hybrid',
)
if query_router.loaded:
    print(f"✓ Query router loaded from {ROUTER_MODEL_PATH}")
else:
    print("✗ Query router model unavailable, fallback to heuristic routing")

# ========== Structured Logging Setup ==========
# Configure structured logging
log_dir = 'logs'
log_file = os.path.join(log_dir, 'events.log')
os.makedirs(log_dir, exist_ok=True)  # Ensure the log directory exists

log_handler = logging.FileHandler(log_file)
formatter = jsonlogger.JsonFormatter('%(asctime)s %(name)s %(levelname)s %(message)s')
log_handler.setFormatter(formatter)
logger = logging.getLogger('search_app')
logger.addHandler(log_handler)
logger.setLevel(logging.INFO)


def _parse_log_line_events(line):
    """从一行 JSON 日志中提取事件对象（兼容旧/新日志格式）。"""
    line = (line or '').strip()
    if not line:
        return []

    try:
        record = json.loads(line)
    except json.JSONDecodeError:
        return []

    if isinstance(record, list):
        return [item for item in record if isinstance(item, dict)]
    if not isinstance(record, dict):
        return []

    events = []

    # 兼容历史写法: logger.info(json.dumps(event))
    message = record.get('message')
    if isinstance(message, str):
        message_str = message.strip()
        if message_str[:1] in ('{', '['):
            try:
                parsed = json.loads(message_str)
                if isinstance(parsed, dict):
                    events.append(parsed)
                elif isinstance(parsed, list):
                    events.extend(item for item in parsed if isinstance(item, dict))
            except json.JSONDecodeError:
                pass

    # 当前结构化日志（/search、/track_click 等）直接读取顶层字段
    event_keys = {
        'type', 'event', 'searchId', 'search_id', 'sessionId',
        'query', 'rankingMethod', 'results_count', 'result_ids'
    }
    if any(k in record for k in event_keys):
        events.append(record)

    return events


def _load_events_from_log(limit=None):
    """读取并解析事件日志。limit 为只取最后 N 行。"""
    if not os.path.exists(log_file):
        return []

    with open(log_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    if limit and limit > 0:
        lines = lines[-limit:]

    events = []
    for line in lines:
        events.extend(_parse_log_line_events(line))
    return events


def _to_positive_int(value):
    """将值转换为正整数，失败返回 None。"""
    try:
        iv = int(value)
    except (TypeError, ValueError):
        return None
    return iv if iv > 0 else None


def _to_non_negative_float(value):
    """将值转换为非负浮点数，失败返回 None。"""
    try:
        fv = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(fv) or fv < 0:
        return None
    return fv


def _percentile(values, p):
    """计算分位数（线性插值）。"""
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
    weight = pos - lo
    return arr[lo] * (1 - weight) + arr[hi] * weight


def _build_event_summary(events_data):
    """构建导出/仪表盘共用的事件汇总。"""
    session_ids = {
        event.get('sessionId')
        for event in events_data
        if isinstance(event, dict) and event.get('sessionId')
    }

    summary = {
        'total_sessions': len(session_ids),
        'total_events': len(events_data),
        'event_types': {},
        'query_stats': {
            'total_queries': 0,
            'unique_queries': set(),
            'abandoned_queries': 0
        },
        'interaction_stats': {
            'total_clicks': 0,
            'confirmed_clicks': 0,
            'total_scrolls': 0,
            'average_session_duration': 0
        },
        'feedback_stats': {
            'total_searches': 0,
            'searches_with_click': 0,
            'ctr': 0.0,
            'ctr_at_1': 0.0,
            'ctr_at_3': 0.0,
            'ctr_at_10': 0.0,
            'avg_click_rank': 0.0,
            'median_click_rank': 0.0,
            'clicks_per_query': 0.0,
            'abandonment_rate': 0.0
        },
        'latency_stats': {
            'sample_count': 0,
            'avg_total_ms': 0.0,
            'p95_total_ms': 0.0,
            'avg_retrieval_ms': 0.0,
            'avg_feature_ms': 0.0,
            'avg_inference_ms': 0.0,
            'by_ranking_method': {}
        },
        'adaptive_stats': {
            'total_routed': 0,
            'easy_count': 0,
            'hard_count': 0,
            'hard_rate': 0.0,
            'avg_confidence': 0.0,
            'model_routed': 0,
            'heuristic_routed': 0
        }
    }

    session_durations = []
    completed_search_ids = []
    click_ranks_by_search = {}
    all_click_ranks = []
    latency_total = []
    latency_retrieval = []
    latency_feature = []
    latency_inference = []
    latency_by_method = {}
    adaptive_confidence = []

    for event in events_data:
        if not isinstance(event, dict):
            continue

        event_type = event.get('type') or event.get('event') or 'unknown'
        summary['event_types'][event_type] = summary['event_types'].get(event_type, 0) + 1

        total_ms = _to_non_negative_float(event.get('total_ms'))
        if total_ms is not None:
            latency_total.append(total_ms)
            method = str(event.get('rankingMethod') or 'unknown')
            latency_by_method.setdefault(method, []).append(total_ms)

            retrieval_ms = _to_non_negative_float(event.get('retrieval_ms'))
            feature_ms = _to_non_negative_float(event.get('feature_ms'))
            inference_ms = _to_non_negative_float(event.get('inference_ms'))
            if retrieval_ms is not None:
                latency_retrieval.append(retrieval_ms)
            if feature_ms is not None:
                latency_feature.append(feature_ms)
            if inference_ms is not None:
                latency_inference.append(inference_ms)

        route_label = str(event.get('route_label') or '').strip().lower()
        if route_label in ('easy', 'hard'):
            summary['adaptive_stats']['total_routed'] += 1
            if route_label == 'easy':
                summary['adaptive_stats']['easy_count'] += 1
            else:
                summary['adaptive_stats']['hard_count'] += 1

            route_confidence = _to_non_negative_float(event.get('route_confidence'))
            if route_confidence is not None:
                adaptive_confidence.append(min(1.0, route_confidence))

            route_source = str(event.get('route_source') or '').strip().lower()
            if route_source == 'model':
                summary['adaptive_stats']['model_routed'] += 1
            elif route_source == 'heuristic':
                summary['adaptive_stats']['heuristic_routed'] += 1

        if event_type == 'query_submitted' and event.get('sessionId'):
            summary['query_stats']['total_queries'] += 1
            summary['query_stats']['unique_queries'].add(event.get('query', ''))

        elif event_type == 'query_abandoned':
            summary['query_stats']['abandoned_queries'] += 1

        elif event_type == 'search_completed' and event.get('sessionId'):
            summary['feedback_stats']['total_searches'] += 1
            search_id = event.get('searchId') or event.get('search_id')
            if search_id:
                completed_search_ids.append(str(search_id))

        elif event_type == 'result_clicked':
            summary['interaction_stats']['total_clicks'] += 1

            rank = _to_positive_int(event.get('rank'))
            if rank is not None:
                all_click_ranks.append(rank)

                search_id = event.get('searchId') or event.get('search_id')
                if search_id:
                    sid = str(search_id)
                    if sid not in click_ranks_by_search:
                        click_ranks_by_search[sid] = []
                    click_ranks_by_search[sid].append(rank)

        elif event_type == 'result_click_confirmed':
            summary['interaction_stats']['confirmed_clicks'] += 1

        elif event_type == 'scroll_action':
            summary['interaction_stats']['total_scrolls'] += 1

        elif event_type == 'session_end':
            duration = event.get('totalDuration', 0)
            if isinstance(duration, (int, float)) and duration > 0:
                session_durations.append(duration)

    summary['query_stats']['unique_queries'] = len(summary['query_stats']['unique_queries'])

    if session_durations:
        summary['interaction_stats']['average_session_duration'] = (
            sum(session_durations) / len(session_durations)
        )

    total_queries = summary['query_stats']['total_queries']
    total_clicks = summary['interaction_stats']['total_clicks']
    abandoned_queries = summary['query_stats']['abandoned_queries']

    if total_queries > 0:
        summary['feedback_stats']['clicks_per_query'] = total_clicks / total_queries
        summary['feedback_stats']['abandonment_rate'] = abandoned_queries / total_queries

    if completed_search_ids:
        clicked = 0
        clicked_at_1 = 0
        clicked_at_3 = 0
        clicked_at_10 = 0

        for search_id in completed_search_ids:
            ranks = click_ranks_by_search.get(search_id, [])
            if not ranks:
                continue

            clicked += 1
            min_rank = min(ranks)
            if min_rank <= 1:
                clicked_at_1 += 1
            if min_rank <= 3:
                clicked_at_3 += 1
            if min_rank <= 10:
                clicked_at_10 += 1

        denom = len(completed_search_ids)
        summary['feedback_stats']['searches_with_click'] = clicked
        summary['feedback_stats']['ctr'] = clicked / denom
        summary['feedback_stats']['ctr_at_1'] = clicked_at_1 / denom
        summary['feedback_stats']['ctr_at_3'] = clicked_at_3 / denom
        summary['feedback_stats']['ctr_at_10'] = clicked_at_10 / denom

    if all_click_ranks:
        ranks_sorted = sorted(all_click_ranks)
        n = len(ranks_sorted)
        summary['feedback_stats']['avg_click_rank'] = sum(ranks_sorted) / n
        if n % 2 == 1:
            summary['feedback_stats']['median_click_rank'] = float(ranks_sorted[n // 2])
        else:
            mid = n // 2
            summary['feedback_stats']['median_click_rank'] = (
                ranks_sorted[mid - 1] + ranks_sorted[mid]
            ) / 2.0

    if latency_total:
        summary['latency_stats']['sample_count'] = len(latency_total)
        summary['latency_stats']['avg_total_ms'] = sum(latency_total) / len(latency_total)
        summary['latency_stats']['p95_total_ms'] = _percentile(latency_total, 95)
    if latency_retrieval:
        summary['latency_stats']['avg_retrieval_ms'] = (
            sum(latency_retrieval) / len(latency_retrieval)
        )
    if latency_feature:
        summary['latency_stats']['avg_feature_ms'] = sum(latency_feature) / len(latency_feature)
    if latency_inference:
        summary['latency_stats']['avg_inference_ms'] = (
            sum(latency_inference) / len(latency_inference)
        )
    for method, vals in latency_by_method.items():
        summary['latency_stats']['by_ranking_method'][method] = {
            'count': len(vals),
            'avg_total_ms': sum(vals) / len(vals),
            'p95_total_ms': _percentile(vals, 95),
        }

    routed = summary['adaptive_stats']['total_routed']
    if routed > 0:
        summary['adaptive_stats']['hard_rate'] = (
            summary['adaptive_stats']['hard_count'] / routed
        )
        if adaptive_confidence:
            summary['adaptive_stats']['avg_confidence'] = (
                sum(adaptive_confidence) / len(adaptive_confidence)
            )

    return summary


def _normalize_scores(scores):
    """Min-Max 归一化分数到 [0, 1]，解决不同模型输出量纲不一致的问题"""
    if not scores:
        return scores
    min_s = min(scores)
    max_s = max(scores)
    if max_s == min_s:
        return [0.5] * len(scores)
    return [(s - min_s) / (max_s - min_s) for s in scores]


def create_index_and_bulk_index():
    """Check if index exists (used only when running app.py directly for dev)"""
    index_name = "documents"
    if not es.indices.exists(index=index_name):
        print(f"⚠ 索引 '{index_name}' 不存在。请先运行 init.py 初始化索引。")
        print("  docker-compose 部署时由 init 服务自动完成。")
        print("  本地开发时请执行: python init.py")
    else:
        count = es.count(index=index_name)['count']
        print(f"✓ 索引 '{index_name}' 已存在，包含 {count} 个文档")


def _is_ltr_available():
    return bool(ltr_ranker and ltr_ranker.is_trained)


def _apply_ranking_mode(query, results, mode, rerank_top_n=None):
    """
    统一执行排序模式。

    Returns:
        tuple: (results, ranking_method, feature_ms, inference_ms)
    """
    if mode == 'ltr' and _is_ltr_available():
        results = ltr_ranker.rerank(query, results)
        return (
            results,
            'LTR',
            float(ltr_ranker.last_timing.get('feature_ms', 0.0)),
            float(ltr_ranker.last_timing.get('inference_ms', 0.0)),
        )

    if mode == 'cross_encoder':
        t0 = time.perf_counter()
        results = cross_encoder_rerank(query, results, top_n=rerank_top_n)
        t1 = time.perf_counter()
        return (results, 'Cross-Encoder', 0.0, (t1 - t0) * 1000.0)

    if mode == 'hybrid':
        if _is_ltr_available():
            top_n = rerank_top_n if rerank_top_n is not None else 10
            results, timing = hybrid_rerank(query, results, top_n=top_n)
            return (
                results,
                f'Hybrid (LTR + Cross-Encoder, top-{int(top_n)})',
                float(timing.get('feature_ms', 0.0)),
                float(timing.get('inference_ms', 0.0)),
            )
        # LTR 不可用时 hard 模式退化为 CE
        t0 = time.perf_counter()
        results = cross_encoder_rerank(query, results, top_n=rerank_top_n)
        t1 = time.perf_counter()
        return (results, 'Cross-Encoder (LTR unavailable)', 0.0, (t1 - t0) * 1000.0)

    if mode == 'baseline':
        return (results, 'Baseline (ES only)', 0.0, 0.0)

    # 默认回退：优先 LTR，其次 Cross-Encoder
    if _is_ltr_available():
        results = ltr_ranker.rerank(query, results)
        return (
            results,
            'LTR',
            float(ltr_ranker.last_timing.get('feature_ms', 0.0)),
            float(ltr_ranker.last_timing.get('inference_ms', 0.0)),
        )
    t0 = time.perf_counter()
    results = cross_encoder_rerank(query, results, top_n=rerank_top_n)
    t1 = time.perf_counter()
    return (results, 'Cross-Encoder', 0.0, (t1 - t0) * 1000.0)


def _resolve_adaptive_route(query):
    """
    根据 router 判定 easy/hard，并映射为最终可执行模式。
    """
    route = query_router.route(query)
    selected_mode = route.get('selected_mode') or 'baseline'
    route_label = route.get('route_label', 'easy')
    hard_top_k = _to_positive_int(route.get('hard_top_k')) or 30

    if selected_mode in ('ltr', 'hybrid') and not _is_ltr_available():
        selected_mode = 'cross_encoder' if route_label == 'hard' else 'baseline'

    route['selected_mode'] = selected_mode
    route['hard_top_k'] = hard_top_k
    route['rerank_top_n'] = (
        hard_top_k
        if route_label == 'hard' and selected_mode in ('cross_encoder', 'hybrid')
        else None
    )
    return route


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/search')
def search():
    query = request.args.get('q', '')
    mode = request.args.get('mode', 'ltr')  # 默认使用 LTR
    search_id = str(uuid.uuid4())

    log_entry = {
        "type": "query_submitted",
        "searchId": search_id,
        "query": query,
        "mode": mode,
    }
    
    if not query:
        logger.info("Empty query received", extra=log_entry)
        return jsonify({"results": [], "search_id": search_id})
    
    try:
        t_start = time.perf_counter()
        # ========== Stage 1: Elasticsearch 召回 ==========
        # 对短单词禁用模糊匹配，避免 "java" 命中 "lava" 等错误召回
        tokens = query.strip().split()
        is_short_single_term = len(tokens) == 1 and len(tokens[0]) <= 4

        multi_match = {
            "query": query,
            "fields": ["title^3", "content^2", "combined_text^1.5"],
            "type": "best_fields"
        }
        if not is_short_single_term:
            multi_match["fuzziness"] = "AUTO"

        hl = request.args.get('hl', 'false').lower() == 'true'
        es_query = {
            "query": {
                "bool": {
                    "should": [
                        {"multi_match": multi_match},
                        {
                            "match_phrase": {
                                "content": {
                                    "query": query,
                                    "boost": 2
                                }
                            }
                        },
                        {
                            "match": {
                                "related_queries": {
                                    "query": query,
                                    "boost": 2
                                }
                            }
                        }
                    ],
                    "minimum_should_match": 1
                }
            },
            "size": 50
        }
        if hl:
            es_query["highlight"] = {"fields": {"title": {}, "content": {}}}
        
        t_es0 = time.perf_counter()
        response = es.search(index="documents", body=es_query)
        t_es1 = time.perf_counter()
        results = response['hits']['hits']
        
        if not results:
            total_ms = (time.perf_counter() - t_start) * 1000.0
            retrieval_ms = (t_es1 - t_es0) * 1000.0
            log_entry.update({
                "rankingMethod": "N/A",
                "results_count": 0,
                "result_ids": [],
                "total_ms": total_ms,
                "retrieval_ms": retrieval_ms,
                "feature_ms": 0.0,
                "inference_ms": 0.0
            })
            logger.info("No results found", extra=log_entry)
            return jsonify({"results": [], "search_id": search_id})
        
        # ========== Stage 2: 选择排序策略 ==========
        route_info = None
        exec_mode = mode
        rerank_top_n = None

        if mode == 'adaptive':
            route_info = _resolve_adaptive_route(query)
            exec_mode = route_info.get('selected_mode', 'baseline')
            rerank_top_n = route_info.get('rerank_top_n')

        results, ranking_method, feature_ms, inference_ms = _apply_ranking_mode(
            query,
            results,
            exec_mode,
            rerank_top_n=rerank_top_n,
        )

        if route_info:
            ranking_method = f"Adaptive ({route_info.get('route_label')} -> {exec_mode}) | {ranking_method}"
            log_entry.update({
                "route_label": route_info.get('route_label'),
                "route_confidence": float(route_info.get('route_confidence', 0.0)),
                "route_source": route_info.get('route_source'),
                "route_selected_mode": exec_mode,
                "route_rerank_top_n": rerank_top_n,
            })
        
        # ========== Stage 3: 返回结果 ==========
        final_results = results[:10]
        
        # 添加排序方法信息（用于分析）
        for result in final_results:
            result['_ranking_method'] = ranking_method

        total_ms = (time.perf_counter() - t_start) * 1000.0
        retrieval_ms = (t_es1 - t_es0) * 1000.0
        log_entry.update({
            "rankingMethod": ranking_method,
            "results_count": len(final_results),
            "result_ids": [r['_id'] for r in final_results],
            "total_ms": total_ms,
            "retrieval_ms": retrieval_ms,
            "feature_ms": feature_ms,
            "inference_ms": inference_ms
        })
        logger.info("Search successful", extra=log_entry)

        response_payload = {"results": final_results, "search_id": search_id}
        if route_info:
            response_payload["routing"] = route_info
        return jsonify(response_payload)
        
    except Exception as e:
        import traceback
        log_entry['error'] = str(e)
        log_entry['traceback'] = traceback.format_exc()
        logger.error("Search failed", extra=log_entry)
        return jsonify({"error": "Search failed", "details": str(e), "search_id": search_id}), 500


def cross_encoder_rerank(query, results, top_n=None):
    """使用 Cross-Encoder 重排序（带分数归一化），支持仅重排前 N 条。"""
    if not results:
        return results

    if top_n is None:
        top_n = len(results)
    top_n = max(1, min(int(top_n), len(results)))

    head = results[:top_n]
    passages = [hit['_source']['content'] for hit in head]
    pairs = [[query, passage] for passage in passages]
    cross_scores = cross_encoder_model.predict(pairs)

    es_scores = [hit['_score'] for hit in head]
    es_norm = _normalize_scores(es_scores)
    ce_norm = _normalize_scores([float(s) for s in cross_scores])

    for i, hit in enumerate(head):
        hit['_cross_score'] = float(cross_scores[i])
        hit['_score'] = 0.3 * es_norm[i] + 0.7 * ce_norm[i]

    head.sort(key=lambda x: x['_score'], reverse=True)
    results[:top_n] = head
    return results


def hybrid_rerank(query, results, top_n=10):
    """LTR + Cross-Encoder 混合重排序（带分数归一化）。"""
    results = ltr_ranker.rerank(query, results)
    ltr_feat_ms = float(ltr_ranker.last_timing.get('feature_ms', 0.0))
    ltr_inf_ms = float(ltr_ranker.last_timing.get('inference_ms', 0.0))
    top_n = max(1, min(int(top_n), len(results)))
    top_results = results[:top_n]
    passages = [hit['_source']['content'] for hit in top_results]
    pairs = [[query, passage] for passage in passages]
    t0 = time.perf_counter()
    cross_scores = cross_encoder_model.predict(pairs)
    t1 = time.perf_counter()

    # 分别归一化 LTR 分数和 CE 分数
    ltr_scores = [hit.get('_ltr_score', hit['_score']) for hit in top_results]
    ltr_norm = _normalize_scores(ltr_scores)
    ce_norm = _normalize_scores([float(s) for s in cross_scores])

    for i, hit in enumerate(top_results):
        hit['_hybrid_components'] = {
            'ltr': float(ltr_scores[i]),
            'cross': float(cross_scores[i])
        }
        hit['_score'] = 0.6 * ltr_norm[i] + 0.4 * ce_norm[i]
    top_results.sort(key=lambda x: x['_score'], reverse=True)
    results[:top_n] = top_results
    timing = {
        'feature_ms': ltr_feat_ms,
        'inference_ms': ltr_inf_ms + (t1 - t0) * 1000.0,
        'rerank_top_n': top_n,
    }
    return results, timing


@app.route('/log', methods=['POST'])
def log_event():
    events = request.get_json(silent=True)
    if not events:
        return jsonify({"status": "error", "message": "No data provided"}), 400

    if isinstance(events, dict):
        events = [events]
    if not isinstance(events, list):
        return jsonify({"status": "error", "message": "Invalid payload format"}), 400

    for event in events:
        if not isinstance(event, dict):
            continue
        # 结构化写入，避免在 message 字段嵌套 JSON 字符串。
        logger.info("client_event", extra=event)

    return jsonify({"status": "success"}), 200


@app.route('/export_data')
def export_data():
    """导出用户交互数据"""
    try:
        from datetime import datetime
        events_data = _load_events_from_log()
        session_id_filter = request.args.get('session_id', '').strip()
        if session_id_filter:
            events_data = [
                event for event in events_data
                if isinstance(event, dict) and event.get('sessionId') == session_id_filter
            ]
        summary = _build_event_summary(events_data)
        
        export_data = {
            'export_timestamp': datetime.now().isoformat(),
            'session_id_filter': session_id_filter or None,
            'summary': summary,
            'raw_events': events_data
        }
        
        return jsonify(export_data)
        
    except Exception as e:
        return jsonify({"error": f"Export failed: {str(e)}"}), 500


@app.route('/research_dashboard')
def research_dashboard():
    """研究仪表盘"""
    try:
        events_data = _load_events_from_log(limit=100)
        summary = _build_event_summary(events_data)
        latency_stats = summary.get('latency_stats', {})
        adaptive_stats = summary.get('adaptive_stats', {})
        
        sessions = {}
        for event in events_data:
            session_id = (
                event.get('sessionId')
                or event.get('searchId')
                or event.get('search_id')
                or 'unknown'
            )
            if session_id not in sessions:
                sessions[session_id] = []
            sessions[session_id].append(event)
        
        dashboard_html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Research Dashboard - LTR Enhanced</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; background: #f5f5f5; }}
                .container {{ max-width: 1200px; margin: 0 auto; }}
                .header {{ background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                          color: white; padding: 30px; border-radius: 10px; margin-bottom: 20px; }}
                .summary {{ background: white; padding: 20px; margin-bottom: 20px; 
                           border-radius: 10px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
                .stats-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); 
                              gap: 15px; margin-top: 15px; }}
                .stat-card {{ background: #f8f9fa; padding: 15px; border-radius: 8px; 
                             border-left: 4px solid #667eea; }}
                .stat-value {{ font-size: 28px; font-weight: bold; color: #667eea; }}
                .stat-label {{ color: #666; font-size: 14px; margin-top: 5px; }}
                .session {{ border: 1px solid #ddd; margin: 10px 0; padding: 15px; 
                           background: white; border-radius: 8px; }}
                .event {{ margin: 8px 0; padding: 10px; background: #f8f9fa; 
                         border-radius: 5px; font-size: 14px; }}
                .event-type {{ display: inline-block; padding: 3px 8px; background: #667eea; 
                              color: white; border-radius: 3px; font-size: 12px; font-weight: bold; }}
                h1, h2 {{ margin: 0; }}
                .ltr-badge {{ background: #28a745; color: white; padding: 5px 10px; 
                             border-radius: 5px; font-size: 14px; display: inline-block; }}
            </style>
        </head>
        <body>
            <div class="container">
                <div class="header">
                    <h1>🔬 Search Engine Research Dashboard</h1>
                    <p style="margin-top: 10px; opacity: 0.9;">
                        <span class="ltr-badge">LTR Enabled</span> 
                        Real-time monitoring of search behavior and ranking performance
                    </p>
                </div>
                
                <div class="summary">
                    <h2>📊 System Statistics</h2>
                    <div class="stats-grid">
                        <div class="stat-card">
                            <div class="stat-value">{len(sessions)}</div>
                            <div class="stat-label">Total Sessions</div>
                        </div>
                        <div class="stat-card">
                            <div class="stat-value">{summary['total_events']}</div>
                            <div class="stat-label">Total Events</div>
                        </div>
                        <div class="stat-card">
                            <div class="stat-value">{summary['query_stats']['total_queries']}</div>
                            <div class="stat-label">Total Queries</div>
                        </div>
                        <div class="stat-card">
                            <div class="stat-value">{summary['interaction_stats']['total_clicks']}</div>
                            <div class="stat-label">Result Clicks</div>
                        </div>
                        <div class="stat-card">
                            <div class="stat-value">{summary['feedback_stats']['ctr_at_3'] * 100:.1f}%</div>
                            <div class="stat-label">CTR@3</div>
                        </div>
                        <div class="stat-card">
                            <div class="stat-value">{summary['feedback_stats']['avg_click_rank']:.2f}</div>
                            <div class="stat-label">Avg Click Rank</div>
                        </div>
                        <div class="stat-card">
                            <div class="stat-value">{summary['feedback_stats']['abandonment_rate'] * 100:.1f}%</div>
                            <div class="stat-label">Abandonment Rate</div>
                        </div>
                        <div class="stat-card">
                            <div class="stat-value">{latency_stats.get('avg_total_ms', 0.0):.1f}</div>
                            <div class="stat-label">Avg Latency (ms)</div>
                        </div>
                        <div class="stat-card">
                            <div class="stat-value">{latency_stats.get('p95_total_ms', 0.0):.1f}</div>
                            <div class="stat-label">P95 Latency (ms)</div>
                        </div>
                        <div class="stat-card">
                            <div class="stat-value">{adaptive_stats.get('hard_rate', 0.0) * 100:.1f}%</div>
                            <div class="stat-label">Adaptive Hard Rate</div>
                        </div>
                        <div class="stat-card">
                            <div class="stat-value">{'✓' if ltr_ranker and ltr_ranker.is_trained else '✗'}</div>
                            <div class="stat-label">LTR Model Status</div>
                        </div>
                        <div class="stat-card">
                            <div class="stat-value">{'✓' if query_router.loaded else 'Heuristic'}</div>
                            <div class="stat-label">Router Status</div>
                        </div>
                        <div class="stat-card">
                            <div class="stat-value">{len(feature_extractor.get_feature_names())}</div>
                            <div class="stat-label">Feature Dimensions</div>
                        </div>
                    </div>
                </div>
                
                <div class="summary">
                    <h2>🔥 Recent Sessions</h2>
        """
        
        for session_id, session_events in list(sessions.items())[:10]:
            dashboard_html += f"""
            <div class="session">
                <h3>Session: {session_id[:16]}...</h3>
                <p style="color: #666;">Events: {len(session_events)}</p>
            """
            
            for event in session_events[-5:]:
                event_type = event.get('type') or event.get('event') or 'unknown'
                timestamp = event.get('timestamp') or event.get('asctime') or 'no timestamp'
                query = event.get('query', '')
                ranking_method = event.get('rankingMethod', 'N/A')
                
                dashboard_html += f"""
                <div class="event">
                    <span class="event-type">{event_type}</span>
                    <span style="color: #999; margin-left: 10px;">{timestamp}</span>
                    {f'<br><strong>Query:</strong> {query}' if query else ''}
                    {f'<br><strong>Ranking:</strong> {ranking_method}' if ranking_method != 'N/A' else ''}
                </div>
                """
            
            dashboard_html += "</div>"
        
        dashboard_html += """
                </div>
            </div>
        </body>
        </html>
        """
        
        return dashboard_html
        
    except Exception as e:
        return f"<html><body><h1>Error</h1><p>{str(e)}</p></body></html>"


# ========== 新增：训练 LTR 模型的端点 ==========

@app.route('/train_ltr', methods=['POST'])
def train_ltr_endpoint():
    """
    训练或重新训练 LTR 模型
    
    POST body:
    {
        "num_queries": 50,
        "n_estimators": 200,
        "learning_rate": 0.05
    }
    """
    try:
        params = request.get_json() or {}
        num_queries = params.get('num_queries', 50)
        n_estimators = params.get('n_estimators', 200)
        learning_rate = params.get('learning_rate', 0.05)
        
        from ranking.training_data_generator import TrainingDataGenerator
        
        # 生成训练数据
        generator = TrainingDataGenerator(es, cross_encoder_model)
        queries = generator.generate_training_queries(num_queries=num_queries)
        training_data = generator.generate_training_data(queries, docs_per_query=30)
        
        # 划分训练集和验证集
        split_idx = int(len(training_data) * 0.8)
        train_set = training_data[:split_idx]
        val_set = training_data[split_idx:]
        
        # 训练模型
        global ltr_ranker
        ltr_ranker = LTRRanker(feature_extractor)
        ltr_ranker.train(
            training_data=train_set,
            validation_data=val_set,
            n_estimators=n_estimators,
            learning_rate=learning_rate
        )
        
        # 保存模型
        os.makedirs('models', exist_ok=True)
        ltr_ranker.save_model(LTR_MODEL_PATH)
        
        return jsonify({
            "status": "success",
            "message": "LTR model trained successfully",
            "training_queries": len(train_set),
            "validation_queries": len(val_set)
        })
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500


@app.route('/model_info')
def model_info():
    """获取 LTR 模型信息"""
    router_info = {
        "status": "loaded" if query_router.loaded else "heuristic_fallback",
        "model_path": ROUTER_MODEL_PATH,
        "easy_mode": query_router.easy_mode,
        "hard_mode": query_router.hard_mode,
        "hard_threshold": query_router.hard_threshold,
        "hard_top_k": query_router.hard_top_k,
        "hard_topk_policy": query_router.hard_topk_policy,
        "load_error": query_router.load_error,
    }

    if not ltr_ranker or not ltr_ranker.is_trained:
        return jsonify({
            "status": "not_trained",
            "message": "LTR model not available",
            "router": router_info,
        })
    
    # 获取特征重要性
    importances = ltr_ranker.model.feature_importances_
    feature_importance = {
        name: float(importance)
        for name, importance in zip(ltr_ranker.feature_names, importances)
    }
    
    # 排序
    sorted_features = sorted(
        feature_importance.items(),
        key=lambda x: x[1],
        reverse=True
    )[:20]
    
    return jsonify({
        "status": "trained",
        "n_features": len(ltr_ranker.feature_names),
        "feature_names": ltr_ranker.feature_names,
        "top_features": dict(sorted_features),
        "model_params": {
            "n_estimators": ltr_ranker.model.n_estimators,
            "max_depth": ltr_ranker.model.max_depth,
            "learning_rate": ltr_ranker.model.learning_rate
        },
        "router": router_info,
    })


@app.route('/track_click', methods=['POST'])
def track_click():
    data = request.get_json(silent=True) or {}
    log_entry = {
        "event": "result_click_confirmed",
        "sessionId": data.get("session_id"),
        "search_id": data.get("search_id"),
        "doc_id": data.get("doc_id"),
        "rank": data.get("rank"),
        "query": data.get("query"),
    }
    logger.info("Document clicked", extra=log_entry)
    return jsonify({"status": "success"}), 200


if __name__ == '__main__':
    create_index_and_bulk_index()
    app.run(host='0.0.0.0', port=5000, debug=True)
