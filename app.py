from flask import Flask, request, jsonify, render_template
from elasticsearch import Elasticsearch
from elasticsearch.helpers import bulk
import json
import logging
import os
from sentence_transformers import CrossEncoder
from pythonjsonlogger import jsonlogger
import uuid

# 导入 LTR 相关模块
from ranking.feature_extractor import FeatureExtractor
from ranking.ranker import LTRRanker

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


def create_index_and_bulk_index():
    """创建索引并导入处理后的数据"""
    index_name = "documents"
    
    if not es.indices.exists(index=index_name):
        print(f"索引 '{index_name}' 不存在，正在创建...")
        
        mappings = {
            "properties": {
                "title": {
                    "type": "text",
                    "analyzer": "standard",
                    "fields": {"keyword": {"type": "keyword"}}
                },
                "content": {
                    "type": "text",
                    "analyzer": "standard"
                },
                "content_full": {
                    "type": "text",
                    "analyzer": "standard"
                },
                "keywords": {
                    "type": "keyword"
                },
                "related_queries": {
                    "type": "text",
                    "analyzer": "standard"
                },
                "combined_text": {
                    "type": "text",
                    "analyzer": "standard"
                },
                "quality": {
                    "type": "keyword"
                }
            }
        }
        
        es.indices.create(index=index_name, mappings=mappings)
        print(f"索引 '{index_name}' 创建完成")

        # 尝试加载处理后的数据（优先 100k 文件名），如果不存在则使用原始数据
        data_paths_processed = [
            "data/msmarco_100k_processed.json",
            "data/msmarco_docs_processed.json"
        ]
        data_paths_raw = [
            "data/msmarco_100k.json",
            "data/msmarco_docs.json"
        ]

        documents = None
        for p in data_paths_processed:
            try:
                with open(p, "r", encoding="utf-8") as f:
                    documents = json.load(f)
                print(f"✓ 使用处理后的数据: {p}")
                break
            except FileNotFoundError:
                continue

        if documents is None:
            print("⚠ 未找到处理后的数据，尝试使用原始数据")
            for p in data_paths_raw:
                try:
                    with open(p, "r", encoding="utf-8") as f:
                        documents = json.load(f)
                    print(f"✓ 使用原始数据: {p}")
                    break
                except FileNotFoundError:
                    continue
            if documents is None:
                raise FileNotFoundError("未找到任何可用的数据文件，请先生成或下载数据集")

        actions = []
        for doc in documents:
            action = {
                "_index": index_name,
                "_id": doc.get("id", ""),
                "_source": {
                    "title": doc.get("title", ""),
                    "content": doc.get("content", ""),
                    "content_full": doc.get("content_full", doc.get("content", "")),
                    "keywords": doc.get("keywords", []),
                    "related_queries": doc.get("related_queries", []),
                    "combined_text": doc.get("combined_text", ""),
                    "quality": doc.get("quality", "unknown")
                }
            }
            actions.append(action)

        # 立即刷新索引，避免首次搜索空结果
        bulk(es, actions, refresh=True)
        print(f"✓ 批量索引完成，共 {len(actions)} 个文档")
    else:
        print(f"索引 '{index_name}' 已存在")


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
            "highlight": {
                "fields": {
                    "title": {},
                    "content": {}
                }
            },
            "size": 100  # 召回更多候选
        }
        
        response = es.search(index="documents", body=es_query)
        results = response['hits']['hits']
        
        if not results:
            log_entry['results_count'] = 0
            logger.info("No results found", extra=log_entry)
            return jsonify({"results": [], "search_id": search_id})
        
        # ========== Stage 2: 选择排序策略 ==========
        
        if mode == 'ltr' and ltr_ranker and ltr_ranker.is_trained:
            # 使用 LTR 模型重排序
            results = ltr_ranker.rerank(query, results)
            ranking_method = 'LTR'
            
        elif mode == 'cross_encoder':
            # 使用 Cross-Encoder 重排序
            results = cross_encoder_rerank(query, results)
            ranking_method = 'Cross-Encoder'
            
        elif mode == 'hybrid':
            # 混合方法：LTR + Cross-Encoder
            if ltr_ranker and ltr_ranker.is_trained:
                results = hybrid_rerank(query, results)
                ranking_method = 'Hybrid (LTR + Cross-Encoder)'
            else:
                results = cross_encoder_rerank(query, results)
                ranking_method = 'Cross-Encoder (LTR unavailable)'
                
        elif mode == 'baseline':
            # 仅使用 Elasticsearch 分数
            ranking_method = 'Baseline (ES only)'
            
        else:
            # 默认：如果有 LTR 就用 LTR，否则用 Cross-Encoder
            if ltr_ranker and ltr_ranker.is_trained:
                results = ltr_ranker.rerank(query, results)
                ranking_method = 'LTR'
            else:
                results = cross_encoder_rerank(query, results)
                ranking_method = 'Cross-Encoder'
        
        # ========== Stage 3: 返回结果 ==========
        final_results = results[:10]
        
        # 添加排序方法信息（用于分析）
        for result in final_results:
            result['_ranking_method'] = ranking_method

        log_entry.update({
            "rankingMethod": ranking_method,
            "results_count": len(final_results),
            "result_ids": [r['_id'] for r in final_results]
        })
        logger.info("Search successful", extra=log_entry)
        
        return jsonify({"results": final_results, "search_id": search_id})
        
    except Exception as e:
        import traceback
        log_entry['error'] = str(e)
        log_entry['traceback'] = traceback.format_exc()
        logger.error("Search failed", extra=log_entry)
        return jsonify({"error": "Search failed", "details": str(e), "search_id": search_id}), 500


def cross_encoder_rerank(query, results):
    """使用 Cross-Encoder 重排序"""
    passages = [hit['_source']['content'] for hit in results]
    pairs = [[query, passage] for passage in passages]
    
    cross_scores = cross_encoder_model.predict(pairs)
    
    for i, hit in enumerate(results):
        original_score = hit['_score']
        cross_score = float(cross_scores[i])
        # 混合分数
        hit['_score'] = 0.3 * original_score + 0.7 * cross_score
        hit['_cross_score'] = cross_score
    
    results.sort(key=lambda x: x['_score'], reverse=True)
    return results


def hybrid_rerank(query, results):
    """混合重排序：LTR + Cross-Encoder"""
    # 先用 LTR 重排
    results = ltr_ranker.rerank(query, results)
    
    # 再用 Cross-Encoder 微调 top 20
    top_results = results[:20]
    passages = [hit['_source']['content'] for hit in top_results]
    pairs = [[query, passage] for passage in passages]
    
    cross_scores = cross_encoder_model.predict(pairs)
    
    for i, hit in enumerate(top_results):
        ltr_score = hit.get('_ltr_score', hit['_score'])
        cross_score = float(cross_scores[i])
        # 混合 LTR 和 Cross-Encoder 分数
        hit['_score'] = 0.6 * ltr_score + 0.4 * cross_score
        hit['_hybrid_components'] = {
            'ltr': ltr_score,
            'cross': cross_score
        }
    
    top_results.sort(key=lambda x: x['_score'], reverse=True)
    
    # 合并回完整结果
    results[:20] = top_results
    return results


@app.route('/log', methods=['POST'])
def log_event():
    events = request.get_json()
    if not events:
        return jsonify({"status": "error", "message": "No data provided"}), 400

    for event in events:
        logger.info(json.dumps(event))

    return jsonify({"status": "success"}), 200


@app.route('/export_data')
def export_data():
    """导出用户交互数据"""
    try:
        from datetime import datetime
        events_data = []
        if os.path.exists('logs/events.log'):
            with open('logs/events.log', 'r') as f:
                for line in f:
                    try:
                        # 每行可能是单个事件或事件数组
                        line_data = json.loads(line.strip().split(' ', 1)[1])  # 跳过时间戳
                        if isinstance(line_data, list):
                            events_data.extend(line_data)
                        else:
                            events_data.append(line_data)
                    except (json.JSONDecodeError, IndexError):
                        continue
        
        summary = {
            'total_sessions': len(set(event.get('sessionId', '') for event in events_data)),
            'total_events': len(events_data),
            'event_types': {},
            'query_stats': {
                'total_queries': 0,
                'unique_queries': set(),
                'abandoned_queries': 0
            },
            'interaction_stats': {
                'total_clicks': 0,
                'total_scrolls': 0,
                'average_session_duration': 0
            }
        }
        
        session_durations = []
        
        for event in events_data:
            event_type = event.get('type', 'unknown')
            summary['event_types'][event_type] = summary['event_types'].get(event_type, 0) + 1
            
            if event_type == 'query_submitted':
                summary['query_stats']['total_queries'] += 1
                summary['query_stats']['unique_queries'].add(event.get('query', ''))
            elif event_type == 'query_abandoned':
                summary['query_stats']['abandoned_queries'] += 1
            elif event_type == 'result_clicked':
                summary['interaction_stats']['total_clicks'] += 1
            elif event_type == 'scroll_action':
                summary['interaction_stats']['total_scrolls'] += 1
            elif event_type == 'session_end':
                duration = event.get('totalDuration', 0)
                if duration > 0:
                    session_durations.append(duration)
        
        summary['query_stats']['unique_queries'] = len(summary['query_stats']['unique_queries'])
        if session_durations:
            summary['interaction_stats']['average_session_duration'] = sum(session_durations) / len(session_durations)
        
        export_data = {
            'export_timestamp': datetime.now().isoformat(),
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
        events_data = []
        if os.path.exists('logs/events.log'):
            with open('logs/events.log', 'r') as f:
                for line in f.readlines()[-100:]:
                    try:
                        line_data = json.loads(line.strip().split(' ', 1)[1])
                        if isinstance(line_data, list):
                            events_data.extend(line_data)
                        else:
                            events_data.append(line_data)
                    except (json.JSONDecodeError, IndexError):
                        continue
        
        sessions = {}
        for event in events_data:
            session_id = event.get('sessionId', 'unknown')
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
                            <div class="stat-value">{len(events_data)}</div>
                            <div class="stat-label">Total Events</div>
                        </div>
                        <div class="stat-card">
                            <div class="stat-value">{'✓' if ltr_ranker and ltr_ranker.is_trained else '✗'}</div>
                            <div class="stat-label">LTR Model Status</div>
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
                event_type = event.get('type', 'unknown')
                timestamp = event.get('timestamp', 'no timestamp')
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
    if not ltr_ranker or not ltr_ranker.is_trained:
        return jsonify({
            "status": "not_trained",
            "message": "LTR model not available"
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
        }
    })


@app.route('/track_click', methods=['POST'])
def track_click():
    data = request.get_json()
    log_entry = {
        "event": "click",
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