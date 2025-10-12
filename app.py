from flask import Flask, request, jsonify, render_template
from elasticsearch import Elasticsearch
from elasticsearch.helpers import bulk
import json
import logging
import os
from sentence_transformers import CrossEncoder

app = Flask(__name__)
es = Elasticsearch([{'host': 'elasticsearch', 'port': 9200, 'scheme': 'http'}])

# Load the cross-encoder model
model = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')

# Setup logging
logging.basicConfig(filename='logs/events.log', level=logging.INFO, format='%(asctime)s %(message)s')

# Function to create index and bulk index data
def create_index_and_bulk_index():
    """
    Creates an Elasticsearch index and bulk indexes the MS MARCO documents if the index doesn't exist.
    """
    index_name = "documents"
    if not es.indices.exists(index=index_name):
        print(f"Index '{index_name}' does not exist. Creating and indexing...")
        mappings = {
            "properties": {
                "title": {
                    "type": "text",
                    "analyzer": "standard",
                    "fields": {
                        "keyword": {"type": "keyword"}
                    }
                },
                "content": {
                    "type": "text",
                    "analyzer": "standard"
                },
                "related_queries": {
                    "type": "text",
                    "analyzer": "standard"
                },
                "combined_text": {
                    "type": "text",
                    "analyzer": "standard"
                }
            }
        }
        es.indices.create(index=index_name, mappings=mappings)
        print(f"Index '{index_name}' created.")

        with open("data/msmarco_docs.json", "r", encoding="utf-8") as f:
            documents = json.load(f)

        actions = []
        for doc in documents:
            # 创建组合文本字段以提高搜索效果
            combined_text = f"{doc['title']} {doc['content']}"
            if doc.get('related_queries'):
                combined_text += " " + " ".join(doc['related_queries'])
            
            actions.append({
                "_index": index_name,
                "_id": doc["id"],
                "_source": {
                    "title": doc["title"],
                    "content": doc["content"],
                    "related_queries": doc.get("related_queries", []),
                    "combined_text": combined_text
                }
            })

        bulk(es, actions)
        print(f"Bulk indexing completed. Indexed {len(actions)} documents.")
    else:
        print(f"Index '{index_name}' already exists.")




@app.route('/')
def index():
    return render_template('index.html')

@app.route('/search')
def search():
    query = request.args.get('q', '')
    mode = request.args.get('mode', 'optimal')
    
    if not query:
        return jsonify([])
    
    try:
        # Enhanced Elasticsearch query with multiple strategies
        es_query = {
            "query": {
                "bool": {
                    "should": [
                        # Multi-match across key fields with different boosts
                        {
                            "multi_match": {
                                "query": query,
                                "fields": ["title^3", "content^2", "combined_text^1.5"],
                                "type": "best_fields",
                                "fuzziness": "AUTO"
                            }
                        },
                        # Phrase matching for content (higher precision)
                        {
                            "match_phrase": {
                                "content": {
                                    "query": query,
                                    "boost": 2
                                }
                            }
                        },
                        # Related queries matching
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
            "size": 50  # Get more results for manipulation
        }
        
        # Execute search
        response = es.search(index="documents", body=es_query)
        results = response['hits']['hits']
        
        # Re-rank using cross-encoder
        if results:
            passages = [hit['_source']['content'] for hit in results]
            pairs = [[query, passage] for passage in passages]
            
            cross_encoder_scores = model.predict(pairs)
            
            # Combine scores: weighted average of ES score and cross-encoder score
            for i, hit in enumerate(results):
                original_score = hit['_score']
                cross_score = float(cross_encoder_scores[i])
                # Weighted combination: 30% ES score, 70% cross-encoder score
                combined_score = 0.3 * original_score + 0.7 * cross_score
                hit['_score'] = combined_score
                hit['_relevance_category'] = 'high' if cross_score > 0.5 else ('medium' if cross_score > 0.2 else 'low')
            
            # Sort by combined score
            results.sort(key=lambda x: x['_score'], reverse=True)
        
        # Apply experimental modes for research purposes
        if mode == 'optimal':
            # Return best results (baseline condition)
            final_results = results[:10]
            
        elif mode == 'degraded_relevance':
            # Mix high-quality results with lower-quality ones
            if len(results) >= 15:
                high_quality = [r for r in results if r.get('_relevance_category') == 'high'][:3]
                medium_quality = [r for r in results if r.get('_relevance_category') == 'medium'][:4]
                low_quality = [r for r in results if r.get('_relevance_category') == 'low'][:3]
                
                # Mix them in a specific pattern for research
                final_results = []
                for i in range(10):
                    if i < 3 and i < len(high_quality):
                        final_results.append(high_quality[i])
                    elif i < 7 and (i-3) < len(medium_quality):
                        final_results.append(medium_quality[i-3])
                    elif (i-7) < len(low_quality):
                        final_results.append(low_quality[i-7])
                    elif i < len(results):
                        final_results.append(results[i])
            else:
                final_results = results[:10]
                
        elif mode == 'highly_irrelevant':
            # Experimental condition: mostly irrelevant results
            if len(results) >= 15:
                high_quality = [r for r in results if r.get('_relevance_category') == 'high'][:2]
                low_quality = [r for r in results if r.get('_relevance_category') == 'low'][:8]
                
                final_results = high_quality + low_quality
                # Shuffle to avoid obvious patterns
                import random
                random.shuffle(final_results[2:])  # Keep first 2 relevant, shuffle the rest
            else:
                final_results = results[:10]
                
        elif mode == 'mixed_coherence':
            # Mix results from different topics/domains
            if len(results) >= 20:
                # Take results from different score ranges to create incoherent SERP
                final_results = []
                final_results.extend(results[0:2])    # Top 2 relevant
                final_results.extend(results[10:13])  # Mid-range results
                final_results.extend(results[20:25])  # Lower relevance results
                final_results = final_results[:10]
            else:
                final_results = results[:10]
        else:
            final_results = results[:10]
        
        # Clean up temporary fields before returning
        for result in final_results:
            if '_relevance_category' in result:
                del result['_relevance_category']
        
        return jsonify(final_results)
        
    except Exception as e:
        print(f"Search error: {e}")
        return jsonify({"error": "Search failed"}), 500

@app.route('/log', methods=['POST'])
def log_event():
    events = request.get_json()
    if not events:
        return jsonify({"status": "error", "message": "No data provided"}), 400

    for event in events:
        logging.info(json.dumps(event))

    return jsonify({"status": "success"}), 200

@app.route('/export_data')
def export_data():
    """Export collected user interaction data for research analysis"""
    try:
        # Read the events log file
        events_data = []
        if os.path.exists('logs/events.log'):
            with open('logs/events.log', 'r') as f:
                for line in f:
                    try:
                        events_data.extend(json.loads(line.strip()))
                    except json.JSONDecodeError:
                        continue
        
        # Create summary statistics
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
        
        # Prepare export data
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
    """Simple research dashboard to view collected data"""
    try:
        # Read recent events
        events_data = []
        if os.path.exists('logs/events.log'):
            with open('logs/events.log', 'r') as f:
                for line in f.readlines()[-100:]:  # Last 100 lines
                    try:
                        events_data.extend(json.loads(line.strip()))
                    except json.JSONDecodeError:
                        continue
        
        # Group by session
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
            <title>Research Dashboard</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                .session {{ border: 1px solid #ccc; margin: 10px 0; padding: 10px; }}
                .event {{ margin: 5px 0; padding: 5px; background: #f5f5f5; }}
                .summary {{ background: #e7f3ff; padding: 15px; margin-bottom: 20px; }}
            </style>
        </head>
        <body>
            <h1>Search Engine Research Dashboard</h1>
            <div class="summary">
                <h2>Summary</h2>
                <p>Total Sessions: {len(sessions)}</p>
                <p>Total Events: {len(events_data)}</p>
            </div>
            <h2>Recent Sessions</h2>
        """
        
        for session_id, session_events in list(sessions.items())[:10]:  # Show last 10 sessions
            dashboard_html += f"""
            <div class="session">
                <h3>Session: {session_id}</h3>
                <p>Events: {len(session_events)}</p>
            """
            
            for event in session_events[-5:]:  # Show last 5 events per session
                dashboard_html += f"""
                <div class="event">
                    <strong>{event.get('type', 'unknown')}</strong> - 
                    {event.get('timestamp', 'no timestamp')}
                    {f" - Query: {event.get('query', '')}" if event.get('query') else ""}
                </div>
                """
            
            dashboard_html += "</div>"
        
        dashboard_html += """
        </body>
        </html>
        """
        
        return dashboard_html
        
    except Exception as e:
        return f"<html><body><h1>Error</h1><p>{str(e)}</p></body></html>"


if __name__ == '__main__':
    create_index_and_bulk_index()
    app.run(host='0.0.0.0', port=5000)