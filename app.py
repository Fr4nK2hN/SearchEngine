from flask import Flask

from webapp.application import build_app_state, create_index_and_bulk_index, register_routes


app = Flask(__name__)
app_state = build_app_state()
app.config["searchengine_state"] = app_state
register_routes(app, app_state)

# Compatibility exports for offline analysis scripts that still import from app.
runtime_config = app_state.runtime_config
es = app_state.es
query_router = app_state.query_router
ADAPTIVE_HARD_TOP_K_CAP = runtime_config.adaptive_hard_top_k_cap
RECALL_RELAX_THRESHOLD = runtime_config.recall_relax_threshold


def _is_ltr_available():
    return app_state.search_pipeline.is_ltr_available()


def _apply_ranking_mode(query, results, mode, rerank_top_n=None):
    return app_state.search_pipeline.apply_ranking_mode(
        query,
        results,
        mode,
        rerank_top_n=rerank_top_n,
    )


if __name__ == '__main__':
    create_index_and_bulk_index()
    app.run(host='0.0.0.0', port=5000, debug=True)
