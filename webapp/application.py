import uuid
from dataclasses import dataclass

from flask import jsonify, render_template, request

from . import bootstrap
from .search_pipeline import SearchPipeline
from .services.documents import fetch_documents_by_ids
from .services.events import build_click_log_entry, log_client_events
from .services.exporting import build_dashboard_html, build_export_payload
from .services.model_info import build_model_info_payload
from .services.search import SearchService
from .services.training import train_ltr_ranker


@dataclass
class WebAppState:
    runtime: object
    runtime_config: object
    es: object
    cross_encoder_model: object
    feature_extractor: object
    ltr_ranker: object
    query_router: object
    logger: object
    search_pipeline: SearchPipeline
    search_service: SearchService


def build_app_state(runtime=None):
    runtime = runtime or bootstrap.build_runtime()
    runtime_config = runtime.config
    search_pipeline = SearchPipeline(
        ltr_ranker=runtime.ltr_ranker,
        cross_encoder_model=runtime.cross_encoder_model,
        query_router=runtime.query_router,
        adaptive_hard_top_k_cap=runtime_config.adaptive_hard_top_k_cap,
        adaptive_guardrails=runtime_config.adaptive_guardrails,
        adaptive_baseline_min_top_score=runtime_config.adaptive_baseline_min_top_score,
    )
    search_service = SearchService(
        es=runtime.es,
        logger=runtime.logger,
        search_pipeline=search_pipeline,
        index_name=runtime_config.index_name,
        recall_relax_threshold=runtime_config.recall_relax_threshold,
    )
    return WebAppState(
        runtime=runtime,
        runtime_config=runtime_config,
        es=runtime.es,
        cross_encoder_model=runtime.cross_encoder_model,
        feature_extractor=runtime.feature_extractor,
        ltr_ranker=runtime.ltr_ranker,
        query_router=runtime.query_router,
        logger=runtime.logger,
        search_pipeline=search_pipeline,
        search_service=search_service,
    )


def register_routes(app, state):
    @app.route("/")
    def index():
        return render_template("index.html")

    @app.route("/search")
    def search():
        payload, status_code = state.search_service.execute(
            query=request.args.get("q", ""),
            mode=request.args.get("mode", "adaptive"),
            session_id=request.args.get("session_id", "").strip(),
            search_id=str(uuid.uuid4()),
            rerank_top_n=request.args.get("rerank_top_n"),
            hl=request.args.get("hl", "false").lower() == "true",
        )
        return jsonify(payload), status_code

    @app.route("/log", methods=["POST"])
    def log_event():
        payload, status_code = log_client_events(state.logger, request.get_json(silent=True))
        return jsonify(payload), status_code

    @app.route("/export_data")
    def export_data():
        try:
            return jsonify(build_export_payload(
                state.runtime_config.log_file,
                session_id_filter=request.args.get("session_id", "").strip(),
            ))
        except Exception as exc:
            return jsonify({"error": f"Export failed: {str(exc)}"}), 500

    @app.route("/documents_by_ids")
    def documents_by_ids():
        return jsonify({
            "documents": fetch_documents_by_ids(
                state.es,
                state.runtime_config.index_name,
                request.args.getlist("id"),
            )
        })

    @app.route("/research_dashboard")
    def research_dashboard():
        try:
            return build_dashboard_html(
                state.runtime_config.log_file,
                ltr_available=bool(state.ltr_ranker and state.ltr_ranker.is_trained),
                router_loaded=bool(state.query_router.loaded),
                feature_count=len(state.feature_extractor.get_feature_names()),
            )
        except Exception as exc:
            return f"<html><body><h1>Error</h1><p>{str(exc)}</p></body></html>"

    @app.route("/train_ltr", methods=["POST"])
    def train_ltr_endpoint():
        try:
            params = request.get_json() or {}
            state.ltr_ranker, train_count, val_count = train_ltr_ranker(
                state.es,
                state.cross_encoder_model,
                state.feature_extractor,
                model_path=state.runtime_config.ltr_model_path,
                num_queries=params.get("num_queries", 50),
                n_estimators=params.get("n_estimators", 200),
                learning_rate=params.get("learning_rate", 0.05),
            )
            state.search_pipeline.ltr_ranker = state.ltr_ranker

            return jsonify({
                "status": "success",
                "message": "LTR model trained successfully",
                "training_queries": train_count,
                "validation_queries": val_count,
            })
        except Exception as exc:
            import traceback

            traceback.print_exc()
            return jsonify({
                "status": "error",
                "message": str(exc),
            }), 500

    @app.route("/model_info")
    def model_info():
        return jsonify(build_model_info_payload(
            state.ltr_ranker,
            state.query_router,
            router_model_path=state.runtime_config.query_router_model_path,
            hard_threshold_override=state.runtime_config.adaptive_hard_threshold,
            hard_top_k_cap=state.runtime_config.adaptive_hard_top_k_cap,
            adaptive_guardrails=state.runtime_config.adaptive_guardrails,
            adaptive_baseline_min_top_score=state.runtime_config.adaptive_baseline_min_top_score,
        ))

    @app.route("/track_click", methods=["POST"])
    def track_click():
        state.logger.info(
            "Document clicked",
            extra=build_click_log_entry(request.get_json(silent=True) or {}),
        )
        return jsonify({"status": "success"}), 200

    return app


def create_index_and_bulk_index():
    """Use the same initialization path in direct-run dev mode and Docker."""
    from .ops.indexing import create_index_and_bulk_index as ops_create_index_and_bulk_index

    ops_create_index_and_bulk_index()
