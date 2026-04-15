import time

from ..config import ALLOWED_ROUTER_MODES
from engine.retrieval import search_documents_with_fallback
from ..search_pipeline import to_positive_int


VALID_SEARCH_MODES = frozenset(set(ALLOWED_ROUTER_MODES) | {"adaptive"})


def normalize_search_mode(mode):
    normalized = (mode or "adaptive").strip().lower()
    if normalized not in VALID_SEARCH_MODES:
        return "adaptive"
    return normalized


def build_ltr_runtime_metrics(search_pipeline, exec_mode):
    if exec_mode not in {"ltr", "hybrid"}:
        return {}
    ltr_ranker = getattr(search_pipeline, "ltr_ranker", None)
    if ltr_ranker is None:
        return {}
    timing = getattr(ltr_ranker, "last_timing", None) or {}
    metric_names = (
        "candidate_docs",
        "precomputed_content_emb_count",
        "precomputed_title_emb_count",
        "encoded_content_emb_count",
        "encoded_title_emb_count",
    )
    return {
        name: int(timing.get(name, 0) or 0)
        for name in metric_names
        if name in timing
    }


class SearchService:
    def __init__(
        self,
        *,
        es,
        logger,
        search_pipeline,
        index_name,
        recall_relax_threshold,
        retrieval_fn=search_documents_with_fallback,
    ):
        self.es = es
        self.logger = logger
        self.search_pipeline = search_pipeline
        self.index_name = index_name
        self.recall_relax_threshold = recall_relax_threshold
        self.retrieval_fn = retrieval_fn

    def execute(
        self,
        *,
        query,
        mode="adaptive",
        session_id="",
        search_id,
        rerank_top_n=None,
        hl=False,
    ):
        mode = normalize_search_mode(mode)
        log_entry = {
            "type": "query_submitted",
            "searchId": search_id,
            "query": query,
            "mode": mode,
        }
        if session_id:
            log_entry["sessionId"] = session_id

        if not query:
            self.logger.info("Empty query received", extra=log_entry)
            return {"results": [], "search_id": search_id}, 200

        try:
            t_start = time.perf_counter()
            requested_rerank_top_n = to_positive_int(rerank_top_n)
            t_es0 = time.perf_counter()
            results, retrieval_strategy = self.retrieval_fn(
                self.es,
                query,
                size=50,
                hl=hl,
                relax_threshold=self.recall_relax_threshold,
                index_name=self.index_name,
            )
            t_es1 = time.perf_counter()

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
                    "inference_ms": 0.0,
                    "retrieval_strategy": retrieval_strategy,
                })
                self.logger.info("No results found", extra=log_entry)
                return {"results": [], "search_id": search_id}, 200

            route_info = None
            exec_mode = mode
            rerank_limit = None

            if mode == "adaptive":
                route_info = self.search_pipeline.resolve_adaptive_route(query, results=results)
                exec_mode = route_info.get("selected_mode", "baseline")
                rerank_limit = route_info.get("rerank_top_n")
            elif mode in {"cross_encoder", "hybrid", "ltr"}:
                rerank_limit = requested_rerank_top_n

            results, ranking_method, feature_ms, inference_ms = self.search_pipeline.apply_ranking_mode(
                query,
                results,
                exec_mode,
                rerank_top_n=rerank_limit,
            )
            ltr_runtime_metrics = build_ltr_runtime_metrics(self.search_pipeline, exec_mode)

            if route_info:
                ranking_method = f"Adaptive ({route_info.get('route_label')} -> {exec_mode}) | {ranking_method}"
                log_entry.update({
                    "route_label": route_info.get("route_label"),
                    "route_confidence": float(route_info.get("route_confidence", 0.0)),
                    "route_source": route_info.get("route_source"),
                    "route_guardrail": route_info.get("route_guardrail"),
                    "route_selected_mode": exec_mode,
                    "route_rerank_top_n": rerank_limit,
                })

            final_results = results[:10]
            for result in final_results:
                result["_ranking_method"] = ranking_method

            total_ms = (time.perf_counter() - t_start) * 1000.0
            retrieval_ms = (t_es1 - t_es0) * 1000.0
            log_entry.update({
                "rankingMethod": ranking_method,
                "results_count": len(final_results),
                "result_ids": [item["_id"] for item in final_results],
                "result_scores": [float(item.get("_score", 0.0) or 0.0) for item in final_results],
                "total_ms": total_ms,
                "retrieval_ms": retrieval_ms,
                "feature_ms": feature_ms,
                "inference_ms": inference_ms,
                "retrieval_strategy": retrieval_strategy,
            })
            log_entry.update(ltr_runtime_metrics)
            self.logger.info("Search successful", extra=log_entry)

            response_payload = {"results": final_results, "search_id": search_id}
            if route_info:
                response_payload["routing"] = route_info
            if requested_rerank_top_n is not None:
                response_payload["requested_rerank_top_n"] = requested_rerank_top_n
            return response_payload, 200

        except Exception as exc:
            import traceback

            log_entry["error"] = str(exc)
            log_entry["traceback"] = traceback.format_exc()
            self.logger.error("Search failed", extra=log_entry)
            return {
                "error": "Search failed",
                "details": str(exc),
                "search_id": search_id,
            }, 500
