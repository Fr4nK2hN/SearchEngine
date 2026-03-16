import time

from ranking.query_router import adaptive_guardrail


def to_positive_int(value):
    """将值转换为正整数，失败返回 None。"""
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return None
    return parsed if parsed > 0 else None


def normalize_scores(scores):
    """Min-Max 归一化分数到 [0, 1]，解决不同模型输出量纲不一致的问题。"""
    if not scores:
        return scores
    min_score = min(scores)
    max_score = max(scores)
    if max_score == min_score:
        return [0.5] * len(scores)
    return [(score - min_score) / (max_score - min_score) for score in scores]


class SearchPipeline:
    def __init__(
        self,
        *,
        ltr_ranker,
        cross_encoder_model,
        query_router,
        adaptive_hard_top_k_cap,
    ):
        self.ltr_ranker = ltr_ranker
        self.cross_encoder_model = cross_encoder_model
        self.query_router = query_router
        self.adaptive_hard_top_k_cap = adaptive_hard_top_k_cap

    def is_ltr_available(self):
        return bool(self.ltr_ranker and self.ltr_ranker.is_trained)

    def apply_adaptive_guardrails(self, query, route):
        """Apply lightweight runtime guardrails on top of the learned router."""
        if not isinstance(route, dict):
            return route

        selected_mode = route.get("selected_mode") or "baseline"
        override = adaptive_guardrail(
            query=query,
            route_label=route.get("route_label"),
            selected_mode=selected_mode,
            ltr_available=self.is_ltr_available(),
        )
        if not override:
            return route

        guarded = dict(route)
        guarded.update(override)
        return guarded

    def cross_encoder_rerank(self, query, results, top_n=None):
        """使用 Cross-Encoder 重排序（带分数归一化），支持仅重排前 N 条。"""
        if not results:
            return results

        if top_n is None:
            top_n = len(results)
        top_n = max(1, min(int(top_n), len(results)))

        head = results[:top_n]
        passages = [hit["_source"]["content"] for hit in head]
        pairs = [[query, passage] for passage in passages]
        cross_scores = self.cross_encoder_model.predict(pairs)

        es_scores = [hit["_score"] for hit in head]
        es_norm = normalize_scores(es_scores)
        ce_norm = normalize_scores([float(score) for score in cross_scores])

        for idx, hit in enumerate(head):
            hit["_cross_score"] = float(cross_scores[idx])
            hit["_score"] = 0.3 * es_norm[idx] + 0.7 * ce_norm[idx]

        head.sort(key=lambda item: item["_score"], reverse=True)
        results[:top_n] = head
        return results

    def hybrid_rerank(self, query, results, top_n=10):
        """LTR + Cross-Encoder 混合重排序（带分数归一化）。"""
        results = self.ltr_ranker.rerank(query, results)
        ltr_feat_ms = float(self.ltr_ranker.last_timing.get("feature_ms", 0.0))
        ltr_inf_ms = float(self.ltr_ranker.last_timing.get("inference_ms", 0.0))
        top_n = max(1, min(int(top_n), len(results)))
        top_results = results[:top_n]
        passages = [hit["_source"]["content"] for hit in top_results]
        pairs = [[query, passage] for passage in passages]
        t0 = time.perf_counter()
        cross_scores = self.cross_encoder_model.predict(pairs)
        t1 = time.perf_counter()

        ltr_scores = [hit.get("_ltr_score", hit["_score"]) for hit in top_results]
        ltr_norm = normalize_scores(ltr_scores)
        ce_norm = normalize_scores([float(score) for score in cross_scores])

        for idx, hit in enumerate(top_results):
            hit["_hybrid_components"] = {
                "ltr": float(ltr_scores[idx]),
                "cross": float(cross_scores[idx]),
            }
            hit["_score"] = 0.6 * ltr_norm[idx] + 0.4 * ce_norm[idx]
        top_results.sort(key=lambda item: item["_score"], reverse=True)
        results[:top_n] = top_results
        timing = {
            "feature_ms": ltr_feat_ms,
            "inference_ms": ltr_inf_ms + (t1 - t0) * 1000.0,
            "rerank_top_n": top_n,
        }
        return results, timing

    def apply_ranking_mode(self, query, results, mode, rerank_top_n=None):
        """统一执行排序模式。"""
        if mode == "ltr" and self.is_ltr_available():
            results = self.ltr_ranker.rerank(query, results)
            return (
                results,
                "LTR",
                float(self.ltr_ranker.last_timing.get("feature_ms", 0.0)),
                float(self.ltr_ranker.last_timing.get("inference_ms", 0.0)),
            )

        if mode == "cross_encoder":
            t0 = time.perf_counter()
            results = self.cross_encoder_rerank(query, results, top_n=rerank_top_n)
            t1 = time.perf_counter()
            return (results, "Cross-Encoder", 0.0, (t1 - t0) * 1000.0)

        if mode == "hybrid":
            if self.is_ltr_available():
                top_n = rerank_top_n if rerank_top_n is not None else 10
                results, timing = self.hybrid_rerank(query, results, top_n=top_n)
                return (
                    results,
                    f"Hybrid (LTR + Cross-Encoder, top-{int(top_n)})",
                    float(timing.get("feature_ms", 0.0)),
                    float(timing.get("inference_ms", 0.0)),
                )
            t0 = time.perf_counter()
            results = self.cross_encoder_rerank(query, results, top_n=rerank_top_n)
            t1 = time.perf_counter()
            return (results, "Cross-Encoder (LTR unavailable)", 0.0, (t1 - t0) * 1000.0)

        if mode == "baseline":
            return (results, "Baseline (ES only)", 0.0, 0.0)

        if self.is_ltr_available():
            results = self.ltr_ranker.rerank(query, results)
            return (
                results,
                "LTR",
                float(self.ltr_ranker.last_timing.get("feature_ms", 0.0)),
                float(self.ltr_ranker.last_timing.get("inference_ms", 0.0)),
            )
        t0 = time.perf_counter()
        results = self.cross_encoder_rerank(query, results, top_n=rerank_top_n)
        t1 = time.perf_counter()
        return (results, "Cross-Encoder", 0.0, (t1 - t0) * 1000.0)

    def resolve_adaptive_route(self, query):
        """根据 router 判定 easy/hard，并映射为最终可执行模式。"""
        route = self.apply_adaptive_guardrails(
            query,
            self.query_router.route(query),
        )
        selected_mode = route.get("selected_mode") or "baseline"
        route_label = route.get("route_label", "easy")
        hard_top_k = to_positive_int(route.get("hard_top_k")) or 30
        if self.adaptive_hard_top_k_cap is not None:
            hard_top_k = min(hard_top_k, self.adaptive_hard_top_k_cap)

        if selected_mode in ("ltr", "hybrid") and not self.is_ltr_available():
            selected_mode = "cross_encoder" if route_label == "hard" else "baseline"

        route["selected_mode"] = selected_mode
        route["hard_top_k"] = hard_top_k
        route["rerank_top_n"] = (
            hard_top_k
            if route_label == "hard" and selected_mode in ("cross_encoder", "hybrid")
            else None
        )
        return route
