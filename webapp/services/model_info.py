def build_router_info(
    query_router,
    *,
    model_path,
    hard_threshold_override,
    hard_top_k_cap,
    adaptive_guardrails,
    adaptive_baseline_min_top_score,
):
    return {
        "status": "loaded" if query_router.loaded else "heuristic_fallback",
        "model_path": model_path,
        "easy_mode": query_router.easy_mode,
        "hard_mode": query_router.hard_mode,
        "hard_threshold": query_router.hard_threshold,
        "hard_threshold_override": hard_threshold_override,
        "hard_top_k": query_router.hard_top_k,
        "hard_top_k_cap": hard_top_k_cap,
        "hard_topk_policy": query_router.hard_topk_policy,
        "adaptive_guardrails": "all" if adaptive_guardrails is None else sorted(adaptive_guardrails),
        "adaptive_baseline_min_top_score": adaptive_baseline_min_top_score,
        "feature_names": getattr(query_router, "feature_names", []),
        "load_error": query_router.load_error,
    }


def build_model_info_payload(
    ltr_ranker,
    query_router,
    *,
    router_model_path,
    hard_threshold_override,
    hard_top_k_cap,
    adaptive_guardrails,
    adaptive_baseline_min_top_score,
):
    router_info = build_router_info(
        query_router,
        model_path=router_model_path,
        hard_threshold_override=hard_threshold_override,
        hard_top_k_cap=hard_top_k_cap,
        adaptive_guardrails=adaptive_guardrails,
        adaptive_baseline_min_top_score=adaptive_baseline_min_top_score,
    )

    if not ltr_ranker or not ltr_ranker.is_trained:
        return {
            "status": "not_trained",
            "message": "LTR model not available",
            "router": router_info,
        }

    importances = ltr_ranker.model.feature_importances_
    feature_importance = {
        name: float(importance)
        for name, importance in zip(ltr_ranker.feature_names, importances)
    }
    sorted_features = sorted(
        feature_importance.items(),
        key=lambda item: item[1],
        reverse=True,
    )[:20]

    return {
        "status": "trained",
        "n_features": len(ltr_ranker.feature_names),
        "feature_names": ltr_ranker.feature_names,
        "top_features": dict(sorted_features),
        "model_params": {
            "n_estimators": ltr_ranker.model.n_estimators,
            "max_depth": ltr_ranker.model.max_depth,
            "learning_rate": ltr_ranker.model.learning_rate,
        },
        "router": router_info,
    }
