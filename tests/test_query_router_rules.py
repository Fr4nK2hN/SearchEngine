from ranking.query_router import adaptive_guardrail


def test_hard_question_prefix_baseline_guardrail():
    override = adaptive_guardrail(
        query="who coined the",
        route_label="hard",
        selected_mode="cross_encoder",
        ltr_available=True,
    )

    assert override == {
        "selected_mode": "baseline",
        "route_guardrail": "hard_question_prefix_baseline",
    }


def test_hard_long_mix_ltr_guardrail():
    override = adaptive_guardrail(
        query="how long cooking chicken legs in the big easy",
        route_label="hard",
        selected_mode="cross_encoder",
        ltr_available=True,
    )

    assert override == {
        "selected_mode": "ltr",
        "route_guardrail": "hard_long_mix_ltr",
    }


def test_easy_topical_ltr_guardrail():
    override = adaptive_guardrail(
        query="machine learning algorithms",
        route_label="easy",
        selected_mode="baseline",
        ltr_available=True,
    )

    assert override == {
        "selected_mode": "ltr",
        "route_guardrail": "topical_easy_ltr",
    }


def test_guardrail_can_be_disabled_for_ablation():
    override = adaptive_guardrail(
        query="machine learning algorithms",
        route_label="easy",
        selected_mode="baseline",
        ltr_available=True,
        enabled_guardrails={"hard_question_prefix_baseline", "hard_long_mix_ltr"},
    )

    assert override is None


def test_two_term_hard_entity_has_no_override():
    override = adaptive_guardrail(
        query="investment strategies",
        route_label="hard",
        selected_mode="cross_encoder",
        ltr_available=True,
    )

    assert override is None
