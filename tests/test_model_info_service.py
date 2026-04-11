import unittest
from types import SimpleNamespace

from webapp.services.model_info import build_model_info_payload


class ModelInfoServiceTests(unittest.TestCase):
    def test_build_model_info_payload_for_untrained_ltr(self):
        query_router = SimpleNamespace(
            loaded=False,
            easy_mode="baseline",
            hard_mode="cross_encoder",
            hard_threshold=0.6062,
            hard_top_k=5,
            hard_topk_policy=[],
            load_error="missing model",
        )

        payload = build_model_info_payload(
            None,
            query_router,
            router_model_path="models/query_router.pkl",
            hard_threshold_override=0.6062,
            hard_top_k_cap=5,
            adaptive_guardrails=frozenset({"hard_question_prefix_baseline"}),
            adaptive_baseline_min_top_score=40.0,
        )

        self.assertEqual(payload["status"], "not_trained")
        self.assertEqual(payload["router"]["status"], "heuristic_fallback")
        self.assertEqual(payload["router"]["load_error"], "missing model")

    def test_build_model_info_payload_for_trained_ltr(self):
        query_router = SimpleNamespace(
            loaded=True,
            easy_mode="baseline",
            hard_mode="cross_encoder",
            hard_threshold=0.6062,
            hard_top_k=5,
            hard_topk_policy=[],
            load_error=None,
        )
        ltr_ranker = SimpleNamespace(
            is_trained=True,
            feature_names=["feature_a", "feature_b"],
            model=SimpleNamespace(
                feature_importances_=[1.0, 3.0],
                n_estimators=100,
                max_depth=6,
                learning_rate=0.05,
            ),
        )

        payload = build_model_info_payload(
            ltr_ranker,
            query_router,
            router_model_path="models/query_router.pkl",
            hard_threshold_override=0.6062,
            hard_top_k_cap=5,
            adaptive_guardrails=frozenset({"hard_question_prefix_baseline"}),
            adaptive_baseline_min_top_score=40.0,
        )

        self.assertEqual(payload["status"], "trained")
        self.assertEqual(payload["n_features"], 2)
        self.assertEqual(list(payload["top_features"].keys())[0], "feature_b")
        self.assertEqual(payload["router"]["status"], "loaded")


if __name__ == "__main__":
    unittest.main()
