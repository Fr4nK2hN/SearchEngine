import importlib
import os
import sys
import tempfile
import unittest
from types import SimpleNamespace
from unittest import mock


class FakeFeatureExtractor:
    def get_feature_names(self):
        return ["feature_a", "feature_b"]


class FakeLTRRanker:
    def __init__(self):
        self.is_trained = True
        self.feature_names = ["feature_a", "feature_b"]
        self.model = SimpleNamespace(
            feature_importances_=[2.0, 1.0],
            n_estimators=100,
            max_depth=6,
            learning_rate=0.05,
        )
        self.last_timing = {"feature_ms": 0.0, "inference_ms": 0.0}

    def rerank(self, query, results, top_n=None):
        for item in results:
            item["_ltr_score"] = item.get("_score", 0.0)
        return results


class FakeQueryRouter:
    def __init__(self):
        self.loaded = True
        self.easy_mode = "baseline"
        self.hard_mode = "ltr"
        self.hard_threshold = 0.6062
        self.hard_top_k = 5
        self.hard_topk_policy = []
        self.load_error = None

    def route(self, query):
        return {
            "route_label": "easy",
            "route_confidence": 0.8,
            "route_source": "model",
            "selected_mode": "baseline",
            "hard_top_k": 5,
        }


def load_app_module():
    handle = tempfile.NamedTemporaryFile("w", delete=False, encoding="utf-8")
    handle.close()
    runtime = SimpleNamespace(
        config=SimpleNamespace(
            ltr_model_path="models/ltr_model_manual.pkl",
            query_router_model_path="models/query_router_retrieval.pkl",
            recall_relax_threshold=5,
            adaptive_hard_top_k_cap=10,
            adaptive_hard_threshold=0.60,
            adaptive_guardrails=frozenset({"hard_question_prefix_baseline"}),
            adaptive_baseline_min_top_score=40.0,
            log_file=handle.name,
            index_name="documents",
        ),
        es=SimpleNamespace(),
        cross_encoder_model=SimpleNamespace(predict=lambda pairs: [0.9 for _ in pairs]),
        feature_extractor=FakeFeatureExtractor(),
        ltr_ranker=FakeLTRRanker(),
        query_router=FakeQueryRouter(),
        logger=mock.Mock(),
    )

    sys.modules.pop("app", None)
    with mock.patch("webapp.bootstrap.build_runtime", return_value=runtime):
        module = importlib.import_module("app")

    return module, handle.name


class AppCompatibilityTests(unittest.TestCase):
    def test_legacy_exports_remain_available_for_tools(self):
        app_module, log_path = load_app_module()
        self.addCleanup(lambda: os.path.exists(log_path) and os.unlink(log_path))

        self.assertEqual(app_module.ADAPTIVE_HARD_TOP_K_CAP, 10)
        self.assertEqual(app_module.RECALL_RELAX_THRESHOLD, 5)
        self.assertTrue(app_module._is_ltr_available())

        results, method, feature_ms, inference_ms = app_module._apply_ranking_mode(
            "alpha",
            [{"_id": "doc-1", "_score": 1.0, "_source": {"content": "alpha content"}}],
            "baseline",
        )

        self.assertEqual(method, "Baseline (ES only)")
        self.assertEqual(feature_ms, 0.0)
        self.assertEqual(inference_ms, 0.0)
        self.assertEqual(results[0]["_id"], "doc-1")


if __name__ == "__main__":
    unittest.main()
