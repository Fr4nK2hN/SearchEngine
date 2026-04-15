import importlib
import json
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
        return results

    def train(self, *args, **kwargs):
        return self

    def save_model(self, filepath):
        return filepath


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


def load_app_module(log_path):
    runtime = SimpleNamespace(
        config=SimpleNamespace(
            ltr_model_path="models/ltr_model_manual.pkl",
            query_router_model_path="models/query_router_retrieval.pkl",
            recall_relax_threshold=5,
            adaptive_hard_top_k_cap=10,
            adaptive_hard_threshold=0.60,
            adaptive_guardrails=frozenset({"hard_question_prefix_baseline"}),
            adaptive_baseline_min_top_score=40.0,
            log_file=log_path,
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
        return importlib.import_module("app")


class AppSmokeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        handle = tempfile.NamedTemporaryFile("w", delete=False, encoding="utf-8")
        handle.close()
        cls.log_path = handle.name
        cls.app_module = load_app_module(cls.log_path)
        cls.client = cls.app_module.app.test_client()

    @classmethod
    def tearDownClass(cls):
        if os.path.exists(cls.log_path):
            os.unlink(cls.log_path)

    def test_search_route_returns_results(self):
        fake_hits = [
            {
                "_id": "doc-1",
                "_score": 1.25,
                "_source": {"title": "Alpha", "content": "Alpha content"},
            }
        ]
        with mock.patch.object(
            self.app_module.app_state.search_service,
            "retrieval_fn",
            return_value=(fake_hits, "strict"),
        ):
            response = self.client.get("/search?q=alpha&mode=baseline")

        self.assertEqual(response.status_code, 200)
        payload = response.get_json()
        self.assertEqual(len(payload["results"]), 1)
        self.assertEqual(payload["results"][0]["_id"], "doc-1")
        self.assertEqual(payload["results"][0]["_ranking_method"], "Baseline (ES only)")
        self.assertTrue(payload["search_id"])

    def test_model_info_reports_loaded_runtime(self):
        response = self.client.get("/model_info")

        self.assertEqual(response.status_code, 200)
        payload = response.get_json()
        self.assertEqual(payload["status"], "trained")
        self.assertEqual(payload["router"]["status"], "loaded")
        self.assertIn("feature_a", payload["top_features"])

    def test_export_data_reads_structured_log(self):
        events = [
            {"type": "query_submitted", "sessionId": "sess-1", "query": "alpha"},
            {
                "type": "search_completed",
                "sessionId": "sess-1",
                "searchId": "search-1",
                "query": "alpha",
            },
            {
                "type": "result_clicked",
                "sessionId": "sess-1",
                "searchId": "search-1",
                "rank": 1,
            },
        ]
        with open(self.log_path, "w", encoding="utf-8") as handle:
            for event in events:
                handle.write(json.dumps(event) + "\n")

        response = self.client.get("/export_data?session_id=sess-1")

        self.assertEqual(response.status_code, 200)
        payload = response.get_json()
        self.assertEqual(payload["summary"]["total_sessions"], 1)
        self.assertEqual(payload["summary"]["query_stats"]["total_queries"], 1)
        self.assertEqual(payload["summary"]["interaction_stats"]["total_clicks"], 1)


if __name__ == "__main__":
    unittest.main()
