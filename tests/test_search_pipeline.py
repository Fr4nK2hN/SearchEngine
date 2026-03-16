import unittest

from webapp.search_pipeline import SearchPipeline, normalize_scores


class DummyLTRRanker:
    def __init__(self, is_trained):
        self.is_trained = is_trained
        self.last_timing = {"feature_ms": 11.0, "inference_ms": 7.0}

    def rerank(self, query, results):
        for item in results:
            item["_ltr_score"] = item.get("_score", 0.0)
        return results


class DummyCrossEncoder:
    def predict(self, pairs):
        return [float(idx + 1) for idx, _ in enumerate(pairs)]


class DummyRouter:
    def __init__(self, route):
        self._route = route

    def route(self, query):
        return dict(self._route)


class SearchPipelineTests(unittest.TestCase):
    def test_normalize_scores_keeps_same_behavior(self):
        self.assertEqual(normalize_scores([]), [])
        self.assertEqual(normalize_scores([3.0, 3.0]), [0.5, 0.5])
        self.assertEqual(normalize_scores([2.0, 4.0]), [0.0, 1.0])

    def test_resolve_adaptive_route_falls_back_when_ltr_unavailable(self):
        pipeline = SearchPipeline(
            ltr_ranker=DummyLTRRanker(is_trained=False),
            cross_encoder_model=DummyCrossEncoder(),
            query_router=DummyRouter(
                {
                    "route_label": "hard",
                    "route_confidence": 0.9,
                    "route_source": "model",
                    "selected_mode": "hybrid",
                    "hard_top_k": 50,
                }
            ),
            adaptive_hard_top_k_cap=20,
        )

        route = pipeline.resolve_adaptive_route("alpha beta gamma")

        self.assertEqual(route["selected_mode"], "cross_encoder")
        self.assertEqual(route["hard_top_k"], 20)
        self.assertEqual(route["rerank_top_n"], 20)

    def test_apply_ranking_mode_baseline_is_passthrough(self):
        results = [{"_id": "doc-1", "_score": 1.0, "_source": {"content": "alpha"}}]
        pipeline = SearchPipeline(
            ltr_ranker=DummyLTRRanker(is_trained=True),
            cross_encoder_model=DummyCrossEncoder(),
            query_router=DummyRouter({"route_label": "easy", "selected_mode": "baseline"}),
            adaptive_hard_top_k_cap=20,
        )

        reranked, method, feature_ms, inference_ms = pipeline.apply_ranking_mode(
            "alpha",
            results,
            "baseline",
        )

        self.assertIs(reranked, results)
        self.assertEqual(method, "Baseline (ES only)")
        self.assertEqual(feature_ms, 0.0)
        self.assertEqual(inference_ms, 0.0)


if __name__ == "__main__":
    unittest.main()
