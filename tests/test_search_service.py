import unittest
from unittest import mock

from webapp.services.search import SearchService, build_ltr_runtime_metrics


class DummySearchPipeline:
    def __init__(self):
        self.resolved_queries = []
        self.ranking_calls = []
        self.ltr_ranker = mock.Mock(
            last_timing={
                "candidate_docs": 50,
                "precomputed_content_emb_count": 40,
                "precomputed_title_emb_count": 38,
                "encoded_content_emb_count": 10,
                "encoded_title_emb_count": 12,
            }
        )

    def resolve_adaptive_route(self, query):
        self.resolved_queries.append(query)
        return {
            "route_label": "easy",
            "route_confidence": 0.8,
            "route_source": "model",
            "selected_mode": "baseline",
            "rerank_top_n": None,
        }

    def apply_ranking_mode(self, query, results, mode, rerank_top_n=None):
        self.ranking_calls.append((query, mode, rerank_top_n))
        return results, "Baseline (ES only)", 0.0, 0.0


class SearchServiceTests(unittest.TestCase):
    def test_build_ltr_runtime_metrics_only_emits_for_ltr_modes(self):
        pipeline = DummySearchPipeline()

        self.assertEqual(build_ltr_runtime_metrics(pipeline, "baseline"), {})
        self.assertEqual(
            build_ltr_runtime_metrics(pipeline, "ltr"),
            {
                "candidate_docs": 50,
                "precomputed_content_emb_count": 40,
                "precomputed_title_emb_count": 38,
                "encoded_content_emb_count": 10,
                "encoded_title_emb_count": 12,
            },
        )

    def test_execute_empty_query_keeps_existing_payload(self):
        logger = mock.Mock()
        service = SearchService(
            es=object(),
            logger=logger,
            search_pipeline=DummySearchPipeline(),
            index_name="documents",
            recall_relax_threshold=5,
            retrieval_fn=mock.Mock(),
        )

        payload, status_code = service.execute(
            query="",
            mode="baseline",
            session_id="sess-1",
            search_id="search-1",
        )

        self.assertEqual(status_code, 200)
        self.assertEqual(payload, {"results": [], "search_id": "search-1"})
        logger.info.assert_called_once()

    def test_execute_invalid_mode_falls_back_to_adaptive_and_returns_routing(self):
        logger = mock.Mock()
        pipeline = DummySearchPipeline()
        service = SearchService(
            es=object(),
            logger=logger,
            search_pipeline=pipeline,
            index_name="documents",
            recall_relax_threshold=5,
            retrieval_fn=mock.Mock(return_value=([
                {"_id": "doc-1", "_score": 1.0, "_source": {"content": "alpha content"}}
            ], "strict")),
        )

        payload, status_code = service.execute(
            query="alpha",
            mode="unknown",
            session_id="sess-1",
            search_id="search-1",
            rerank_top_n="15",
            hl=True,
        )

        self.assertEqual(status_code, 200)
        self.assertEqual(payload["search_id"], "search-1")
        self.assertEqual(payload["requested_rerank_top_n"], 15)
        self.assertEqual(payload["routing"]["selected_mode"], "baseline")
        self.assertEqual(
            payload["results"][0]["_ranking_method"],
            "Adaptive (easy -> baseline) | Baseline (ES only)",
        )
        self.assertEqual(pipeline.resolved_queries, ["alpha"])
        self.assertEqual(pipeline.ranking_calls, [("alpha", "baseline", None)])
        logger.info.assert_called_once()

    def test_execute_no_results_returns_empty_payload(self):
        logger = mock.Mock()
        service = SearchService(
            es=object(),
            logger=logger,
            search_pipeline=DummySearchPipeline(),
            index_name="documents",
            recall_relax_threshold=5,
            retrieval_fn=mock.Mock(return_value=([], "relaxed")),
        )

        payload, status_code = service.execute(
            query="alpha",
            mode="baseline",
            session_id="sess-1",
            search_id="search-1",
        )

        self.assertEqual(status_code, 200)
        self.assertEqual(payload, {"results": [], "search_id": "search-1"})
        logger.info.assert_called_once()

    def test_execute_ltr_logs_embedding_runtime_metrics(self):
        logger = mock.Mock()
        pipeline = DummySearchPipeline()

        def rerank(query, results, mode, rerank_top_n=None):
            return results, "LTR", 15.0, 2.0

        pipeline.apply_ranking_mode = rerank
        service = SearchService(
            es=object(),
            logger=logger,
            search_pipeline=pipeline,
            index_name="documents",
            recall_relax_threshold=5,
            retrieval_fn=mock.Mock(return_value=([
                {"_id": "doc-1", "_score": 1.0, "_source": {"content": "alpha content"}}
            ], "strict")),
        )

        payload, status_code = service.execute(
            query="alpha",
            mode="ltr",
            session_id="sess-1",
            search_id="search-1",
        )

        self.assertEqual(status_code, 200)
        self.assertEqual(payload["results"][0]["_ranking_method"], "LTR")
        logger.info.assert_called_once()
        extra = logger.info.call_args.kwargs["extra"]
        self.assertEqual(extra["candidate_docs"], 50)
        self.assertEqual(extra["precomputed_content_emb_count"], 40)
        self.assertEqual(extra["encoded_content_emb_count"], 10)


if __name__ == "__main__":
    unittest.main()
