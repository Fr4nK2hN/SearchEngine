import unittest

from webapp.analytics import build_event_summary, parse_log_line_events


class AnalyticsTests(unittest.TestCase):
    def test_parse_log_line_supports_structured_and_legacy_message(self):
        structured = parse_log_line_events(
            '{"type":"query_submitted","sessionId":"s1","query":"alpha"}'
        )
        legacy = parse_log_line_events(
            '{"message":"{\\"type\\": \\"result_clicked\\", \\"rank\\": 2}"}'
        )

        self.assertEqual(structured[0]["query"], "alpha")
        self.assertEqual(legacy[0]["type"], "result_clicked")
        self.assertEqual(legacy[0]["rank"], 2)

    def test_build_event_summary_keeps_existing_counting_behavior(self):
        events = [
            {"type": "query_submitted", "sessionId": "sess-1", "query": "alpha"},
            {
                "type": "search_completed",
                "sessionId": "sess-1",
                "searchId": "search-1",
                "query": "alpha",
                "route_label": "hard",
                "route_source": "model",
                "route_confidence": 0.8,
                "rankingMethod": "Adaptive",
                "total_ms": 120.0,
                "retrieval_ms": 20.0,
                "feature_ms": 40.0,
                "inference_ms": 60.0,
            },
            {"type": "result_clicked", "sessionId": "sess-1", "searchId": "search-1", "rank": 2},
            {"event": "result_click_confirmed", "sessionId": "sess-1", "search_id": "search-1", "rank": 2},
            {"type": "session_end", "sessionId": "sess-1", "totalDuration": 3000},
        ]

        summary = build_event_summary(events)

        self.assertEqual(summary["total_sessions"], 1)
        self.assertEqual(summary["query_stats"]["total_queries"], 1)
        self.assertEqual(summary["feedback_stats"]["total_searches"], 1)
        self.assertEqual(summary["interaction_stats"]["total_clicks"], 1)
        self.assertEqual(summary["interaction_stats"]["confirmed_clicks"], 1)
        self.assertEqual(summary["adaptive_stats"]["hard_count"], 1)
        self.assertEqual(summary["adaptive_stats"]["model_routed"], 1)
        self.assertAlmostEqual(summary["latency_stats"]["avg_total_ms"], 120.0)


if __name__ == "__main__":
    unittest.main()
