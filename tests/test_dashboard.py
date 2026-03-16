import unittest

from webapp.dashboard import group_events_by_session, render_research_dashboard_html


class DashboardTests(unittest.TestCase):
    def test_group_events_by_session_uses_existing_fallback_order(self):
        events = [
            {"sessionId": "session-1", "type": "query_submitted"},
            {"searchId": "search-2", "type": "search_completed"},
            {"search_id": "search-3", "event": "result_click_confirmed"},
            {"type": "orphan_event"},
        ]

        sessions = group_events_by_session(events)

        self.assertEqual(list(sessions.keys()), ["session-1", "search-2", "search-3", "unknown"])
        self.assertEqual(sessions["unknown"][0]["type"], "orphan_event")

    def test_render_research_dashboard_html_contains_summary_and_session_details(self):
        summary = {
            "total_events": 3,
            "query_stats": {"total_queries": 1},
            "interaction_stats": {"total_clicks": 1},
            "feedback_stats": {
                "ctr_at_3": 0.5,
                "avg_click_rank": 1.0,
                "abandonment_rate": 0.25,
            },
            "latency_stats": {"avg_total_ms": 120.0, "p95_total_ms": 200.0},
            "adaptive_stats": {"hard_rate": 0.4},
        }
        sessions = {
            "session-1": [
                {
                    "type": "query_submitted",
                    "timestamp": "2026-03-15T10:00:00",
                    "query": "alpha",
                    "rankingMethod": "Baseline (ES only)",
                }
            ]
        }

        html = render_research_dashboard_html(
            summary,
            sessions,
            ltr_available=True,
            router_loaded=False,
            feature_count=12,
        )

        self.assertIn("Search Research Dashboard", html)
        self.assertIn("Total Sessions", html)
        self.assertIn(">1</div>", html)
        self.assertIn("Session: session-1...", html)
        self.assertIn("Baseline (ES only)", html)
        self.assertIn("Heuristic", html)
        self.assertIn(">12</div>", html)


if __name__ == "__main__":
    unittest.main()
