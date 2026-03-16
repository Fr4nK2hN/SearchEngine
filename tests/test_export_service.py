import unittest
from datetime import datetime

from webapp.services.exporting import build_dashboard_html, build_export_payload, filter_events_by_session


class ExportServiceTests(unittest.TestCase):
    def test_filter_events_by_session_keeps_matching_items(self):
        events = [
            {"sessionId": "sess-1", "type": "query_submitted"},
            {"sessionId": "sess-2", "type": "query_submitted"},
            {"type": "orphan"},
        ]

        filtered = filter_events_by_session(events, "sess-1")

        self.assertEqual(filtered, [{"sessionId": "sess-1", "type": "query_submitted"}])

    def test_build_export_payload_uses_filtered_events(self):
        payload = build_export_payload(
            "ignored.log",
            session_id_filter="sess-1",
            load_events_fn=lambda _: [
                {"sessionId": "sess-1", "type": "query_submitted"},
                {"sessionId": "sess-2", "type": "query_submitted"},
            ],
            summary_fn=lambda events: {"count": len(events)},
            now_factory=lambda: datetime(2026, 3, 15, 10, 0, 0),
        )

        self.assertEqual(payload["session_id_filter"], "sess-1")
        self.assertEqual(payload["summary"], {"count": 1})
        self.assertEqual(payload["export_timestamp"], "2026-03-15T10:00:00")

    def test_build_dashboard_html_wires_summary_sessions_and_rendering(self):
        html = build_dashboard_html(
            "ignored.log",
            ltr_available=True,
            router_loaded=False,
            feature_count=12,
            load_events_fn=lambda path, limit=None: [{"sessionId": "sess-1", "type": "query_submitted"}],
            summary_fn=lambda events: {"total_events": len(events)},
            session_group_fn=lambda events: {"sess-1": events},
            render_fn=lambda summary, sessions, **kwargs: f"{summary['total_events']}|{len(sessions)}|{kwargs['feature_count']}",
        )

        self.assertEqual(html, "1|1|12")


if __name__ == "__main__":
    unittest.main()
