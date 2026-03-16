import unittest
from unittest import mock

from webapp.services.events import build_click_log_entry, log_client_events


class EventServiceTests(unittest.TestCase):
    def test_log_client_events_rejects_empty_payload(self):
        payload, status_code = log_client_events(mock.Mock(), None)

        self.assertEqual(status_code, 400)
        self.assertEqual(payload["message"], "No data provided")

    def test_log_client_events_logs_only_dict_items(self):
        logger = mock.Mock()
        payload, status_code = log_client_events(logger, [{"type": "a"}, "skip-me"])

        self.assertEqual(status_code, 200)
        self.assertEqual(payload, {"status": "success"})
        logger.info.assert_called_once_with("client_event", extra={"type": "a"})

    def test_build_click_log_entry_preserves_existing_keys(self):
        entry = build_click_log_entry({
            "session_id": "sess-1",
            "search_id": "search-1",
            "doc_id": "doc-1",
            "rank": 2,
            "query": "alpha",
        })

        self.assertEqual(entry["event"], "result_click_confirmed")
        self.assertEqual(entry["sessionId"], "sess-1")
        self.assertEqual(entry["search_id"], "search-1")


if __name__ == "__main__":
    unittest.main()
