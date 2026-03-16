import os
import unittest
from unittest import mock

from webapp.config import load_config


class ConfigTests(unittest.TestCase):
    def test_service_defaults_match_current_runtime(self):
        with mock.patch.dict(os.environ, {}, clear=True):
            config = load_config()

        self.assertEqual(config.es_host, "elasticsearch")
        self.assertEqual(config.es_port, 9200)
        self.assertEqual(config.es_scheme, "http")
        self.assertEqual(config.adaptive_easy_mode, "baseline")
        self.assertEqual(config.adaptive_hard_mode, "cross_encoder")
        self.assertEqual(config.adaptive_hard_threshold, 0.6062)
        self.assertEqual(config.adaptive_hard_top_k_cap, 5)
        self.assertEqual(config.recall_relax_threshold, 5)
        self.assertEqual(config.log_file, os.path.join("logs", "events.log"))

    def test_env_overrides_are_normalized(self):
        env = {
            "SEARCHENGINE_ES_HOST": "localhost",
            "SEARCHENGINE_ES_PORT": "9201",
            "SEARCHENGINE_ES_SCHEME": "https",
            "ADAPTIVE_EASY_MODE": "ltr",
            "ADAPTIVE_HARD_MODE": "hybrid",
            "ADAPTIVE_HARD_THRESHOLD": "0.45",
            "ADAPTIVE_HARD_TOP_K_CAP": "30",
            "RECALL_RELAX_THRESHOLD": "7",
            "SEARCHENGINE_LOG_DIR": "tmp_logs",
            "SEARCHENGINE_LOG_FILENAME": "runtime.log",
        }
        with mock.patch.dict(os.environ, env, clear=True):
            config = load_config()

        self.assertEqual(config.es_host, "localhost")
        self.assertEqual(config.es_port, 9201)
        self.assertEqual(config.es_scheme, "https")
        self.assertEqual(config.adaptive_easy_mode, "ltr")
        self.assertEqual(config.adaptive_hard_mode, "hybrid")
        self.assertEqual(config.adaptive_hard_threshold, 0.45)
        self.assertEqual(config.adaptive_hard_top_k_cap, 30)
        self.assertEqual(config.recall_relax_threshold, 7)
        self.assertEqual(config.log_file, os.path.join("tmp_logs", "runtime.log"))


if __name__ == "__main__":
    unittest.main()
