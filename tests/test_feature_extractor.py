import math
import unittest
from collections import OrderedDict

import numpy as np

from ranking.feature_extractor import FeatureExtractor


class FeatureExtractorTests(unittest.TestCase):
    class DummySemanticModel:
        def __init__(self):
            self.calls = []

        def encode(self, texts, batch_size=None, show_progress_bar=None):
            self.calls.append(texts)
            if isinstance(texts, str):
                return np.array([float(len(texts)), 1.0])
            return [np.array([float(len(text)), 1.0]) for text in texts]

    def build_extractor(self):
        extractor = FeatureExtractor.__new__(FeatureExtractor)
        extractor.stop_words = {"the", "is", "a", "of", "to", "in", "and"}
        extractor.idf_cache = {"alpha": 2.0, "beta": 1.0}
        extractor._default_idf = 3.0
        extractor._emb_cache_limit = 16
        extractor._emb_batch_size = 8
        extractor._emb_cache = OrderedDict()
        extractor._text_cache_limit = 8
        extractor._text_cache = OrderedDict()
        extractor.semantic_model = self.DummySemanticModel()
        return extractor

    def test_tokenize_and_text_analysis_are_cached(self):
        extractor = self.build_extractor()

        tokens = extractor._tokenize("Alpha, beta! alpha")
        analysis_a = extractor._analyze_text("Alpha beta. Gamma")
        analysis_b = extractor._analyze_text("Alpha beta. Gamma")

        self.assertEqual(tokens, ["alpha", "beta", "alpha"])
        self.assertIs(analysis_a, analysis_b)
        self.assertEqual(analysis_a["first_sentence_token_set"], {"alpha", "beta"})

    def test_extract_all_features_keeps_expected_values(self):
        extractor = self.build_extractor()
        document = {
            "title": "Alpha title",
            "content": "Alpha beta alpha. Gamma beta",
            "content_emb": [1.0, 0.0],
            "title_emb": [0.0, 1.0],
        }

        features = extractor.extract_all_features(
            "Alpha beta",
            document,
            es_score=4.0,
            query_emb=np.array([1.0, 0.0]),
        )

        self.assertEqual(features["exact_match_content"], 1.0)
        self.assertEqual(features["exact_match_title"], 0.0)
        self.assertEqual(features["term_coverage"], 1.0)
        self.assertEqual(features["covered_term_count"], 2)
        self.assertEqual(features["query_length"], 2)
        self.assertEqual(features["doc_length"], 5)
        self.assertAlmostEqual(features["title_match_ratio"], 0.5)
        self.assertAlmostEqual(features["first_sentence_match"], 1.0)
        self.assertAlmostEqual(features["semantic_sim_content"], 1.0)
        self.assertAlmostEqual(features["semantic_sim_title"], 0.0)
        self.assertAlmostEqual(features["title_content_consistency"], 0.0)
        self.assertAlmostEqual(features["avg_query_term_freq"], 2.0)
        self.assertAlmostEqual(features["bm25_title"], 4.0)
        self.assertAlmostEqual(features["es_log_score"], math.log(5.0))

    def test_cosine_similarity_handles_zero_vector(self):
        extractor = self.build_extractor()

        self.assertEqual(
            extractor._cosine_similarity(np.array([0.0, 0.0]), np.array([1.0, 0.0])),
            0.0,
        )

    def test_encode_many_texts_batches_and_reuses_cache(self):
        extractor = self.build_extractor()

        first = extractor._encode_many_texts(["alpha", "beta", "alpha"])
        second = extractor._encode_many_texts(["beta"])

        self.assertEqual(len(extractor.semantic_model.calls), 1)
        self.assertEqual(extractor.semantic_model.calls[0], ["alpha", "beta"])
        self.assertEqual(len(first), 3)
        self.assertTrue(np.array_equal(first[0], first[2]))
        self.assertTrue(np.array_equal(second[0], first[1]))

    def test_hydrate_document_embeddings_only_fills_missing_values(self):
        extractor = self.build_extractor()
        existing_title = np.array([9.0, 1.0])
        docs = [
            {"title": "Alpha", "content": "Alpha beta"},
            {"title": "Beta", "content": "Beta gamma", "title_emb": existing_title},
        ]

        extractor.hydrate_document_embeddings(docs)

        self.assertIn("content_emb", docs[0])
        self.assertIn("title_emb", docs[0])
        self.assertIn("content_emb", docs[1])
        self.assertTrue(np.array_equal(docs[1]["title_emb"], existing_title))
        self.assertEqual(len(extractor.semantic_model.calls), 2)


if __name__ == "__main__":
    unittest.main()
