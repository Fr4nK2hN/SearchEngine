import unittest

from tools.analysis.analyze_embedding_coverage import compute_coverage_stats, format_ratio


class AnalyzeEmbeddingCoverageTests(unittest.TestCase):
    def test_format_ratio_handles_zero_total(self):
        self.assertEqual(format_ratio(0, 0), "0.00%")
        self.assertEqual(format_ratio(5, 10), "50.00%")

    def test_compute_coverage_stats_derives_missing_counts(self):
        stats = compute_coverage_stats(100, 70, 60, 55)

        self.assertEqual(stats["missing_content_emb_docs"], 30)
        self.assertEqual(stats["missing_title_emb_docs"], 40)
        self.assertEqual(stats["missing_any_emb_docs"], 45)
        self.assertEqual(stats["content_emb_ratio"], "70.00%")
        self.assertEqual(stats["both_emb_ratio"], "55.00%")


if __name__ == "__main__":
    unittest.main()
