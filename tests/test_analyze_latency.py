import unittest

from tools.analysis.analyze_latency import collect_groups, ratio_text, summarize_embedding_metrics


class AnalyzeLatencyTests(unittest.TestCase):
    def test_ratio_text_handles_zero_denominator(self):
        self.assertEqual(ratio_text(0, 0), "N/A")
        self.assertEqual(ratio_text(3, 4), "75.00%")

    def test_collect_groups_reads_embedding_metrics(self):
        lines = [
            '{"message":"Search successful","rankingMethod":"LTR","total_ms":100,"retrieval_ms":5,"feature_ms":90,"inference_ms":5,"candidate_docs":50,"precomputed_content_emb_count":40,"encoded_content_emb_count":10,"precomputed_title_emb_count":35,"encoded_title_emb_count":10}',
            '{"message":"Search successful","rankingMethod":"Baseline (ES only)","total_ms":8,"retrieval_ms":8,"feature_ms":0,"inference_ms":0}',
        ]

        groups = collect_groups(lines)

        self.assertEqual(groups["LTR"]["candidate_docs"], [50])
        self.assertEqual(groups["LTR"]["encoded_content_emb_count"], [10])
        self.assertEqual(groups["Baseline (ES only)"]["candidate_docs"], [])

    def test_summarize_embedding_metrics_computes_hit_ratios(self):
        summary = summarize_embedding_metrics(
            {
                "candidate_docs": [50, 40],
                "precomputed_content_emb_count": [40, 30],
                "precomputed_title_emb_count": [35, 20],
                "encoded_content_emb_count": [10, 10],
                "encoded_title_emb_count": [10, 5],
            }
        )

        self.assertTrue(summary["has_embedding_metrics"])
        self.assertEqual(summary["content_hit_ratio"], "77.78%")
        self.assertEqual(summary["title_hit_ratio"], "78.57%")
        self.assertAlmostEqual(summary["avg_candidate_docs"], 45.0)
        self.assertAlmostEqual(summary["avg_online_content_fill"], 10.0)


if __name__ == "__main__":
    unittest.main()
