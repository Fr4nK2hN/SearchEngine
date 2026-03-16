import unittest

from tools.maintenance.precompute_embeddings import (
    build_embedding_work_items,
    build_update_actions,
    has_embedding,
)


class PrecomputeEmbeddingsTests(unittest.TestCase):
    def test_has_embedding_requires_non_empty_sequence(self):
        self.assertFalse(has_embedding({}, "content_emb"))
        self.assertFalse(has_embedding({"content_emb": []}, "content_emb"))
        self.assertTrue(has_embedding({"content_emb": [0.1, 0.2]}, "content_emb"))

    def test_build_embedding_work_items_skips_fully_precomputed_docs(self):
        hits = [
            {
                "_id": "doc-1",
                "_source": {
                    "title": "Alpha",
                    "content": "Alpha content",
                    "title_emb": [0.1],
                    "content_emb": [0.2],
                },
            },
            {
                "_id": "doc-2",
                "_source": {
                    "title": "Beta",
                    "content_full": "Beta full content",
                    "content_emb": [0.3],
                },
            },
        ]

        work_items = build_embedding_work_items(hits, overwrite=False)

        self.assertEqual(len(work_items), 1)
        self.assertEqual(work_items[0]["id"], "doc-2")
        self.assertTrue(work_items[0]["needs_title"])
        self.assertFalse(work_items[0]["needs_content"])
        self.assertEqual(work_items[0]["content"], "Beta full content")

    def test_build_update_actions_only_writes_missing_fields(self):
        work_items = [
            {
                "id": "doc-1",
                "title": "Alpha",
                "content": "Alpha content",
                "needs_title": True,
                "needs_content": True,
            },
            {
                "id": "doc-2",
                "title": "Beta",
                "content": "Beta content",
                "needs_title": False,
                "needs_content": True,
            },
        ]

        actions = build_update_actions(
            "documents",
            work_items,
            title_embeddings=[[1.0, 2.0]],
            content_embeddings=[[3.0, 4.0], [5.0, 6.0]],
        )

        self.assertEqual(len(actions), 2)
        self.assertEqual(actions[0]["doc"]["title_emb"], [1.0, 2.0])
        self.assertEqual(actions[0]["doc"]["content_emb"], [3.0, 4.0])
        self.assertNotIn("title_emb", actions[1]["doc"])
        self.assertEqual(actions[1]["doc"]["content_emb"], [5.0, 6.0])


if __name__ == "__main__":
    unittest.main()
