import unittest
from unittest import mock

from webapp.services.documents import fetch_documents_by_ids, normalize_doc_ids


class DocumentServiceTests(unittest.TestCase):
    def test_normalize_doc_ids_drops_empty_values(self):
        self.assertEqual(normalize_doc_ids([" doc-1 ", "", None]), ["doc-1"])

    def test_fetch_documents_by_ids_preserves_request_order(self):
        es = mock.Mock()
        es.mget.return_value = {
            "docs": [
                {"_id": "doc-2", "found": True, "_source": {"title": "B"}},
                {"_id": "doc-1", "found": True, "_source": {"title": "A"}},
                {"_id": "doc-3", "found": False, "_source": {"title": "C"}},
            ]
        }

        documents = fetch_documents_by_ids(es, "documents", ["doc-1", "doc-2", "doc-3"])

        self.assertEqual([doc["_id"] for doc in documents], ["doc-1", "doc-2"])


if __name__ == "__main__":
    unittest.main()
