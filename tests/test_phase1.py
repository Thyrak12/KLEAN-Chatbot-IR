"""Unit tests for Phase 1: Data Loading & Document Conversion."""

import json
from pathlib import Path
import tempfile
import unittest

from src.utils.data_loader import REQUIRED_FIELDS, DataLoader
from src.utils.document_converter import DocumentConverter, SearchDocument


class TestDataLoader(unittest.TestCase):
    """Test suite for DataLoader."""

    def setUp(self):
        self.loader = DataLoader()

    def test_load_dataset(self):
        records = self.loader.load()
        self.assertGreaterEqual(len(records), 5, "Dataset should have at least 5 records")

    def test_dataset_schema_compliance(self):
        records = self.loader.load()
        for rec in records:
            # Check exact keys presence
            self.assertEqual(
                REQUIRED_FIELDS,
                set(rec.keys()),
                f"Record ID {rec.get('id')} has mismatched schema keys"
            )
            # Check social media keys
            social = rec["social_media"]
            self.assertIsInstance(social, dict)
            self.assertEqual(set(social.keys()), {"facebook", "instagram", "tiktok"})
            # Check coordinates types
            self.assertIsInstance(rec["latitude"], (int, float))
            self.assertIsInstance(rec["longitude"], (int, float))
            self.assertIsInstance(rec["sub_category"], list)

    def test_immutability_preservation(self):
        records = self.loader.load()
        first_id = records[0]["id"]
        original_name = records[0]["name"]

        # Mutate returned record
        records[0]["name"] = "MODIFIED NAME"

        # Re-fetch from loader
        re_fetched = self.loader.get_by_id(first_id)
        self.assertEqual(re_fetched["name"], original_name, "Internal record must remain unmodified")

    def test_structured_filtering(self):
        records = self.loader.load()
        # Filter by sub-category
        italian_recs = self.loader.filter_by(sub_category="Italian")
        self.assertTrue(all("Italian" in r["sub_category"] for r in italian_recs))

        # Filter by district keyword
        bkk1_recs = self.loader.filter_by(district_keyword="BKK1")
        self.assertTrue(all("bkk1" in r["address"].lower() for r in bkk1_recs))

    def test_to_dataframe(self):
        df = self.loader.to_dataframe()
        self.assertEqual(len(df), len(self.loader.records))
        self.assertIn("name", df.columns)
        self.assertIn("category", df.columns)


class TestDocumentConverter(unittest.TestCase):
    """Test suite for DocumentConverter."""

    def setUp(self):
        self.loader = DataLoader()
        self.records = self.loader.load()
        self.converter = DocumentConverter()

    def test_conversion_structure(self):
        docs = self.converter.convert_all(self.records)
        self.assertEqual(len(docs), len(self.records))

        for doc in docs:
            self.assertIsInstance(doc, SearchDocument)
            self.assertIsInstance(doc.doc_id, int)
            self.assertIsInstance(doc.text, str)
            self.assertGreater(len(doc.text), 10)
            self.assertIn("Name:", doc.text)
            self.assertIn("Category:", doc.text)

    def test_null_value_exclusion(self):
        # Mock record with null postal_code
        mock_rec = {
            "id": 999,
            "name": "Test Null Cafe",
            "category": "Cafe",
            "sub_category": ["Coffee"],
            "address": "Street 100",
            "postal_code": None,
            "website": None,
            "phone": None,
            "email": None,
            "social_media": {"facebook": None, "instagram": None, "tiktok": None},
            "latitude": 11.5,
            "longitude": 104.9,
        }
        doc = self.converter.convert_record(mock_rec)
        self.assertNotIn("Postal Code", doc.text, "Null postal code should be excluded from generated text")
        self.assertNotIn("Website", doc.text, "Null website should be excluded from text")

    def test_mapping_retrieval(self):
        docs = self.converter.convert_all(self.records)
        for doc in docs:
            rec = self.converter.get_record(doc.doc_id)
            self.assertIsNotNone(rec)
            self.assertEqual(rec["id"], doc.doc_id)

            fetched_doc = self.converter.get_document(doc.doc_id)
            self.assertEqual(fetched_doc.text, doc.text)

    def test_corpus_generation(self):
        docs = self.converter.convert_all(self.records)
        corpus = self.converter.get_corpus()
        self.assertEqual(len(corpus), len(self.records))
        self.assertEqual(set(corpus.keys()), {r["id"] for r in self.records})


if __name__ == "__main__":
    unittest.main()
