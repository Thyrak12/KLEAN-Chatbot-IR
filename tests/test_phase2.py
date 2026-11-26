"""Unit tests for Phase 2: Text Preprocessing and Inverted Index."""

import tempfile
import unittest
from pathlib import Path

from src.preprocessing.text_preprocessing import TextPreprocessor
from src.indexing.inverted_index import InvertedIndex


class TestTextPreprocessor(unittest.TestCase):
    """Test suite for TextPreprocessor."""

    def setUp(self):
        self.preprocessor = TextPreprocessor(
            lowercase=True,
            remove_punctuation=True,
            remove_stopwords=True,
            enable_khmer=True,
        )

    def test_english_preprocessing(self):
        text = "The Best Italian Pizza & Pasta in Phnom Penh!"
        tokens = self.preprocessor.preprocess(text)
        self.assertIn("best", tokens)
        self.assertIn("italian", tokens)
        self.assertIn("pizza", tokens)
        self.assertIn("pasta", tokens)
        self.assertIn("phnom", tokens)
        self.assertIn("penh", tokens)
        # Check stopwords removed
        self.assertNotIn("the", tokens)
        self.assertNotIn("in", tokens)
        # Check punctuation removed
        self.assertNotIn("&", tokens)
        self.assertNotIn("!", tokens)

    def test_khmer_preprocessing(self):
        khmer_text = "ហាងមី ឆាយ អ៊ីម ផ្លូវភ្លោះអូឫស្សី"
        tokens = self.preprocessor.preprocess(khmer_text)
        self.assertGreaterEqual(len(tokens), 2)
        # Check that Khmer tokens are preserved and not stripped by English stopwords
        self.assertTrue(any(self.preprocessor.contains_khmer(t) for t in tokens))
        self.assertIn("មី", tokens)  # Noodle
        self.assertIn("ហាង", tokens)  # Shop

    def test_bilingual_preprocessing(self):
        bilingual_text = "Bai Sor បាយស | Traditional Khmer Food on St 240"
        tokens = self.preprocessor.preprocess(bilingual_text)
        self.assertIn("bai", tokens)
        self.assertIn("sor", tokens)
        self.assertIn("បាយ", tokens)  # Rice
        self.assertIn("ស", tokens)    # White
        self.assertIn("traditional", tokens)
        self.assertIn("khmer", tokens)
        self.assertIn("food", tokens)
        self.assertIn("240", tokens)
        # English stopword 'on' should be removed
        self.assertNotIn("on", tokens)

    def test_configurable_options(self):
        # Case where stopwords are kept
        no_stop_prep = TextPreprocessor(remove_stopwords=False)
        tokens = no_stop_prep.preprocess("The food is on the table")
        self.assertIn("the", tokens)
        self.assertIn("is", tokens)
        self.assertIn("on", tokens)

    def test_empty_and_none(self):
        self.assertEqual(self.preprocessor.preprocess(None), [])
        self.assertEqual(self.preprocessor.preprocess(""), [])
        self.assertEqual(self.preprocessor.preprocess("   "), [])


class TestInvertedIndex(unittest.TestCase):
    """Test suite for InvertedIndex."""

    def setUp(self):
        self.index = InvertedIndex()
        self.corpus = {
            1: ["khmer", "kitchen", "amok", "seafood", "phnom", "penh"],
            2: ["italian", "pizza", "pasta", "wine", "bkk1"],
            3: ["khmer", "noodles", "soup", "amok", "street", "food"],
            4: ["japanese", "sushi", "ramen", "bkk1"],
        }
        self.index.build(self.corpus)

    def test_vocabulary_and_stats(self):
        self.assertEqual(self.index.num_documents, 4)
        self.assertGreater(self.index.vocabulary_size, 10)
        self.assertIn("amok", self.index.vocabulary)
        self.assertIn("bkk1", self.index.vocabulary)
        self.assertGreater(self.index.avg_doc_length, 0)

    def test_term_lookup_and_postings(self):
        # 'amok' is in doc 1 and 3
        postings = self.index.get_postings("amok")
        self.assertEqual(postings, [1, 3])

        # 'bkk1' is in doc 2 and 4
        self.assertEqual(self.index.get_postings("bkk1"), [2, 4])

        # non-existent term
        self.assertEqual(self.index.get_postings("nonexistent"), [])

    def test_frequencies(self):
        # Document frequency
        self.assertEqual(self.index.get_document_frequency("amok"), 2)
        self.assertEqual(self.index.get_document_frequency("pizza"), 1)
        self.assertEqual(self.index.get_document_frequency("tacos"), 0)

        # Term frequency
        self.assertEqual(self.index.get_term_frequency("amok", 1), 1)
        self.assertEqual(self.index.get_term_frequency("pizza", 1), 0)

    def test_document_lookup(self):
        terms = self.index.get_document_terms(2)
        self.assertEqual(terms["pizza"], 1)
        self.assertEqual(terms["pasta"], 1)
        self.assertNotIn("amok", terms)

    def test_add_and_update_document(self):
        # Add new document 5
        self.index.add_document(5, ["mexican", "tacos", "burritos", "bkk1"])
        self.assertEqual(self.index.num_documents, 5)
        self.assertEqual(self.index.get_postings("tacos"), [5])
        self.assertEqual(self.index.get_postings("bkk1"), [2, 4, 5])

        # Update document 5 (remove tacos, add nachos)
        self.index.update_document(5, ["mexican", "nachos"])
        self.assertEqual(self.index.get_postings("tacos"), [])
        self.assertEqual(self.index.get_postings("nachos"), [5])

    def test_remove_document(self):
        # Remove document 2
        removed = self.index.remove_document(2)
        self.assertTrue(removed)
        self.assertEqual(self.index.num_documents, 3)
        self.assertEqual(self.index.get_postings("pizza"), [])
        self.assertEqual(self.index.get_postings("bkk1"), [4])

    def test_serialization(self):
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            temp_path = Path(f.name)

        try:
            self.index.save(temp_path)
            loaded_index = InvertedIndex.load(temp_path)

            self.assertEqual(loaded_index.num_documents, self.index.num_documents)
            self.assertEqual(loaded_index.vocabulary_size, self.index.vocabulary_size)
            self.assertEqual(loaded_index.get_postings("amok"), [1, 3])
            self.assertEqual(loaded_index.get_term_frequency("amok", 1), 1)
        finally:
            if temp_path.exists():
                temp_path.unlink()


if __name__ == "__main__":
    unittest.main()
