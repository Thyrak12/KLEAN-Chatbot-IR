"""Unit tests for Phase 3: Boolean Retrieval, TF-IDF, and BM25."""

import unittest

from src.indexing.inverted_index import InvertedIndex
from src.preprocessing.text_preprocessing import TextPreprocessor
from src.retrieval.boolean_retrieval import BooleanRetriever
from src.retrieval.tfidf import TFIDFRetriever
from src.retrieval.bm25 import BM25Retriever


class TestPhase3Retrieval(unittest.TestCase):
    """Test suite for Phase 3 Retrieval algorithms."""

    def setUp(self):
        self.preprocessor = TextPreprocessor(lowercase=True, remove_stopwords=False)
        self.index = InvertedIndex()
        # Controlled corpus
        self.raw_corpus = {
            1: "Khmer Traditional Kitchen Seafood Amok Phnom Penh",
            2: "Italian Pizza Pasta Wine Bar BKK1",
            3: "Khmer Noodles Soup Amok Street Food",
            4: "Japanese Sushi Ramen Izakaya BKK1",
            5: "American Burgers BBQ Craft Beer BKK1",
        }
        self.tokenized_corpus = {
            doc_id: self.preprocessor.preprocess(text)
            for doc_id, text in self.raw_corpus.items()
        }
        self.index.build(self.tokenized_corpus)

        self.bool_retriever = BooleanRetriever(self.index, self.preprocessor)
        self.tfidf_retriever = TFIDFRetriever(self.index, self.preprocessor)
        self.bm25_retriever = BM25Retriever(self.index, self.preprocessor, k1=1.5, b=0.75)

    # ---------------------------------------------------------
    # Boolean Retrieval Tests
    # ---------------------------------------------------------
    def test_boolean_and(self):
        # Khmer AND Amok -> doc 1 and 3
        res = self.bool_retriever.search("khmer AND amok")
        self.assertEqual(res, [1, 3])

        # Italian AND Pizza -> doc 2
        res2 = self.bool_retriever.search("italian AND pizza")
        self.assertEqual(res2, [2])

    def test_boolean_or(self):
        # Pizza OR Ramen -> doc 2 and 4
        res = self.bool_retriever.search("pizza OR ramen")
        self.assertEqual(res, [2, 4])

    def test_boolean_not(self):
        # Khmer AND NOT Seafood -> only doc 3 (doc 1 has seafood)
        res = self.bool_retriever.search("khmer AND NOT seafood")
        self.assertEqual(res, [3])

    def test_boolean_nested_expression(self):
        # bkk1 AND (pizza OR sushi) -> doc 2 and 4
        res = self.bool_retriever.search("bkk1 AND (pizza OR sushi)")
        self.assertEqual(res, [2, 4])

    def test_boolean_implicit_and(self):
        # 'italian pasta' without AND keyword
        res = self.bool_retriever.search("italian pasta")
        self.assertEqual(res, [2])

    # ---------------------------------------------------------
    # TF-IDF Retrieval Tests
    # ---------------------------------------------------------
    def test_tfidf_ranking(self):
        results = self.tfidf_retriever.search("italian pizza", top_k=3)
        self.assertGreater(len(results), 0)
        # Top document must be doc 2
        self.assertEqual(results[0]["doc_id"], 2)
        self.assertGreater(results[0]["score"], 0.0)

    def test_tfidf_rare_term_weighting(self):
        # 'ramen' is in only 1 doc, 'bkk1' is in 3 docs.
        # Query 'ramen bkk1' should place doc 4 at rank 1.
        results = self.tfidf_retriever.search("ramen bkk1", top_k=3)
        self.assertEqual(results[0]["doc_id"], 4)

    def test_tfidf_sklearn_comparison(self):
        comparison = self.tfidf_retriever.compare_with_sklearn(
            self.raw_corpus, query="italian pizza", top_k=3
        )
        self.assertIn("custom_rankings", comparison)
        self.assertIn("sklearn_rankings", comparison)
        # Both should rank doc 2 first
        self.assertEqual(comparison["custom_rankings"][0]["doc_id"], 2)
        self.assertEqual(comparison["sklearn_rankings"][0]["doc_id"], 2)
        self.assertGreater(comparison["top_k_overlap_percentage"], 50.0)

    # ---------------------------------------------------------
    # BM25 Retrieval Tests
    # ---------------------------------------------------------
    def test_bm25_ranking(self):
        results = self.bm25_retriever.search("khmer noodles soup", top_k=3)
        self.assertGreater(len(results), 0)
        # Doc 3 has both 'noodles' and 'soup' and 'khmer'
        self.assertEqual(results[0]["doc_id"], 3)
        self.assertGreater(results[0]["score"], 0.0)

    def test_bm25_score_explanation(self):
        explanation = self.bm25_retriever.explain_score("noodles soup", doc_id=3)
        self.assertEqual(explanation["doc_id"], 3)
        self.assertIn("terms", explanation)
        self.assertGreater(explanation["total_bm25_score"], 0.0)
        self.assertEqual(len(explanation["terms"]), 2)

    def test_bm25_empty_query(self):
        self.assertEqual(self.bm25_retriever.search(""), [])
        self.assertEqual(self.bm25_retriever.search("nonexistenttermxyz"), [])


if __name__ == "__main__":
    unittest.main()
