"""Unit tests for Phase 4: Embedding Generation and FAISS Vector Search."""

from pathlib import Path
import tempfile
import unittest

import numpy as np

from src.embeddings.embedding_model import EmbeddingModel
from src.embeddings.embedding_generator import EmbeddingGenerator
from src.embeddings.vector_store import VectorStore
from src.retrieval.vector_search import VectorSearcher


class TestPhase4EmbeddingsAndVectorSearch(unittest.TestCase):
    """Test suite for Phase 4."""

    @classmethod
    def setUpClass(cls):
        # Initialize model once for test suite
        cls.model = EmbeddingModel()
        cls.corpus = {
            1: "Name: Trattoria Roma | Category: Restaurant | Sub Category: Italian, Pizza, Pasta",
            2: "Name: Phnom Penh Noodle House | Category: Restaurant | Sub Category: Khmer, Soup, Noodles",
            3: "Name: Sakura Sushi | Category: Restaurant | Sub Category: Japanese, Sushi, Ramen",
        }

    def test_embedding_model_properties(self):
        self.assertEqual(self.model.get_dimension(), 384)
        queries = ["query: italian pizza"]
        embeddings = self.model.encode_queries(queries, normalize_embeddings=True)
        self.assertEqual(embeddings.shape, (1, 384))
        # Verify L2 normalization: norm should be approximately 1.0
        norm = np.linalg.norm(embeddings[0])
        self.assertAlmostEqual(norm, 1.0, places=4)

    def test_embedding_generator_caching(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            generator = EmbeddingGenerator(embedding_model=self.model, vector_dir=tmpdir)
            embeddings1, doc_ids1 = generator.generate_or_load(self.corpus)

            self.assertEqual(len(embeddings1), 3)
            self.assertEqual(doc_ids1, [1, 2, 3])
            self.assertTrue((Path(tmpdir) / "embeddings.npy").exists())
            self.assertTrue((Path(tmpdir) / "embedding_metadata.json").exists())

            # Second call should load from cache without recomputing
            embeddings2, doc_ids2 = generator.generate_or_load(self.corpus)
            np.testing.assert_array_almost_equal(embeddings1, embeddings2)
            self.assertEqual(doc_ids1, doc_ids2)

    def test_vector_store_operations(self):
        vectors = np.array([
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ], dtype=np.float32)
        doc_ids = [101, 102, 103]

        store = VectorStore(dimension=3)
        store.add(vectors, doc_ids)
        self.assertEqual(store.total_vectors, 3)

        # Search with vector identical to doc 102
        query_vec = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        results = store.search(query_vec, top_k=2)

        self.assertEqual(len(results), 2)
        top_id, top_score = results[0]
        self.assertEqual(top_id, 102)
        self.assertAlmostEqual(top_score, 1.0, places=4)

    def test_vector_store_persistence(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            idx_file = Path(tmpdir) / "test_faiss.bin"
            meta_file = Path(tmpdir) / "test_meta.json"

            vectors = np.array([[0.6, 0.8]], dtype=np.float32)
            store = VectorStore(dimension=2)
            store.add(vectors, [99])
            store.save(idx_file, meta_file)

            loaded_store = VectorStore.load(idx_file, meta_file)
            self.assertEqual(loaded_store.total_vectors, 1)
            self.assertEqual(loaded_store.idx_to_doc_id, [99])

            query = np.array([0.6, 0.8], dtype=np.float32)
            res = loaded_store.search(query, top_k=1)
            self.assertEqual(res[0][0], 99)
            self.assertAlmostEqual(res[0][1], 1.0, places=4)

    def test_vector_searcher_semantic_ranking(self):
        # Generate embeddings for our 3 test restaurants
        passages = [self.corpus[i] for i in [1, 2, 3]]
        embeddings = self.model.encode_passages(passages, normalize_embeddings=True)

        store = VectorStore(dimension=self.model.get_dimension())
        store.add(embeddings, [1, 2, 3])

        searcher = VectorSearcher(vector_store=store, embedding_model=self.model)

        # Query 1: Italian food -> should rank doc 1 first
        results_italian = searcher.search("delicious pasta and pizza dining", top_k=3)
        self.assertEqual(results_italian[0]["doc_id"], 1)

        # Query 2: Japanese food -> should rank doc 3 first
        results_japanese = searcher.search("fresh sushi sashimi and ramen", top_k=3)
        self.assertEqual(results_japanese[0]["doc_id"], 3)


if __name__ == "__main__":
    unittest.main()
