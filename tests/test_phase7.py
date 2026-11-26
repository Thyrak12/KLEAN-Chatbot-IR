"""Unit tests for Phase 7: RAG Pipeline and LLM Integration."""

import unittest
from unittest.mock import MagicMock

from src.rag.context_builder import ContextBuilder
from src.rag.llm_client import MockLLMClient, get_llm_client
from src.rag.prompt_templates import PromptManager
from src.rag.rag_pipeline import RAGPipeline


class TestPhase7RAGPipeline(unittest.TestCase):
    """Test suite for RAG Context Building, Prompting, and Pipeline."""

    def setUp(self):
        self.sample_records = {
            101: {
                "id": 101,
                "name": "Trattoria Roma",
                "category": "Restaurant",
                "sub_category": ["Italian", "Pizza", "Pasta"],
                "address": "Street 308, Tonle Bassac, Phnom Penh",
                "phone": "+855 23 123 456",
                "website": "https://roma.kh",
            },
            102: {
                "id": 102,
                "name": "Khmer Traditional Soup",
                "category": "Cambodian Restaurant",
                "sub_category": ["Khmer", "Soup"],
                "address": "Street 125, Daun Penh",
                "phone": None,
                "website": None,
            },
        }

    def test_context_builder_formatting(self):
        builder = ContextBuilder(records_by_id=self.sample_records)
        retrieval_results = [
            {"doc_id": 101, "score": 0.01639, "rank": 1, "bm25_rank": 1, "vector_rank": 2},
            {"doc_id": 102, "score": 0.01500, "rank": 2, "bm25_rank": 2, "vector_rank": 3},
        ]
        context = builder.build_context(retrieval_results, max_restaurants=2)

        self.assertIn("Trattoria Roma", context)
        self.assertIn("Street 308", context)
        self.assertIn("+855 23 123 456", context)
        self.assertIn("Khmer Traditional Soup", context)
        # Should exclude None fields gracefully
        self.assertNotIn("None", context)

    def test_context_builder_empty(self):
        builder = ContextBuilder(records_by_id=self.sample_records)
        context = builder.build_context([])
        self.assertEqual(context, "")

    def test_prompt_manager_with_context(self):
        pm = PromptManager()
        prompt = pm.build_prompt(
            query="romantic spot",
            context="[1] Trattoria Roma (ID: 101)",
        )
        self.assertIn("Trattoria Roma", prompt)
        self.assertIn("romantic spot", prompt)
        self.assertIn("CONTEXT INFORMATION:", prompt)

    def test_prompt_manager_no_context(self):
        pm = PromptManager()
        prompt = pm.build_prompt(query="alien cuisine", context="")
        self.assertIn("No matching restaurants were found", prompt)
        self.assertIn("alien cuisine", prompt)

    def test_mock_llm_client(self):
        client = MockLLMClient()
        prompt = "CONTEXT:\n[1] Trattoria Roma (ID: 101)\n\nUSER QUESTION: pizza?"
        answer = client.generate(prompt)
        self.assertIn("Trattoria Roma", answer)

        no_context_prompt = "No matching restaurants were found in the Phnom Penh database."
        empty_answer = client.generate(no_context_prompt)
        self.assertIn("couldn't find any restaurants", empty_answer.lower())

    def test_get_llm_client_factory(self):
        client = get_llm_client(provider="mock")
        self.assertIsInstance(client, MockLLMClient)

        # Fallback to mock when missing key
        fallback_client = get_llm_client(provider="openai", api_key="", fallback_to_mock=True)
        self.assertIsInstance(fallback_client, MockLLMClient)

    def test_rag_pipeline_end_to_end(self):
        mock_retriever = MagicMock()
        mock_retriever.search.return_value = [
            {"doc_id": 101, "score": 0.01639, "rank": 1, "bm25_rank": 1, "vector_rank": 2}
        ]

        context_builder = ContextBuilder(records_by_id=self.sample_records)
        mock_llm = MockLLMClient()

        pipeline = RAGPipeline(
            retriever=mock_retriever,
            context_builder=context_builder,
            llm_client=mock_llm,
            default_top_k=2,
            retrieval_name="TestHybrid",
        )

        res = pipeline.ask("Best italian food in town?")

        self.assertEqual(res["query"], "Best italian food in town?")
        self.assertIn("Trattoria Roma", res["answer"])
        self.assertEqual(res["num_retrieved"], 1)
        self.assertEqual(res["retrieved_restaurants"][0]["name"], "Trattoria Roma")
        self.assertEqual(res["retrieval_method"], "TestHybrid")
        self.assertGreater(res["latency_ms"], 0.0)

    def test_rag_pipeline_empty_query(self):
        mock_retriever = MagicMock()
        pipeline = RAGPipeline(retriever=mock_retriever)
        res = pipeline.ask("   ")
        self.assertEqual(res["num_retrieved"], 0)
        mock_retriever.search.assert_not_called()


if __name__ == "__main__":
    unittest.main()
