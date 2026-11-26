"""End-to-end Retrieval-Augmented Generation (RAG) Pipeline for KLEAN Chatbot.

Orchestrates:
  1. Retrieve: Fetches top restaurant candidates via Hybrid / BM25 / Vector search.
  2. Augment: Constructs structured context with strict citation requirements.
  3. Generate: Calls LLM to produce grounded, conversational recommendations.
"""

import logging
import time
from typing import Any, Dict, Iterator, List, Optional, Tuple, Union

from src.rag.context_builder import ContextBuilder
from src.rag.llm_client import BaseLLMClient, MockLLMClient, get_llm_client
from src.rag.prompt_templates import PromptManager

logger = logging.getLogger(__name__)


class RAGPipeline:
    """End-to-end RAG pipeline for Phnom Penh restaurant discovery."""

    def __init__(
        self,
        retriever: Any,
        context_builder: Optional[ContextBuilder] = None,
        llm_client: Optional[BaseLLMClient] = None,
        prompt_manager: Optional[PromptManager] = None,
        default_top_k: int = 3,
        retrieval_name: str = "Hybrid RRF",
    ):
        """Initialize RAG Pipeline.

        Args:
            retriever: Search retriever instance (HybridRetriever, BM25, etc.).
            context_builder: ContextBuilder instance.
            llm_client: BaseLLMClient instance. Defaults to get_llm_client().
            prompt_manager: PromptManager instance.
            default_top_k: Default number of top restaurant candidates to pass to LLM.
            retrieval_name: Identifier for the retrieval algorithm.
        """
        self.retriever = retriever
        self.context_builder = context_builder or ContextBuilder()
        self.llm_client = llm_client or get_llm_client()
        self.prompt_manager = prompt_manager or PromptManager()
        self.default_top_k = default_top_k
        self.retrieval_name = retrieval_name

    def ask(
        self,
        query: str,
        top_k: Optional[int] = None,
        temperature: float = 0.3,
        max_tokens: int = 1000,
    ) -> Dict[str, Any]:
        """Process a user question end-to-end and generate a grounded answer.

        Args:
            query: Natural language user query.
            top_k: Number of candidate restaurants to retrieve and inject.
            temperature: LLM sampling temperature.
            max_tokens: Maximum response tokens.

        Returns:
            Dict containing:
              - 'query': Original user query
              - 'answer': Generated response text
              - 'context': The context block passed to the LLM
              - 'retrieved_restaurants': List of retrieved candidates with metadata
              - 'num_retrieved': Count of retrieved restaurants
              - 'latency_ms': Total end-to-end execution time in milliseconds
              - 'retrieval_method': Name of retrieval engine
              - 'llm_provider': Class name of active LLM client
        """
        clean_query = query.strip()
        t0 = time.perf_counter()

        if not clean_query:
            return {
                "query": query,
                "answer": "Please ask a question or specify the type of restaurant, cuisine, or neighborhood you are looking for!",
                "context": "",
                "retrieved_restaurants": [],
                "num_retrieved": 0,
                "latency_ms": 0.0,
                "retrieval_method": self.retrieval_name,
                "llm_provider": self.llm_client.__class__.__name__,
            }

        k = top_k or self.default_top_k

        # 1. RETRIEVE: Fetch ranked candidates
        t_ret_start = time.perf_counter()
        raw_candidates = self.retriever.search(clean_query, top_k=k)
        retrieval_time_ms = (time.perf_counter() - t_ret_start) * 1000.0

        # Attach full records to candidates
        enriched_candidates = []
        for item in raw_candidates:
            doc_id = item["doc_id"]
            rec = self.context_builder.records_by_id.get(doc_id, {})
            enriched_item = dict(item)
            enriched_item["name"] = rec.get("name", "Unknown")
            enriched_item["category"] = rec.get("category", "")
            enriched_item["sub_category"] = rec.get("sub_category", [])
            enriched_item["address"] = rec.get("address", "")
            enriched_item["phone"] = rec.get("phone", "")
            enriched_item["website"] = rec.get("website", "")
            enriched_candidates.append(enriched_item)

        # 2. AUGMENT: Construct context block and grounded prompt
        context_str = self.context_builder.build_context(raw_candidates, max_restaurants=k)
        prompt = self.prompt_manager.build_prompt(query=clean_query, context=context_str)

        # 3. GENERATE: Query LLM
        t_llm_start = time.perf_counter()
        answer = self.llm_client.generate(
            prompt=prompt,
            system_prompt=self.prompt_manager.system_prompt,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        llm_time_ms = (time.perf_counter() - t_llm_start) * 1000.0

        total_latency_ms = (time.perf_counter() - t0) * 1000.0

        return {
            "query": clean_query,
            "answer": answer,
            "context": context_str,
            "prompt": prompt,
            "retrieved_restaurants": enriched_candidates,
            "num_retrieved": len(enriched_candidates),
            "retrieval_latency_ms": retrieval_time_ms,
            "llm_latency_ms": llm_time_ms,
            "latency_ms": total_latency_ms,
            "retrieval_method": self.retrieval_name,
            "llm_provider": self.llm_client.__class__.__name__,
        }

    def stream_ask(
        self,
        query: str,
        top_k: Optional[int] = None,
        temperature: float = 0.3,
        max_tokens: int = 1000,
    ) -> Tuple[List[Dict[str, Any]], Iterator[str], Dict[str, Any]]:
        """Stream a grounded RAG response token by token.

        Returns:
            Tuple of:
              - List of enriched candidate restaurant dicts (with lat/lon, address, phone)
              - Token generator yielding string chunks
              - Metadata dict (query, retrieval_method, retrieval_latency_ms, etc.)
        """
        clean_query = query.strip()
        t0 = time.perf_counter()
        k = top_k or self.default_top_k

        # 1. RETRIEVE: Fetch ranked candidates using configured retriever (Hybrid RRF by default)
        raw_candidates = self.retriever.search(clean_query, top_k=k)
        retrieval_time_ms = (time.perf_counter() - t0) * 1000.0

        enriched_candidates = []
        for item in raw_candidates:
            doc_id = item["doc_id"]
            rec = self.context_builder.records_by_id.get(doc_id, {})
            enriched_item = dict(item)
            enriched_item["name"] = rec.get("name", "Unknown")
            enriched_item["category"] = rec.get("category", "")
            enriched_item["sub_category"] = rec.get("sub_category", [])
            enriched_item["address"] = rec.get("address", "")
            enriched_item["phone"] = rec.get("phone", "")
            enriched_item["website"] = rec.get("website", "")
            enriched_item["latitude"] = rec.get("latitude")
            enriched_item["longitude"] = rec.get("longitude")
            enriched_candidates.append(enriched_item)

        # 2. AUGMENT: Construct context block and grounded prompt
        context_str = self.context_builder.build_context(raw_candidates, max_restaurants=k)
        prompt = self.prompt_manager.build_prompt(query=clean_query, context=context_str)

        # 3. GENERATE STREAM: Token generator
        stream_gen = self.llm_client.stream_generate(
            prompt=prompt,
            system_prompt=self.prompt_manager.system_prompt,
            temperature=temperature,
            max_tokens=max_tokens,
        )

        meta = {
            "query": clean_query,
            "retrieval_method": self.retrieval_name,
            "llm_provider": self.llm_client.__class__.__name__,
            "retrieval_latency_ms": retrieval_time_ms,
            "context": context_str,
            "prompt": prompt,
        }

        return enriched_candidates, stream_gen, meta
