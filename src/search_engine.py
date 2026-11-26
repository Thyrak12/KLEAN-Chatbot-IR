"""Unified Search Engine Facade for KLEAN Chatbot.

Provides a unified single entry point to all Information Retrieval algorithms
(Boolean, TF-IDF, BM25, FAISS Vector Search, Hybrid RRF) and RAG generation,
with structured metadata pre/post-filtering.
"""

import logging
from pathlib import Path
import time
from typing import Any, Dict, Iterator, List, Optional, Set, Tuple, Union

from src.embeddings.embedding_model import EmbeddingModel
from src.embeddings.vector_store import VectorStore
from src.indexing.inverted_index import InvertedIndex
from src.preprocessing.text_preprocessing import TextPreprocessor
from src.rag.context_builder import ContextBuilder
from src.rag.llm_client import BaseLLMClient, get_llm_client
from src.rag.prompt_templates import PromptManager
from src.rag.rag_pipeline import RAGPipeline
from src.retrieval.bm25 import BM25Retriever
from src.retrieval.boolean_retrieval import BooleanRetriever
from src.retrieval.hybrid import HybridRetriever
from src.retrieval.tfidf import TFIDFRetriever
from src.retrieval.vector_search import VectorSearcher
from src.utils.data_loader import DataLoader
from src.utils.document_converter import DocumentConverter

logger = logging.getLogger(__name__)

DEFAULT_DATASET_PATH = Path(__file__).resolve().parent.parent / "dataset" / "restaurants.json"
DEFAULT_VECTOR_DIR = Path(__file__).resolve().parent.parent / "vector_db"


class SearchEngine:
    """Unified search and recommendation engine managing all IR algorithms and RAG."""

    def __init__(
        self,
        dataset_path: Optional[Union[str, Path]] = None,
        vector_dir: Optional[Union[str, Path]] = None,
        llm_client: Optional[BaseLLMClient] = None,
        load_vector_db: bool = True,
    ):
        """Initialize SearchEngine.

        Args:
            dataset_path: Path to restaurants.json.
            vector_dir: Path to directory containing faiss_index.bin and metadata.
            llm_client: Optional pre-configured LLM client.
            load_vector_db: Whether to load FAISS vector index.
        """
        self.dataset_path = Path(dataset_path) if dataset_path else DEFAULT_DATASET_PATH
        self.vector_dir = Path(vector_dir) if vector_dir else DEFAULT_VECTOR_DIR

        logger.info("Initializing SearchEngine with dataset: %s", self.dataset_path)

        # 1. Load Data
        self.loader = DataLoader(self.dataset_path)
        self.records = self.loader.load()
        self.records_by_id = {r["id"]: r for r in self.records}

        # 2. Document Conversion
        self.converter = DocumentConverter()
        self.documents = self.converter.convert_all(self.records)
        self.corpus = self.converter.get_corpus()

        # 3. Text Preprocessing & Inverted Index
        self.preprocessor = TextPreprocessor()
        self.tokenized_corpus = self.preprocessor.preprocess_documents(self.corpus)
        self.index = InvertedIndex()
        self.index.build(self.tokenized_corpus)

        # 4. Lexical Retrievers
        self.boolean_retriever = BooleanRetriever(self.index, self.preprocessor)
        self.tfidf_retriever = TFIDFRetriever(self.index, self.preprocessor)
        self.bm25_retriever = BM25Retriever(self.index, self.preprocessor)

        # 5. Dense Vector & Hybrid Retrievers
        self.vector_store: Optional[VectorStore] = None
        self.vector_searcher: Optional[VectorSearcher] = None
        self.hybrid_retriever: Optional[HybridRetriever] = None

        if load_vector_db:
            idx_file = self.vector_dir / "faiss_index.bin"
            meta_file = self.vector_dir / "index_metadata.json"
            if idx_file.exists() and meta_file.exists():
                try:
                    self.vector_store = VectorStore.load(idx_file, meta_file)
                    self.embedding_model = EmbeddingModel()
                    self.vector_searcher = VectorSearcher(
                        vector_store=self.vector_store,
                        embedding_model=self.embedding_model,
                    )
                    self.hybrid_retriever = HybridRetriever(
                        bm25_retriever=self.bm25_retriever,
                        vector_searcher=self.vector_searcher,
                        bm25_weight=0.5,
                        vector_weight=0.5,
                        rrf_k=60,
                    )
                    logger.info("Vector store and Hybrid retriever loaded successfully.")
                except Exception as e:
                    logger.warning("Failed to load vector database: %s", e)

        # 6. RAG Pipeline
        self.context_builder = ContextBuilder(records_by_id=self.records_by_id)
        self.prompt_manager = PromptManager()
        self.llm_client = llm_client or get_llm_client(fallback_to_mock=True)

        if self.hybrid_retriever:
            rag_retriever = self.hybrid_retriever
            ret_name = "Hybrid RRF"
        else:
            rag_retriever = self.bm25_retriever
            ret_name = "BM25"

        self.rag_pipeline = RAGPipeline(
            retriever=rag_retriever,
            context_builder=self.context_builder,
            llm_client=self.llm_client,
            prompt_manager=self.prompt_manager,
            default_top_k=3,
            retrieval_name=ret_name,
        )

        logger.info("SearchEngine fully initialized and ready.")

    def search(
        self,
        query: str,
        algorithm: str = "hybrid",
        top_k: int = 10,
        category: Optional[str] = None,
        district: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Unified search across all Information Retrieval algorithms with metadata filtering.

        Args:
            query: Search query string.
            algorithm: 'hybrid', 'bm25', 'tfidf', 'vector', or 'boolean'.
            top_k: Number of ranked results to return.
            category: Optional category filter (e.g. 'Restaurant', 'Coffee Shop').
            district: Optional district/neighborhood keyword filter (e.g. 'Toul Kork', 'BKK1').

        Returns:
            List of ranked result dicts enriched with full restaurant metadata.
        """
        clean_query = query.strip()
        if not clean_query:
            return []

        algo = algorithm.lower().strip()

        # If filtering is requested, we retrieve a larger pool then filter
        pool_size = top_k * 4 if (category or district) else top_k

        # 1. Dispatch retrieval
        if algo == "hybrid":
            if self.hybrid_retriever is None:
                logger.warning("Hybrid retriever unavailable, falling back to BM25.")
                raw_results = self.bm25_retriever.search(clean_query, top_k=pool_size)
            else:
                raw_results = self.hybrid_retriever.search(clean_query, top_k=pool_size)
        elif algo == "bm25":
            raw_results = self.bm25_retriever.search(clean_query, top_k=pool_size)
        elif algo == "tfidf":
            raw_results = self.tfidf_retriever.search(clean_query, top_k=pool_size)
        elif algo == "vector":
            if self.vector_searcher is None:
                raise ValueError("Vector searcher is not available. Build vector database first.")
            raw_results = self.vector_searcher.search(clean_query, top_k=pool_size)
        elif algo == "boolean":
            matching_ids = self.boolean_retriever.search(clean_query)
            raw_results = [
                {"doc_id": doc_id, "score": 1.0, "rank": r}
                for r, doc_id in enumerate(matching_ids[:pool_size], start=1)
            ]
        else:
            raise ValueError(
                f"Unknown algorithm '{algorithm}'. Choose from: 'hybrid', 'bm25', 'tfidf', 'vector', 'boolean'."
            )

        # 2. Enrich and apply metadata filters
        enriched_results: List[Dict[str, Any]] = []

        cat_filter = category.lower().strip() if category else None
        dist_filter = district.lower().strip() if district else None

        for item in raw_results:
            doc_id = item["doc_id"]
            rec = self.records_by_id.get(doc_id)
            if not rec:
                continue

            # Check category filter
            if cat_filter:
                r_cat = (rec.get("category") or "").lower()
                r_subs = [s.lower() for s in (rec.get("sub_category") or [])]
                if cat_filter not in r_cat and not any(cat_filter in s for s in r_subs):
                    continue

            # Check district filter
            if dist_filter:
                addr = (rec.get("address") or "").lower()
                if dist_filter not in addr:
                    continue

            enriched = dict(item)
            enriched["name"] = rec.get("name", "")
            enriched["category"] = rec.get("category", "")
            enriched["sub_category"] = rec.get("sub_category", [])
            enriched["address"] = rec.get("address", "")
            enriched["phone"] = rec.get("phone")
            enriched["website"] = rec.get("website")
            enriched["coordinates"] = rec.get("coordinates")

            enriched_results.append(enriched)
            if len(enriched_results) >= top_k:
                break

        # Re-assign 1-based ranks after filtering
        for new_rank, item in enumerate(enriched_results, start=1):
            item["rank"] = new_rank

        return enriched_results

    def chat(
        self,
        query: str,
        top_k: int = 3,
        temperature: float = 0.3,
    ) -> Dict[str, Any]:
        """Conversational recommendation using the grounded RAG pipeline."""
        return self.rag_pipeline.ask(query=query, top_k=top_k, temperature=temperature)

    def stream_chat(
        self,
        query: str,
        top_k: int = 3,
        temperature: float = 0.3,
        max_tokens: int = 1000,
    ) -> Tuple[List[Dict[str, Any]], Iterator[str], Dict[str, Any]]:
        """Stream a conversational recommendation using the final hybrid retrieval system + LLM."""
        return self.rag_pipeline.stream_ask(
            query=query,
            top_k=top_k,
            temperature=temperature,
            max_tokens=max_tokens,
        )

    def get_restaurant(self, doc_id: int) -> Optional[Dict[str, Any]]:
        """Retrieve full restaurant details by ID."""
        return self.records_by_id.get(doc_id)

    def get_statistics(self) -> Dict[str, Any]:
        """Return engine metadata, index sizes, and vocabulary counts."""
        return {
            "total_restaurants": len(self.records),
            "vocabulary_size": self.index.vocabulary_size,
            "vector_search_available": self.vector_store is not None,
            "total_indexed_vectors": self.vector_store.total_vectors if self.vector_store else 0,
            "rag_available": True,
            "active_llm_provider": self.llm_client.__class__.__name__,
        }
