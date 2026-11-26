"""Interactive Search CLI for KLEAN Chatbot.

Allows testing user queries in real-time against BM25, TF-IDF, or Boolean Retrieval.
"""

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.data_loader import DataLoader
from src.utils.document_converter import DocumentConverter
from src.preprocessing.text_preprocessing import TextPreprocessor
from src.indexing.inverted_index import InvertedIndex
from src.retrieval.boolean_retrieval import BooleanRetriever
from src.retrieval.tfidf import TFIDFRetriever
from src.retrieval.bm25 import BM25Retriever
from src.embeddings.vector_store import VectorStore
from src.retrieval.vector_search import VectorSearcher
from src.retrieval.hybrid import HybridRetriever
from dotenv import load_dotenv
from src.rag.context_builder import ContextBuilder
from src.rag.prompt_templates import PromptManager
from src.rag.llm_client import get_llm_client
from src.rag.rag_pipeline import RAGPipeline


def main():
    print("=" * 70)
    print(" KLEAN CHATBOT — INTERACTIVE SEARCH ENGINE TESTER")
    print("=" * 70)
    print("Loading 698 Phnom Penh restaurants and building index...")

    loader = DataLoader(PROJECT_ROOT / "dataset" / "restaurants.json")
    records = loader.load()

    converter = DocumentConverter()
    docs = converter.convert_all(records)
    corpus = converter.get_corpus()

    preprocessor = TextPreprocessor()
    tokenized_corpus = preprocessor.preprocess_documents(corpus)

    index = InvertedIndex()
    index.build(tokenized_corpus)

    bm25 = BM25Retriever(index, preprocessor)
    tfidf = TFIDFRetriever(index, preprocessor)
    boolean_retriever = BooleanRetriever(index, preprocessor)

    # Load environment variables
    load_dotenv(PROJECT_ROOT / ".env")

    # Initialize Vector Store, Hybrid Retriever & RAG Pipeline if available
    vector_searcher = None
    hybrid_retriever = None
    rag_pipeline = None
    idx_path = PROJECT_ROOT / "vector_db" / "faiss_index.bin"
    meta_path = PROJECT_ROOT / "vector_db" / "index_metadata.json"
    if idx_path.exists() and meta_path.exists():
        vector_store = VectorStore.load(idx_path, meta_path)
        vector_searcher = VectorSearcher(vector_store=vector_store)
        hybrid_retriever = HybridRetriever(
            bm25_retriever=bm25,
            vector_searcher=vector_searcher,
            bm25_weight=0.5,
            vector_weight=0.5,
            rrf_k=60,
        )
        records_by_id = {r["id"]: r for r in records}
        context_builder = ContextBuilder(records_by_id=records_by_id)
        llm_client = get_llm_client(fallback_to_mock=True)
        rag_pipeline = RAGPipeline(
            retriever=hybrid_retriever,
            context_builder=context_builder,
            llm_client=llm_client,
            default_top_k=3,
        )
        print("Loaded FAISS Vector Store, Hybrid Retriever, & RAG Pipeline.")

    print(f"Index built with {index.vocabulary_size:,} unique terms.")
    print("=" * 70)
    print("Algorithms:")
    print("  [1] BM25             [2] TF-IDF          [3] Boolean")
    print("  [4] Dense Vector     [5] Hybrid (RRF)    [6] RAG Chatbot (LLM)")
    print("Type 'exit' or 'quit' to stop.")
    print("=" * 70)

    mode = "rag" if rag_pipeline is not None else ("hybrid" if hybrid_retriever is not None else "bm25")
    print(f"Active mode: {mode.upper()}")

    while True:
        try:
            user_input = input(f"\n[{mode.upper()}] Enter search query: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nExiting search...")
            break

        if not user_input:
            continue

        if user_input.lower() in ("exit", "quit", "q"):
            print("Goodbye!")
            break

        if user_input.lower() in ("mode 1", "mode bm25"):
            mode = "bm25"
            print("Switched mode to: BM25")
            continue
        elif user_input.lower() in ("mode 2", "mode tfidf"):
            mode = "tfidf"
            print("Switched mode to: TF-IDF")
            continue
        elif user_input.lower() in ("mode 3", "mode boolean"):
            mode = "boolean"
            print("Switched mode to: BOOLEAN")
            continue
        elif user_input.lower() in ("mode 4", "mode vector", "mode dense"):
            if vector_searcher is None:
                print("Vector database not found. Please run 'python scripts/build_vector_db.py' first.")
                continue
            mode = "vector"
            print("Switched mode to: VECTOR (Dense Semantic Search)")
            continue
        elif user_input.lower() in ("mode 5", "mode hybrid", "mode rrf"):
            if hybrid_retriever is None:
                print("Hybrid retriever not available (Vector database missing).")
                continue
            mode = "hybrid"
            print("Switched mode to: HYBRID (BM25 + FAISS via RRF)")
            continue
        elif user_input.lower() in ("mode 6", "mode rag", "mode chat"):
            if rag_pipeline is None:
                print("RAG pipeline not available (Vector database missing).")
                continue
            mode = "rag"
            print("Switched mode to: RAG (Conversational Recommendation Chatbot)")
            continue

        # Execute Search / RAG
        if mode == "rag":
            print(f"Consulting KLEAN Chatbot for '{user_input}'...")
            rag_res = rag_pipeline.ask(user_input, top_k=3)
            print("\n" + "=" * 70)
            print(f"KLEAN RECOMMENDATION (Provider: {rag_res['llm_provider']}, Time: {rag_res['latency_ms']:.1f}ms):")
            print("=" * 70)
            print(rag_res["answer"])
            print("\n--- Verified Sources Cited ---")
            for src in rag_res["retrieved_restaurants"]:
                print(f"  • {src['name']} | Address: {src['address']}")
                if src.get("phone"):
                    print(f"    Phone: {src['phone']}")
            print("=" * 70)
            continue

        if mode == "bm25":
            results = bm25.search(user_input, top_k=5)
        elif mode == "tfidf":
            results = tfidf.search(user_input, top_k=5)
        elif mode == "vector":
            results = vector_searcher.search(user_input, top_k=5)
        elif mode == "hybrid":
            results = hybrid_retriever.search(user_input, top_k=5)
        else:
            matching_ids = boolean_retriever.search(user_input)
            results = [{"doc_id": doc_id, "score": 1.0, "rank": r} for r, doc_id in enumerate(matching_ids[:5], start=1)]

        if not results:
            print(f"  No restaurants found matching '{user_input}'.")
            continue

        print(f"\nTop {len(results)} Results for: '{user_input}'")
        print("-" * 70)
        for r in results:
            doc_id = r["doc_id"]
            rec = converter.get_record(doc_id)
            if mode == "hybrid":
                bm_r = f"BM25 #{r['bm25_rank']}" if r.get("bm25_rank") else "No BM25"
                vc_r = f"Vec #{r['vector_rank']}" if r.get("vector_rank") else "No Vec"
                score_str = f"RRF: {r['score']:.5f} ({bm_r}, {vc_r})"
            elif mode != "boolean":
                score_str = f"Score: {r['score']:.4f}"
            else:
                score_str = "Match"
            print(f"Rank {r['rank']} [{score_str}] | ID: {doc_id}")
            print(f"  Name        : {rec['name']}")
            print(f"  Category    : {rec['category']}")
            print(f"  Cuisine     : {', '.join(rec['sub_category']) if rec['sub_category'] else 'N/A'}")
            print(f"  Address     : {rec['address']}")
            if rec.get("phone"):
                print(f"  Phone       : {rec['phone']}")
            if rec.get("website"):
                print(f"  Website     : {rec['website']}")
            print("-" * 70)


if __name__ == "__main__":
    main()
