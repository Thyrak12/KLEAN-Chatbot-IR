"""Command-line script to run the full IR Evaluation Benchmark on KLEAN Chatbot.

Evaluates and compares:
  1. TF-IDF (Vector Space Model)
  2. Okapi BM25 (Probabilistic Lexical)
  3. FAISS Dense Vector Search (Multilingual-E5)
  4. Hybrid Retrieval (BM25 + FAISS via RRF)

Saves results to:
  - outputs/evaluation_benchmark_results.csv
  - outputs/evaluation_category_breakdown.csv
"""

import logging
from pathlib import Path
import sys
import time

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd

from src.utils.data_loader import DataLoader
from src.utils.document_converter import DocumentConverter
from src.preprocessing.text_preprocessing import TextPreprocessor
from src.indexing.inverted_index import InvertedIndex
from src.retrieval.tfidf import TFIDFRetriever
from src.retrieval.bm25 import BM25Retriever
from src.embeddings.vector_store import VectorStore
from src.embeddings.embedding_model import EmbeddingModel
from src.retrieval.vector_search import VectorSearcher
from src.retrieval.hybrid import HybridRetriever
from src.evaluation.evaluator import RetrieverEvaluator
from src.evaluation.benchmark_dataset import get_benchmark_queries

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("run_evaluation")


def main():
    print("=" * 80)
    print(" KLEAN CHATBOT — INFORMATION RETRIEVAL BENCHMARK EVALUATOR")
    print("=" * 80)
    print("Initializing indexes, models, and ground-truth benchmark queries...")

    # 1. Load Data
    dataset_path = PROJECT_ROOT / "dataset" / "restaurants.json"
    loader = DataLoader(dataset_path)
    records = loader.load()
    converter = DocumentConverter()
    converter.convert_all(records)
    corpus = converter.get_corpus()

    # 2. Inverted Index & Lexical Retrievers
    preprocessor = TextPreprocessor()
    tokenized_corpus = preprocessor.preprocess_documents(corpus)
    index = InvertedIndex()
    index.build(tokenized_corpus)

    tfidf = TFIDFRetriever(index, preprocessor)
    bm25 = BM25Retriever(index, preprocessor)

    # 3. Dense Vector & Hybrid Retrievers
    idx_path = PROJECT_ROOT / "vector_db" / "faiss_index.bin"
    meta_path = PROJECT_ROOT / "vector_db" / "index_metadata.json"
    if not idx_path.exists() or not meta_path.exists():
        logger.error("Vector database not found! Run scripts/build_vector_db.py first.")
        sys.exit(1)

    vector_store = VectorStore.load(idx_path, meta_path)
    embedding_model = EmbeddingModel()
    vector_searcher = VectorSearcher(vector_store=vector_store, embedding_model=embedding_model)

    hybrid = HybridRetriever(
        bm25_retriever=bm25,
        vector_searcher=vector_searcher,
        bm25_weight=0.5,
        vector_weight=0.5,
        rrf_k=60,
    )

    retrievers = {
        "TF-IDF": tfidf,
        "BM25": bm25,
        "Dense Vector (FAISS)": vector_searcher,
        "Hybrid (BM25 + FAISS RRF)": hybrid,
    }

    # 4. Run Benchmark Evaluation
    queries = get_benchmark_queries()
    evaluator = RetrieverEvaluator(queries=queries)
    print(f"\nRunning benchmark across {len(queries)} queries for {len(retrievers)} retrievers...\n")

    overall_df = evaluator.compare_retrievers(retrievers, top_k=10)
    category_df = evaluator.evaluate_by_category(retrievers, top_k=10)

    # 5. Display Formatted Results
    print("=" * 80)
    print(" OVERALL BENCHMARK COMPARISON (20 Ground-Truth Queries, Top-10 Cutoff)")
    print("=" * 80)

    display_cols = ["retriever", "P@1", "P@5", "Recall@5", "Recall@10", "MRR", "MAP", "NDCG@10", "mean_latency_ms"]
    summary_table = overall_df[display_cols].copy()
    for col in ["P@1", "P@5", "Recall@5", "Recall@10", "MRR", "MAP", "NDCG@10"]:
        summary_table[col] = summary_table[col].apply(lambda x: f"{x:.4f}")
    summary_table["mean_latency_ms"] = summary_table["mean_latency_ms"].apply(lambda x: f"{x:.2f} ms")

    print(summary_table.to_string(index=False))

    # 6. Save to outputs/
    output_dir = PROJECT_ROOT / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)

    csv_path = output_dir / "evaluation_benchmark_results.csv"
    overall_df.to_csv(csv_path, index=False)
    logger.info("Saved overall evaluation results to: %s", csv_path)

    cat_csv_path = output_dir / "evaluation_category_breakdown.csv"
    category_df.to_csv(cat_csv_path, index=False)
    logger.info("Saved category breakdown to: %s", cat_csv_path)

    print("\nBenchmark completed successfully!\n")


if __name__ == "__main__":
    main()
