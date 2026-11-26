"""Precompute and persist dense embeddings and FAISS index for all restaurants.

Generates:
  - vector_db/embeddings.npy
  - vector_db/embedding_metadata.json
  - vector_db/faiss_index.bin
  - vector_db/index_metadata.json
"""

import logging
from pathlib import Path
import sys
import time

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.data_loader import DataLoader
from src.utils.document_converter import DocumentConverter
from src.embeddings.embedding_model import EmbeddingModel
from src.embeddings.embedding_generator import EmbeddingGenerator
from src.embeddings.vector_store import VectorStore

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("build_vector_db")


def build_vector_database(force_regenerate: bool = False):
    logger.info("Starting offline Vector Database construction...")
    start_time = time.time()

    # 1. Load dataset
    dataset_path = PROJECT_ROOT / "dataset" / "restaurants.json"
    loader = DataLoader(dataset_path)
    records = loader.load()
    logger.info("Loaded %d restaurant records from %s", len(records), dataset_path)

    # 2. Convert to search documents
    converter = DocumentConverter()
    converter.convert_all(records)
    corpus = converter.get_corpus()
    logger.info("Generated text corpus for %d documents", len(corpus))

    # 3. Initialize Embedding Model & Generator
    model = EmbeddingModel()
    vector_dir = PROJECT_ROOT / "vector_db"
    generator = EmbeddingGenerator(embedding_model=model, vector_dir=vector_dir)

    # 4. Generate or load cached embeddings
    embeddings, doc_ids = generator.generate_or_load(
        corpus=corpus,
        force_regenerate=force_regenerate,
    )
    logger.info(
        "Embeddings ready. Shape: %s, Data type: %s",
        embeddings.shape,
        embeddings.dtype,
    )

    # 5. Populate and save FAISS VectorStore
    logger.info("Populating FAISS VectorStore (IndexFlatIP)...")
    store = VectorStore(dimension=model.get_dimension())
    store.add(embeddings, doc_ids)

    index_path = vector_dir / "faiss_index.bin"
    metadata_path = vector_dir / "index_metadata.json"
    store.save(index_path=index_path, metadata_path=metadata_path)

    elapsed = time.time() - start_time
    logger.info(
        "Vector database build complete in %.2f seconds! Total indexed vectors: %d",
        elapsed,
        store.total_vectors,
    )
    logger.info("Artifacts saved to: %s", vector_dir)


if __name__ == "__main__":
    force = "--force" in sys.argv
    build_vector_database(force_regenerate=force)
