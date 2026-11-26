# KLEAN Chatbot — Restaurant Information Retrieval and Recommendation

An Information Retrieval and Recommendation Engine for discovering restaurants in Phnom Penh, Cambodia using natural-language queries.

The project systematically builds and compares traditional IR algorithms, dense embedding search, hybrid retrieval, and Retrieval-Augmented Generation (RAG).

---

## Project Roadmap

| Phase | Description | Status |
|---|---|---|
| **Phase 1** | **Project setup, dataset loading, document conversion** | **Completed** |
| **Phase 2** | **Text preprocessing and inverted index** | **Completed** |
| **Phase 3** | **Boolean Retrieval, TF-IDF, and BM25** | **Completed** |
| **Phase 4** | **Embedding generation and FAISS vector search** | **Completed** |
| **Phase 5** | **Hybrid Retrieval (BM25 + Vector Search via RRF)** | **Completed** |
| **Phase 6** | **Evaluation and comparison (P@K, Recall@K, MRR)** | **Completed** |
| **Phase 7** | **RAG and LLM integration** | **Completed** |
| **Phase 8** | **FastAPI backend integration** | **Completed** |

---

## Phase 1 Implementation Summary

In **Phase 1**, the foundational architecture of the project was established:

1. **Folder Structure**: Fully modularized Python package structure with separation of concerns (`dataset/`, `notebooks/`, `src/`, `app/`, `vector_db/`, `outputs/`, `tests/`).
2. **Schema Compliance**: Implemented mock dataset (`dataset/restaurants.json`) with 10 varied restaurant records in Phnom Penh strictly following the schema specification.
3. **Data Loader (`src/utils/data_loader.py`)**:
   - Safe immutable JSON loading with schema validation.
   - Preserves original structured restaurant data.
   - Structured metadata filtering by category, sub-category, postal code, and district keyword.
   - Fast ID indexing and Pandas DataFrame export.
4. **Document Converter (`src/utils/document_converter.py`)**:
   - Converts structured records into searchable textual documents.
   - Automatically excludes null/empty attributes.
   - Retains bidirectional mapping: `doc_id -> SearchDocument` and `doc_id -> restaurant record`.
   - Generates corpus format `{doc_id: text}` for subsequent indexers.
5. **Notebooks**:
   - `notebooks/01_dataset_exploration.ipynb`: Dataset exploration and statistics.
   - `notebooks/02_document_conversion.ipynb`: Document conversion demonstration.
6. **Unit Tests**:
   - `tests/test_phase1.py`: 9 comprehensive test cases verifying schema compliance, immutability, filtering, null-handling, and mappings.

---

## Phase 2 Implementation Summary

In **Phase 2**, text preprocessing and inverted index construction were established:

1. **Text Preprocessor (`src/preprocessing/text_preprocessing.py`)**:
   - Configurable lowercasing, punctuation stripping, and tokenization.
   - Dual-language support: English tokenization and stopword removal, plus Khmer word segmentation (`khmer-nltk`) preserving authentic Khmer terms without stripping English stopwords.
   - Handles bilingual text seamlessly.
2. **Inverted Index (`src/indexing/inverted_index.py`)**:
   - Postings list retrieval ($O(1)$ dictionary lookups).
   - Within-document Term Frequency (TF) and Document Frequency (DF) metrics.
   - Document token length tracking and average document length computation.
   - Dynamic document addition (`add_document`), update (`update_document`), and deletion (`remove_document`).
   - JSON serialization (`save` / `load`).
3. **Notebooks**:
   - `notebooks/03_text_preprocessing.ipynb`: Demonstrates English, Khmer, and bilingual preprocessing.
   - `notebooks/04_inverted_index.ipynb`: Builds index on 698 restaurants, tests postings lookups, dynamic updates, and persistence.
4. **Unit Tests**:
   - `tests/test_phase2.py`: 12 automated test cases covering normalization, Khmer segmentation, bilingual handling, postings lookups, dynamic updates, deletions, and persistence.

---

## Phase 3 Implementation Summary

In **Phase 3**, three classic Information Retrieval ranking models were implemented and verified:

1. **Boolean Retrieval (`src/retrieval/boolean_retrieval.py`)**:
   - Set-based deterministic matching using postings lists from the Inverted Index.
   - Supports `AND` (intersection), `OR` (union), and `NOT` (complement).
   - Infix query parser with Shunting-Yard algorithm supporting nested parentheses (e.g. `soup AND (noodles OR beef)`).
2. **TF-IDF Retrieval (`src/retrieval/tfidf.py`)**:
   - Custom implementation from first principles using sublinear TF ($1 + \ln(\text{TF})$) and Smooth Robertson IDF ($\ln((N+1)/(\text{DF}+1)) + 1$).
   - Sparse document vector representation with $L_2$ Euclidean normalization.
   - Cosine similarity ranking using an inverted index postings accumulator.
   - Verified against Scikit-learn's `TfidfVectorizer`.
3. **Okapi BM25 (`src/retrieval/bm25.py`)**:
   - Standard Okapi BM25 probabilistic model.
   - Robertson-Spärck Jones IDF with non-negative smoothing.
   - Configurable $k_1$ (term frequency saturation, default 1.5) and $b$ (document length normalization, default 0.75).
   - `explain_score()` provides full term-by-term mathematical breakdown of candidate scores.
4. **Notebooks**:
   - `notebooks/05_boolean_retrieval.ipynb`: Boolean query evaluation on Phnom Penh restaurants.
   - `notebooks/06_tfidf.ipynb`: TF-IDF vector space modeling and Scikit-learn comparison.
   - `notebooks/07_bm25.ipynb`: BM25 parameter tuning, score breakdown, and side-by-side comparison with TF-IDF.
5. **Unit Tests**:
   - `tests/test_phase3.py`: 11 automated test cases for Boolean queries, TF-IDF ranking, Scikit-learn overlap, BM25 scoring, and score breakdown.

---

## Phase 4 Implementation Summary

In **Phase 4**, dense semantic representation generation and vector database search were established:

1. **Embedding Model (`src/embeddings/embedding_model.py`)**:
   - Local CPU-optimized dense embeddings using Hugging Face's `intfloat/multilingual-e5-small` (384 dimensions).
   - Asymmetric prefix support: `passage: ` for indexed restaurant documents, `query: ` for user search queries.
   - Built-in $L_2$ unit normalization ensuring inner products are mathematically equivalent to cosine similarities.
2. **Embedding Generator & Caching (`src/embeddings/embedding_generator.py`)**:
   - Batch encoding with progress reporting.
   - MD5 content-hash and document ID validation preventing redundant vector computations.
   - Saves precomputed representations to `vector_db/embeddings.npy` and `vector_db/embedding_metadata.json`.
3. **FAISS Vector Store (`src/embeddings/vector_store.py`)**:
   - Wraps Facebook AI Similarity Search (`faiss.IndexFlatIP`) for exact, high-speed cosine similarity retrieval.
   - Bidirectional mapping between internal FAISS 0-indexed positions and restaurant `doc_id` values.
   - Persistence methods (`save` / `load`) managing `vector_db/faiss_index.bin` and `vector_db/index_metadata.json`.
4. **Vector Search Retriever (`src/retrieval/vector_search.py`)**:
   - End-to-end semantic query retriever encoding raw natural-language queries and returning ranked results.
   - Successfully bridges the vocabulary mismatch problem (e.g. matching mood queries like *"romantic rooftop dinner"* or cross-lingual Khmer queries like *"ភោជនីយដ្ឋានភីហ្សា"* to English restaurant records).
5. **CLI & Offline Tools**:
   - `scripts/build_vector_db.py`: Offline script to precalculate and index all 698 restaurants into `vector_db/`.
   - `scripts/search_cli.py`: Updated with `[4] Dense Vector (FAISS)` interactive search mode.
6. **Notebooks**:
   - `notebooks/08_embedding_generation.ipynb`: Demonstrates Multilingual-E5 embedding generation, $L_2$ normalization, and cross-lingual English-Khmer alignment.
   - `notebooks/09_vector_search.ipynb`: Demonstrates FAISS vector search, latency benchmarks, and side-by-side comparison with BM25.
7. **Unit Tests**:
   - `tests/test_phase4.py`: 5 automated unit test cases verifying model properties, caching, FAISS operations, index persistence, and semantic ranking. Total test suite: 37/37 passing.

---

## Phase 5 Implementation Summary

In **Phase 5**, a production-grade Hybrid Retrieval engine was implemented:

1. **Hybrid Retriever (`src/retrieval/hybrid.py`)**:
   - Combines lexical retrieval (`BM25Retriever`) and dense semantic retrieval (`VectorSearcher`).
   - Primary algorithm: **Weighted Reciprocal Rank Fusion (RRF)**:
     $$\text{RRF}(d) = \sum_{m \in \{\text{BM25}, \text{Vector}\}} \frac{w_m}{k + r_m(d)}$$
     with standard smoothing constant $k=60$.
   - Alternative algorithm: **Min-Max Normalized Linear Score Fusion**:
     $$\text{Score}(d) = w_{\text{bm25}} \cdot \text{BM25}_{\text{norm}}(d) + w_{\text{vector}} \cdot \text{Vector}_{\text{norm}}(d)$$
   - Full weight tuning support ($w_{\text{bm25}} : w_{\text{vector}}$) configurable at initialization or per-query.
   - Built-in `explain_fusion()` method providing complete mathematical transparency and audit trails.
2. **Interactive CLI (`scripts/search_cli.py`)**:
   - Enhanced with **Mode 5: Hybrid Retrieval (RRF)** as the primary recommended default mode.
   - Enriches output with dual-engine ranking diagnostics (`BM25 #X`, `Vec #Y`).
3. **Notebooks**:
   - `notebooks/10_hybrid_retrieval.ipynb`: Comprehensive multi-engine evaluation across 4 query archetypes, systematic weight sweep ($1.0/0.0$ to $0.0/1.0$), RRF vs. Linear score normalization comparison, and score breakdowns.
4. **Unit Tests**:
   - `tests/test_phase5.py`: 5 automated unit test cases covering RRF mathematical precision, weight shift behavior, linear score fusion, score explanation, and edge-case handling. Total test suite: 42/42 passing.

---

## Phase 6 Implementation Summary

In **Phase 6**, a comprehensive Information Retrieval evaluation framework and comparative benchmark were established:

1. **Evaluation Metrics Module (`src/evaluation/metrics.py`)**:
   - `precision_at_k`: Precision at arbitrary cutoff rank $K$.
   - `recall_at_k`: Recall at arbitrary cutoff rank $K$.
   - `f1_at_k`: Harmonic mean of $P@K$ and $\text{Recall}@K$.
   - `reciprocal_rank`: Reciprocal rank of first relevant result (basis for **MRR**).
   - `average_precision`: Uninterpolated average precision (basis for **MAP**).
   - `ndcg_at_k`: Normalized Discounted Cumulative Gain with logarithmic discount.
2. **Curated Ground-Truth Benchmark (`src/evaluation/benchmark_dataset.py`)**:
   - 20 realistic ground-truth evaluation queries across 5 distinct query archetypes:
     - `exact_entity`: Specific brand names and street numbers (e.g., *"13Eo Breakfast"*, *"Long Nget St 258"*).
     - `cuisine_dish`: Specialized culinary requests (e.g., *"Japanese sushi sashimi"*, *"beef hotpot soup"*).
     - `ambiance_concept`: Atmosphere and meal types (e.g., *"breakfast and brunch morning cafe"*).
     - `location_neighborhood`: Geographic references (e.g., *"coffee shop in Toul Kork"*).
     - `khmer_script`: Authentic Khmer Unicode queries (e.g., *"គុយទាវ"*, *"ស៊ុបសាច់គោ"*, *"បុកល្ហុង"*).
3. **Automated Benchmark Evaluator (`src/evaluation/evaluator.py`)**:
   - `RetrieverEvaluator`: Benchmarks any retriever conforming to the `.search()` protocol.
   - Computes overall mean metrics and category-wise breakdowns.
   - Exports results to Pandas DataFrames and CSV formats.
4. **Offline Evaluation Tool (`scripts/run_evaluation.py`)**:
   - Runs full benchmark suite across all 4 retrievers and outputs:
     - `outputs/evaluation_benchmark_results.csv`
     - `outputs/evaluation_category_breakdown.csv`
5. **Empirical Benchmark Results (20 Queries, Top-10 Cutoff)**:

   | Retriever | P@1 | P@5 | Recall@5 | Recall@10 | MRR | MAP | NDCG@10 | Latency (ms) |
   |---|---|---|---|---|---|---|---|---|
   | **TF-IDF** | 0.7500 | 0.3300 | 0.6424 | 0.7325 | 0.8092 | 0.6269 | 0.7158 | **0.22 ms** |
   | **Okapi BM25** | **0.8000** | **0.3400** | **0.6632** | 0.7189 | **0.8604** | **0.6447** | **0.7272** | 1.15 ms |
   | **Dense Vector (FAISS)** | 0.5500 | 0.2100 | 0.5205 | 0.6288 | 0.6350 | 0.5099 | 0.5922 | 23.12 ms |
   | **Hybrid (BM25 + FAISS RRF)** | 0.7000 | 0.2900 | 0.5974 | 0.6733 | 0.7767 | 0.5744 | 0.6627 | 19.80 ms |

6. **Notebooks**:
   - `notebooks/11_retrieval_evaluation.ipynb`: Interactive benchmarking, category-wise pivot tables, error analysis, and trade-off insights.
7. **Unit Tests**:
   - `tests/test_phase6.py`: 7 automated unit test cases verifying metrics correctness against exact mathematical derivations, benchmark integrity, and evaluator execution. Total test suite: 49/49 passing.

---

## Phase 7 Implementation Summary

In **Phase 7**, the custom Retrieval-Augmented Generation (RAG) conversational pipeline was implemented:

1. **Context Builder (`src/rag/context_builder.py`)**:
   - Converts top-$k$ retrieved search candidates into compact, structured metadata context blocks.
   - Preserves restaurant names, categories, cuisines, addresses, phone numbers, and websites while suppressing null attributes to optimize token consumption.
2. **Prompt Manager & Guardrails (`src/rag/prompt_templates.py`)**:
   - Implements strict grounding guidelines that instruct the LLM to only recommend restaurants present in the provided context.
   - Enforces fallback behavior when zero relevant candidates are found (prevents hallucinations).
   - Bilingual support: Responds fluently in Khmer for Khmer queries, and English for English inquiries.
3. **Multi-Provider LLM Client (`src/rag/llm_client.py`)**:
   - Universal `BaseLLMClient` interface with providers:
     - `DeepSeekLLMClient`: Native OpenAI-compatible client for DeepSeek (`deepseek-chat` / `deepseek-flash`).
     - `OpenAILLMClient`: OpenAI API client (`gpt-4o-mini`, `gpt-4o`).
     - `GeminiLLMClient`: Google Gemini REST client (`gemini-1.5-flash`).
     - `OllamaLLMClient`: Local offline model client (`llama3.2`).
     - `MockLLMClient`: Deterministic offline fallback ensuring test suites and CI run without paid credentials or network access.
   - `get_llm_client()` factory dynamically reads `.env` configuration.
4. **End-to-End RAG Pipeline (`src/rag/rag_pipeline.py`)**:
   - Seamlessly binds the top-performing Hybrid Retriever with Context Building, Prompting, and LLM Generation.
   - Returns verified answers alongside full source citations and latency breakdowns.
5. **Interactive CLI & Notebooks**:
   - `scripts/search_cli.py`: Added **Mode 6: RAG Chatbot** for live conversational dining advice.
   - `notebooks/12_rag_experiment.ipynb`: Demonstrates conversational QA, Khmer queries, anti-hallucination verification, and latency decomposition.
6. **Unit Tests**:
   - `tests/test_phase7.py`: 8 automated unit test cases verifying context formatting, prompt templating, mock generation, factory fallbacks, anti-hallucination guardrails, and end-to-end pipeline execution. Total test suite: 58/58 passing.

---

## Phase 8 Implementation Summary

In **Phase 8**, the complete application was encapsulated into a unified search engine facade and served via a high-performance **FastAPI REST API**:

1. **Unified Search Engine Facade (`src/search_engine.py`)**:
   - Single programmatic entry point orchestrating all components: DataLoader, Inverted Index, Boolean, TF-IDF, BM25, FAISS Vector Search, Hybrid RRF, and RAG.
   - Built-in metadata pre-filtering and post-filtering by category, cuisine keywords, and district names.
   - Methods:
     - `search(query, algorithm='hybrid', top_k=10, category=None, district=None)`
     - `chat(query, top_k=3, temperature=0.3)`
     - `get_restaurant(doc_id)`
     - `get_statistics()`
2. **FastAPI Application (`app/main.py`)**:
   - Modern async FastAPI service with lifespan preloading of indexes and models.
   - Cross-Origin Resource Sharing (CORS) enabled for easy web frontend or mobile integration.
   - Interactive Swagger API documentation automatically available at `http://localhost:8000/docs`.
3. **RESTful Endpoints**:
   - `GET  /api/v1/search?q={query}&algorithm={hybrid|bm25|tfidf|vector|boolean}&category={cat}&district={dist}`: Fast query param search.
   - `POST /api/v1/search`: Structured JSON body search with advanced filtering.
   - `POST /api/v1/chat`: Conversational RAG recommendation endpoint returning grounded answers and cited source cards.
   - `GET  /api/v1/restaurants/{id}`: Detailed restaurant record lookup.
   - `GET  /api/v1/stats`: Corpus and vocabulary analytics.
   - `GET  /health`: Health monitoring endpoint.
   - `GET  /` & `GET /ui`: Modern Interactive Web UI with Leaflet.js Phnom Penh restaurant map, dual Chatbot & IR Search views, live filter drawer, and bilingual support.
4. **Interactive Web UI (`app/static/index.html`)**:
   - **Interactive Leaflet Map**: Displays Phnom Penh restaurant locations with custom color-coded pins, rating badges, popups, and direct pan/zoom on result click.
   - **Dual Interface Modes**:
     - *Chat Mode*: Conversational assistant powered by DeepSeek / RAG with real-time markdown formatting, source attribution cards, and quick prompt pills.
     - *Search Mode*: Classic IR search engine with algorithm selector (Hybrid RRF, BM25, Dense FAISS, TF-IDF, Boolean), district & category filters, and latency timing badges.
   - **Responsive & Khmer-Friendly**: Styled with Tailwind-inspired dark theme, glassmorphism headers, and Google Font Battambang for Khmer script.
5. **Service Layer & Dependency Injection (`app/services/search_service.py`)**:
   - Thread-safe singleton instance provider with clean dependency injection for route handlers.
6. **How to Run the Application**:
   ```bash
   uvicorn app.main:app --reload --port 8000
   ```
   Open your browser to `http://localhost:8000/` or `http://localhost:8000/ui`.
7. **Unit Tests**:
   - `tests/test_phase8.py`: 11 automated unit test cases testing all REST endpoints via FastAPI `TestClient` (root HTML vs API detection, `/ui`, health, GET search, POST search, filters, invalid algorithms, 404 handling, stats, and chat RAG). Total test suite: 69/69 passing.

---

## Project Structure

```text
KLEAN-Chatbot/
│
├── dataset/
│   └── restaurants.json              # Restaurant dataset (Phnom Penh)
│
├── notebooks/
│   ├── 01_dataset_exploration.ipynb   # Phase 1: Data exploration
│   ├── 02_document_conversion.ipynb   # Phase 1: Conversion to text documents
│   ├── 03_text_preprocessing.ipynb    # Phase 2
│   ├── 04_inverted_index.ipynb        # Phase 2
│   ├── 05_boolean_retrieval.ipynb     # Phase 3
│   ├── 06_tfidf.ipynb                 # Phase 3
│   ├── 07_bm25.ipynb                  # Phase 3
│   ├── 08_embedding_generation.ipynb  # Phase 4
│   ├── 09_vector_search.ipynb         # Phase 4
│   ├── 10_hybrid_retrieval.ipynb      # Phase 5
│   ├── 11_retrieval_evaluation.ipynb  # Phase 6
│   └── 12_rag_experiment.ipynb        # Phase 7
│
├── src/
│   ├── preprocessing/                 # Text cleaning & tokenization
│   ├── indexing/                      # Inverted index
│   ├── retrieval/                     # Boolean, TF-IDF, BM25, Vector, Hybrid
│   ├── embeddings/                    # Local multilingual-e5 embeddings
│   ├── evaluation/                    # P@K, Recall@K, MRR metrics
│   ├── rag/                           # Prompt templates, LLM wrappers, RAG pipeline
│   ├── utils/
│   │   ├── __init__.py
│   │   ├── data_loader.py             # Schema validation, loading & filtering
│   │   └── document_converter.py      # Conversion from JSON to searchable docs
│   └── search_engine.py               # Unified retrieval engine entry point
│
├── app/                               # FastAPI backend
│   ├── main.py
│   ├── routes/
│   └── services/
│
├── vector_db/                         # FAISS index and metadata storage
├── outputs/                           # Evaluation reports, CSVs, benchmarks
├── tests/
│   ├── __init__.py
│   └── test_phase1.py                 # Unit tests for Phase 1
│
├── requirements.txt
├── .env.example
├── .gitignore
└── README.md
```

---

## Setup and Verification

### 1. Environment Setup

Create and activate a virtual environment (optional if using system Python):

```bash
python -m venv venv
# On Windows:
venv\Scripts\activate
# On Linux/macOS:
source venv/bin/activate
```

Install requirements:

```bash
pip install -r requirements.txt
```

### 2. Run Unit Tests

Execute the full automated test suite across all 8 phases (69 tests):

```bash
python -m unittest discover tests
```

Or run individual phase test suites:

```bash
python -m unittest tests/test_phase8.py
```

### 3. Launch the Interactive Web Application

Start the FastAPI application with live reloading:

```bash
uvicorn app.main:app --reload --port 8000
```

Then visit:
- **Web UI & Map**: `http://localhost:8000/` or `http://localhost:8000/ui`
- **Interactive Swagger API Docs**: `http://localhost:8000/docs`

### 4. Interactive Terminal CLI

Search or chat directly from your terminal:

```bash
python scripts/search_cli.py
```

### 3. Quick Python Verification

You can verify dataset loading and conversion via a quick Python snippet:

```python
from src.utils.data_loader import DataLoader
from src.utils.document_converter import DocumentConverter

# Load data
loader = DataLoader("dataset/restaurants.json")
records = loader.load()
print(f"Loaded {len(records)} restaurants")

# Convert to search documents
converter = DocumentConverter()
docs = converter.convert_all(records)
print(f"Sample Document [1]:\n{docs[0].text}")
```
