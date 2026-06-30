# implementation.md — RAG Pipeline for Product Description Q&A

---

## 1. Project Overview

### Purpose

Build a Retrieval-Augmented Generation (RAG) pipeline, delivered as a self-contained Jupyter notebook, that answers natural-language questions over a corpus of approximately 500 short product-description documents. The pipeline retrieves the most relevant text chunks and assembles a context window suitable for passage to a language model — without requiring any paid API or cloud service.

### Scope

| In Scope | Out of Scope |
|---|---|
| Synthetic corpus generation (~500 documents) | Production deployment / serving layer |
| Two or more configurable chunking strategies | Fine-tuning any model |
| Pluggable embedding wrapper (local or mock) | Persistent disk-backed vector databases |
| In-memory vector store with CRUD and top-k search | Real-time data ingestion pipelines |
| Hybrid dense + sparse retrieval | Multi-modal (image/audio) content |
| Context budget management and deduplication | Authentication / multi-user session management |
| Mock LLM component that prints assembled context | Paid APIs (OpenAI, Cohere, etc.) |

### Success Criteria

1. The notebook executes end-to-end (`Run All`) without errors on a standard Python 3.10+ environment with dependencies installed.
2. All four core components — `DocumentChunker`, `EmbeddingWrapper`, `VectorStore`, and `ContextManager` — expose the interfaces defined in Section 7.
3. A natural-language query over the synthetic corpus returns a ranked, deduplicated, and budget-capped context string within 5 seconds on a CPU-only laptop.
4. Unit tests for each module pass with ≥ 90% line coverage.
5. The design-question section of the notebook contains substantive written answers to all three architectural questions.

---

## 2. System Architecture

### Component Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                        RAG PIPELINE                             │
│                                                                 │
│  ┌──────────────┐   chunks   ┌──────────────┐  vectors+meta    │
│  │  Document    │ ─────────► │  Embedding   │ ───────────────► │
│  │  Chunker     │            │  Wrapper     │                  │
│  └──────────────┘            └──────────────┘                  │
│         ▲                                                       │
│         │ raw docs                          ┌──────────────┐   │
│  ┌──────────────┐                           │  Vector      │   │
│  │  Corpus      │                           │  Store       │   │
│  │  Generator   │                           │  (dense+     │   │
│  └──────────────┘                           │   sparse)    │   │
│                                             └──────────────┘   │
│                                                    │            │
│                                          top-k chunks          │
│                                                    ▼            │
│  ┌──────────────┐  context string  ┌──────────────────────┐   │
│  │  Mock LLM    │ ◄─────────────── │  Context Manager     │   │
│  │  Component   │                  │  (select/order/dedup/ │   │
│  └──────────────┘                  │   truncate)          │   │
│                                    └──────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
```

### Data Flow (Query Path)

```
User Query (str)
      │
      ▼
EmbeddingWrapper.encode(query)          ← dense query vector
      │
      ├──► VectorStore.search_dense(query_vec, top_k=20)   ← cosine sim
      │
      └──► VectorStore.search_sparse(query_tokens, top_k=20) ← BM25
                │                              │
                └──────── score_fusion ─────────┘
                               │
                       top-k CandidateChunks
                               │
                               ▼
                    ContextManager.build_context(
                        candidates,
                        budget_tokens=2048,
                        dedup_threshold=0.85
                    )
                               │
                         context_str
                               │
                               ▼
                    MockLLM.answer(query, context_str)
                               │
                         printed answer
```

### Indexing Path

```
CorpusGenerator.generate(n=500)
      │ list[ProductDocument]
      ▼
DocumentChunker.chunk(doc, strategy="fixed_overlap")
      │ list[Chunk]
      ▼
EmbeddingWrapper.encode_batch(chunk.texts)
      │ np.ndarray [N, D]
      ▼
VectorStore.add(chunks, vectors)
      (also indexes BM25 internally)
```

---

## 3. Tech Stack

| Library | Version | Role | Rationale |
|---|---|---|---|
| Python | ≥ 3.10 | Runtime | `match` syntax; widespread availability |
| `numpy` | ≥ 1.24 | Dense vector math | Vectorised cosine similarity; zero overhead |
| `pandas` | ≥ 2.0 | Corpus DataFrame, metadata joins | Ergonomic tabular operations |
| `scikit-learn` | ≥ 1.3 | `TfidfVectorizer` (sparse baseline), `cosine_similarity` | Well-tested; no external service |
| `sentence-transformers` | ≥ 2.2 | Local embedding model (`all-MiniLM-L6-v2`) | 384-dim, CPU-friendly, Apache-2.0 |
| `rank_bm25` | ≥ 0.2 | BM25 sparse retrieval | Fast pure-Python BM25Okapi |
| `nltk` | ≥ 3.8 | Sentence tokenisation for sentence-based chunking | `sent_tokenize` is robust and offline |
| `tiktoken` | ≥ 0.5 | Token counting for context budget | Accurate GPT-family token counts; usable offline |
| `pytest` | ≥ 7.0 | Unit and integration tests | Standard; compatible with notebooks via `nbval` |
| `ipykernel` | ≥ 6.0 | Notebook runtime | Required for Jupyter execution |
| Jupyter / JupyterLab | ≥ 4.0 | Delivery format | Specified by requirements |
| `faker` | ≥ 19.0 | Synthetic corpus generation | Realistic names, descriptions |

**Why `sentence-transformers` over a mock by default:** `all-MiniLM-L6-v2` downloads once (~90 MB), runs in < 100 ms per batch on CPU, requires no API key, and produces semantically meaningful vectors. The `EmbeddingWrapper` interface is designed so this model can be swapped for a mock in one line without changing any downstream code.

**Why `rank_bm25` over `sklearn.TfidfVectorizer` for sparse retrieval:** BM25 corrects for document-length bias that TF-IDF does not handle. For short product descriptions with variable token counts, BM25 gives more consistent recall.

---

## 4. Directory & File Structure

```
rag_pipeline/
│
├── rag_pipeline.ipynb          # Main deliverable — single executable notebook
│                               # Contains all narrative, code, and design answers
│
├── rag/                        # Importable Python package (cells also import from here)
│   ├── __init__.py
│   ├── corpus.py               # CorpusGenerator: synthetic document creation
│   ├── chunker.py              # DocumentChunker: all chunking strategies
│   ├── embeddings.py           # EmbeddingWrapper: interface + implementations
│   ├── vector_store.py         # VectorStore: in-memory CRUD + dense/sparse search
│   ├── context_manager.py      # ContextManager: selection, dedup, budget truncation
│   ├── mock_llm.py             # MockLLM: prints assembled context, no API calls
│   └── models.py               # Dataclasses: ProductDocument, Chunk, SearchResult
│
├── tests/
│   ├── test_corpus.py
│   ├── test_chunker.py
│   ├── test_embeddings.py
│   ├── test_vector_store.py
│   └── test_context_manager.py
│
├── requirements.txt            # Pinned dependencies
└── README.md                   # Setup instructions (3 commands to run)
```

All business logic lives in `rag/`. The notebook imports from `rag/` and provides narrative, configuration, demonstrations, and design-question answers. This separation keeps the notebook readable and the modules independently testable.

---

## 5. Core Modules

### 5.1 `rag/models.py` — Data Models

Defines the canonical data structures shared across all modules. No business logic.

**Responsibilities:** Declare `ProductDocument`, `Chunk`, `SearchResult`, and `ContextBundle` as frozen `dataclasses`. Validate field types on construction.

### 5.2 `rag/corpus.py` — CorpusGenerator

**Responsibility:** Generate a varied synthetic corpus of `n` `ProductDocument` objects using `faker` and handcrafted templates. Ensures category distribution, feature variation, and specification diversity so retrieval quality can be meaningfully evaluated.

**Inputs:** `n: int = 500`, `seed: int = 42`

**Outputs:** `list[ProductDocument]`

**Key design decision:** Each document is generated from one of 10 product categories (electronics, furniture, clothing, sports, kitchen, beauty, toys, automotive, books, office). Within each category, features and specifications are sampled from category-specific pools. This produces realistic lexical diversity without requiring a real corpus.

### 5.3 `rag/chunker.py` — DocumentChunker

**Responsibility:** Split a `ProductDocument` into a list of `Chunk` objects using a configurable strategy.

**Inputs:**
- `doc: ProductDocument`
- `strategy: Literal["fixed_overlap", "sentence", "paragraph", "recursive"]`
- `chunk_size: int` (tokens for fixed; sentences for sentence-based)
- `overlap: int` (tokens for fixed; sentences for sentence-based)

**Outputs:** `list[Chunk]`

**Strategies implemented:**

| Strategy | Default? | Description | Best for |
|---|---|---|---|
| `fixed_overlap` | **Yes** | Split on token count with configurable token overlap | Short uniform texts; predictable chunk sizes |
| `sentence` | No | Split on NLTK sentence boundaries; group N sentences per chunk | Documents with varied sentence lengths |
| `paragraph` | No | Split on double-newline; keep paragraphs intact | Structured documents with logical paragraph breaks |
| `recursive` | No | Attempt paragraph → sentence → token splits in order | Long heterogeneous documents |

**Why `fixed_overlap` is the default:** Product descriptions are short (50–300 words), structurally uniform, and contain dense keyword-bearing phrases. Fixed-size chunks with a 20% token overlap ensure that phrases at split boundaries appear in at least one chunk in full, and chunk sizes stay predictable — which keeps embedding batch sizes uniform and retrieval latency stable.

### 5.4 `rag/embeddings.py` — EmbeddingWrapper

**Responsibility:** Provide a uniform interface for converting text strings into dense float vectors. Two concrete implementations are provided: `SentenceTransformerEmbedder` (production path) and `MockEmbedder` (test path).

**Interface (abstract base):**

```python
class EmbeddingWrapper(ABC):
    @property
    def dim(self) -> int: ...
    def encode(self, text: str) -> np.ndarray: ...          # shape (dim,)
    def encode_batch(self, texts: list[str]) -> np.ndarray: # shape (N, dim)
```

**`SentenceTransformerEmbedder`:**
- Model: `all-MiniLM-L6-v2`
- Embedding dimension: **384**
- Output dtype: `np.float32`
- Normalisation: L2-normalised (enables cosine similarity via dot product)
- Batch size: 64 (configurable)

**`MockEmbedder`:**
- Embedding dimension: **64** (configurable)
- Implementation: deterministic hash-seeded `np.random` vectors, L2-normalised
- Same interface as `SentenceTransformerEmbedder`
- Use when: CI environments without model downloads, unit tests requiring speed

**Justification for mock:** The surrounding pipeline — chunking, indexing, retrieval, deduplication, budget truncation — is fully exercised even with random vectors. Semantic ranking quality degrades, but pipeline correctness does not. The mock is documented as semantically meaningless and not used in demonstration runs.

### 5.5 `rag/vector_store.py` — VectorStore

**Responsibility:** Maintain an in-memory store of `Chunk` objects indexed by both dense vectors (NumPy array) and sparse BM25 index. Expose add, delete, and dual-mode search.

**Internal state:**
- `_chunks: dict[str, Chunk]` — keyed by `chunk_id`
- `_vectors: np.ndarray` — shape `(N, dim)`, L2-normalised
- `_ids: list[str]` — parallel list of chunk IDs (index alignment with `_vectors`)
- `_bm25: BM25Okapi` — rebuilt on each `add` or `delete` call
- `_metadata_index: dict[str, list[str]]` — maps metadata key → list of matching chunk IDs

**Operations:** See Section 7 for full signatures.

**Top-k dense search:** Cosine similarity via `np.dot(query_vec, _vectors.T)`, `np.argsort` descending. O(N · D) time.

**Top-k sparse search:** `BM25Okapi.get_scores(query_tokens)`, `np.argsort` descending. O(N · V) where V = vocabulary size.

**Score fusion:** Reciprocal Rank Fusion (RRF) with `k=60`. RRF is parameter-robust and avoids scale mismatch between cosine similarities and BM25 scores.

```
rrf_score(d, dense_rank, sparse_rank) = 1/(k + dense_rank) + 1/(k + sparse_rank)
```

**Metadata filtering:** `search` accepts an optional `filter_metadata: dict[str, str]` that restricts candidates to chunks whose metadata contains all specified key-value pairs before scoring.

### 5.6 `rag/context_manager.py` — ContextManager

**Responsibility:** Given a list of `SearchResult` objects from `VectorStore`, produce a single context string that fits within a token budget, with duplicates removed and results ordered by relevance.

**Inputs:**
- `results: list[SearchResult]`
- `budget_tokens: int = 2048`
- `dedup_threshold: float = 0.85` (cosine similarity above which two chunks are considered duplicates)
- `ordering: Literal["score", "document_order"] = "score"`

**Outputs:** `ContextBundle` containing `context_str`, `included_chunk_ids`, `total_tokens_used`

**Deduplication algorithm:**
1. Sort results by descending RRF score.
2. For each candidate chunk (in score order), compute cosine similarity against all already-included chunks.
3. Include the candidate if its maximum similarity to included chunks is below `dedup_threshold`.
4. Stop when adding the next chunk would exceed `budget_tokens`.

**Token counting:** Uses `tiktoken.get_encoding("cl100k_base")` for byte-pair encoding counts. This encoding is available offline after the first run.

**Context string format:**

```
[1] (score=0.847) [Category: Electronics | Doc: prod_0042]
Wireless noise-cancelling headphones with 30-hour battery life...

[2] (score=0.791) [Category: Electronics | Doc: prod_0017]
Over-ear studio monitor headphones with flat frequency response...
```

### 5.7 `rag/mock_llm.py` — MockLLM

**Responsibility:** Simulate LLM completion without any API call. Accepts a query and context string; prints the assembled prompt and returns a placeholder answer string. This component proves the pipeline produces a correctly formatted prompt and makes the notebook runnable without any external credentials.

**Behaviour:** Prints the full prompt (system message + context + query) to stdout. Returns `"[MockLLM: answer would appear here. Replace with real LLM call.]"`.

**Replacement path:** To use a real model, subclass `BaseLLM` and implement `answer(query: str, context: str) -> str`. The notebook contains a commented example using `llama.cpp` via `llama-cpp-python` for a fully local LLM path.

---

## 6. Data Models

### `ProductDocument`

```python
@dataclass(frozen=True)
class ProductDocument:
    doc_id:      str            # "prod_{n:04d}" — zero-padded 4-digit integer
    name:        str            # Product name, 3–8 words
    category:    str            # One of 10 predefined categories
    features:    list[str]      # 3–7 bullet-point feature strings
    specs:       dict[str, str] # Key-value pairs, e.g. {"Weight": "1.2kg"}
    description: str            # Free-text, 50–300 words
    created_at:  str            # ISO-8601 date string, e.g. "2024-03-15"
```

**Validation rules:**
- `doc_id` matches `^prod_\d{4}$`
- `category` is in `VALID_CATEGORIES` (defined as a module-level constant)
- `features` has length 3–7
- `description` has word count 50–300
- All string fields are stripped of leading/trailing whitespace

### `Chunk`

```python
@dataclass(frozen=True)
class Chunk:
    chunk_id:    str            # "{doc_id}_c{chunk_index:03d}"
    doc_id:      str            # Parent document ID
    text:        str            # The chunk's text content
    strategy:    str            # Chunking strategy used to produce this chunk
    chunk_index: int            # 0-based position within the source document
    metadata:    dict[str, str] # Arbitrary key-value metadata (e.g. category, doc_id)
    token_count: int            # Precomputed token count via tiktoken
```

**Validation rules:**
- `chunk_id` matches `^prod_\d{4}_c\d{3}$`
- `text` is non-empty after stripping whitespace
- `token_count` >= 1

### `SearchResult`

```python
@dataclass(frozen=True)
class SearchResult:
    chunk:       Chunk
    dense_score: float          # Cosine similarity ∈ [0, 1] after L2 normalisation
    sparse_score: float         # BM25 score ∈ [0, ∞)
    rrf_score:   float          # Reciprocal Rank Fusion score ∈ [0, 1]
    vector:      np.ndarray     # Shape (dim,) — used by ContextManager for dedup
```

### `ContextBundle`

```python
@dataclass(frozen=True)
class ContextBundle:
    context_str:        str
    included_chunk_ids: list[str]
    total_tokens_used:  int
    budget_tokens:      int
    query:              str
```

---

## 7. API / Interface Contracts

### `CorpusGenerator`

```python
class CorpusGenerator:
    def __init__(self, seed: int = 42) -> None: ...

    def generate(self, n: int = 500) -> list[ProductDocument]:
        """
        Generate n synthetic ProductDocument objects.
        n: Number of documents to generate. Must be >= 1.
        Returns: list of n ProductDocument objects with varied content.
        """
```

### `DocumentChunker`

```python
class DocumentChunker:
    def __init__(
        self,
        strategy: Literal["fixed_overlap", "sentence", "paragraph", "recursive"] = "fixed_overlap",
        chunk_size: int = 128,   # tokens (fixed/recursive) or sentences (sentence)
        overlap: int = 25,       # tokens (fixed/recursive) or sentences (sentence)
    ) -> None: ...

    def chunk(self, doc: ProductDocument) -> list[Chunk]:
        """
        Split doc into Chunk objects using the configured strategy.
        Returns at least one Chunk per document.
        Never returns empty-text Chunks.
        """

    def chunk_batch(self, docs: list[ProductDocument]) -> list[Chunk]:
        """
        Convenience method: chunk all docs and return flat list.
        """
```

### `EmbeddingWrapper` (abstract base)

```python
class EmbeddingWrapper(ABC):
    @property
    @abstractmethod
    def dim(self) -> int:
        """Return embedding dimension."""

    @abstractmethod
    def encode(self, text: str) -> np.ndarray:
        """
        Encode a single text string.
        Returns: np.ndarray of shape (dim,), dtype float32, L2-normalised.
        """

    @abstractmethod
    def encode_batch(self, texts: list[str], batch_size: int = 64) -> np.ndarray:
        """
        Encode a list of text strings.
        Returns: np.ndarray of shape (len(texts), dim), dtype float32, L2-normalised.
        """
```

### `SentenceTransformerEmbedder(EmbeddingWrapper)`

```python
class SentenceTransformerEmbedder(EmbeddingWrapper):
    def __init__(self, model_name: str = "all-MiniLM-L6-v2") -> None: ...
    # dim = 384
```

### `MockEmbedder(EmbeddingWrapper)`

```python
class MockEmbedder(EmbeddingWrapper):
    def __init__(self, dim: int = 64, seed: int = 42) -> None: ...
    # Deterministic: same text always produces same vector within one process.
```

### `VectorStore`

```python
class VectorStore:
    def __init__(self, embedding_dim: int) -> None: ...

    def add(self, chunks: list[Chunk], vectors: np.ndarray) -> None:
        """
        Add chunks and their corresponding L2-normalised vectors.
        chunks and vectors must have equal length.
        Duplicate chunk_ids raise ValueError.
        Rebuilds BM25 index after insertion.
        """

    def delete(self, chunk_ids: list[str]) -> int:
        """
        Remove chunks by ID. Returns count of successfully deleted chunks.
        IDs not present are silently skipped.
        Rebuilds BM25 index after deletion.
        """

    def get(self, chunk_id: str) -> Chunk | None:
        """Return chunk by ID, or None if not found."""

    def search(
        self,
        query_vector: np.ndarray,
        query_text: str,
        top_k: int = 10,
        filter_metadata: dict[str, str] | None = None,
        mode: Literal["dense", "sparse", "hybrid"] = "hybrid",
    ) -> list[SearchResult]:
        """
        Retrieve the top_k most relevant chunks.
        query_vector: shape (dim,), L2-normalised.
        query_text: raw string for BM25 tokenisation.
        filter_metadata: if provided, restrict search to chunks matching all key-values.
        mode: "dense" uses cosine only; "sparse" uses BM25 only; "hybrid" uses RRF.
        Returns: list[SearchResult] sorted by descending rrf_score (or dense/sparse score
                 when mode != "hybrid"), length <= top_k.
        """

    def list_metadata_values(self, key: str) -> list[str]:
        """Return all distinct values for a metadata key across all chunks."""

    def __len__(self) -> int:
        """Return number of chunks currently stored."""
```

### `ContextManager`

```python
class ContextManager:
    def __init__(
        self,
        budget_tokens: int = 2048,
        dedup_threshold: float = 0.85,
        ordering: Literal["score", "document_order"] = "score",
        encoding_name: str = "cl100k_base",
    ) -> None: ...

    def build_context(
        self,
        results: list[SearchResult],
        query: str,
    ) -> ContextBundle:
        """
        Select, deduplicate, order, and truncate results into a context string.
        Deduplication: cosine similarity between chunk vectors above dedup_threshold
                       causes the lower-scoring chunk to be dropped.
        Truncation: chunks are added greedily in score order until budget_tokens is reached.
                    A chunk is included in full or not at all (no mid-chunk truncation).
        Returns: ContextBundle with formatted context_str and accounting fields.
        """
```

### `MockLLM`

```python
class MockLLM:
    def answer(self, query: str, context: ContextBundle) -> str:
        """
        Print assembled prompt to stdout.
        Return placeholder answer string.
        Never raises.
        """
```

---

## 8. Implementation Phases

### Phase 0 — Environment Setup (Day 1)

**Deliverables:**
- `requirements.txt` with pinned versions
- `README.md` with three-command setup (`pip install`, `python -m nltk.downloader punkt`, `jupyter lab`)
- Skeleton `rag/` package with empty modules and `__init__.py`

**Dependencies:** None.

**Definition of done:** `import rag` succeeds in a fresh virtualenv.

---

### Phase 1 — Data Models and Corpus (Day 1)

**Deliverables:**
- `rag/models.py` with all four dataclasses, field validation in `__post_init__`
- `rag/corpus.py` with `CorpusGenerator.generate()`
- `tests/test_corpus.py` with tests for: correct count, category distribution, field validation, description length bounds, spec/feature presence

**Dependencies:** Phase 0.

**Definition of done:** `pytest tests/test_corpus.py` passes; `CorpusGenerator().generate(500)` returns 500 distinct `ProductDocument` objects in < 3 seconds.

---

### Phase 2 — Document Chunker (Day 2)

**Deliverables:**
- `rag/chunker.py` with all four strategies
- `tests/test_chunker.py`: test each strategy for non-empty output, overlap correctness, chunk token counts within expected bounds

**Dependencies:** Phase 1.

**Definition of done:** `pytest tests/test_chunker.py` passes; `DocumentChunker(strategy="fixed_overlap").chunk_batch(500 docs)` completes in < 2 seconds.

---

### Phase 3 — Embedding Wrapper (Day 2)

**Deliverables:**
- `rag/embeddings.py` with `EmbeddingWrapper` ABC, `SentenceTransformerEmbedder`, and `MockEmbedder`
- `tests/test_embeddings.py`: test output shapes, dtypes, L2 norms ≈ 1.0, determinism for `MockEmbedder`, and that `SentenceTransformerEmbedder` loads without error

**Dependencies:** Phase 0.

**Definition of done:** `MockEmbedder().encode_batch(["hello"] * 100)` returns shape `(100, 64)` with all row norms within `1e-6` of 1.0.

---

### Phase 4 — Vector Store (Day 3)

**Deliverables:**
- `rag/vector_store.py`
- `tests/test_vector_store.py`: test add, delete, get, search (dense/sparse/hybrid), filter, dedup-invariance after deletion

**Dependencies:** Phases 2, 3.

**Definition of done:** `VectorStore.search()` on a 500-document corpus (approx. 1500 chunks) returns results in < 500 ms per query; all CRUD operations maintain internal consistency.

---

### Phase 5 — Context Manager and Mock LLM (Day 3)

**Deliverables:**
- `rag/context_manager.py`
- `rag/mock_llm.py`
- `tests/test_context_manager.py`: test budget enforcement, deduplication at various thresholds, ordering modes, edge cases (empty results, single result, all results identical)

**Dependencies:** Phase 4.

**Definition of done:** `ContextManager(budget_tokens=512).build_context(results, query)` always returns a `ContextBundle` whose `total_tokens_used` ≤ 512.

---

### Phase 6 — Notebook Assembly and Design Questions (Day 4)

**Deliverables:**
- `rag_pipeline.ipynb` with all sections:
  - Introduction and architecture diagram
  - Corpus generation with distribution summary
  - Chunking demonstration (all strategies, with statistics)
  - Embedding walkthrough (batch encoding, timing)
  - Indexing (add all chunks, show store stats)
  - Query demonstration (3+ example queries with printed results)
  - Context assembly demonstration
  - Mock LLM invocation
  - Design questions section with written answers

**Dependencies:** Phases 1–5.

**Definition of done:** `jupyter nbconvert --to notebook --execute rag_pipeline.ipynb` completes without errors.

---

### Phase 7 — Integration Tests and Polish (Day 4–5)

**Deliverables:**
- Full end-to-end test: generate corpus → chunk → embed → index → query → context
- Edge case hardening: empty corpus, single-token query, query with no matches, budget of 0
- README review pass

**Dependencies:** Phase 6.

**Definition of done:** `pytest tests/` passes with ≥ 90% line coverage; notebook produces visually clean output.

---

## 9. Configuration & Environment

### `requirements.txt`

```
numpy>=1.24,<2.0
pandas>=2.0,<3.0
scikit-learn>=1.3,<2.0
sentence-transformers>=2.2,<3.0
rank_bm25>=0.2,<1.0
nltk>=3.8,<4.0
tiktoken>=0.5,<1.0
faker>=19.0,<25.0
pytest>=7.0,<9.0
ipykernel>=6.0,<7.0
jupyterlab>=4.0,<5.0
```

### Environment Variables

No environment variables are required for the default configuration. All components are local or mock.

| Variable | Required | Default | Purpose |
|---|---|---|---|
| `RAG_EMBEDDING_MODEL` | No | `all-MiniLM-L6-v2` | Override the sentence-transformers model name |
| `RAG_USE_MOCK_EMBEDDER` | No | `false` | Set to `true` to force `MockEmbedder` regardless of model availability |
| `RAG_BUDGET_TOKENS` | No | `2048` | Default context budget for `ContextManager` |
| `RAG_CHUNK_SIZE` | No | `128` | Default token chunk size for `DocumentChunker` |
| `RAG_CHUNK_OVERLAP` | No | `25` | Default overlap tokens for `DocumentChunker` |

### Configuration Object

A `RagConfig` dataclass in `rag/__init__.py` centralises all tuneable parameters and reads from environment variables at import time:

```python
@dataclass
class RagConfig:
    embedding_model: str = os.getenv("RAG_EMBEDDING_MODEL", "all-MiniLM-L6-v2")
    use_mock_embedder: bool = os.getenv("RAG_USE_MOCK_EMBEDDER", "false").lower() == "true"
    budget_tokens: int = int(os.getenv("RAG_BUDGET_TOKENS", "2048"))
    chunk_size: int = int(os.getenv("RAG_CHUNK_SIZE", "128"))
    chunk_overlap: int = int(os.getenv("RAG_CHUNK_OVERLAP", "25"))
    dedup_threshold: float = 0.85
    rrf_k: int = 60
    default_top_k: int = 20
```

### Secrets Management

No secrets are required. If a real LLM or embedding API is added later, API keys must be stored in `.env` (never committed) and loaded via `python-dotenv`. A `.gitignore` entry for `.env` is included in the skeleton from Phase 0.

---

## 10. Error Handling & Edge Cases

### Failure Modes and Mitigations

| Failure Mode | Detection Point | Mitigation |
|---|---|---|
| `sentence-transformers` model not downloaded | `SentenceTransformerEmbedder.__init__` | Catch `OSError`; log clear message; offer `MockEmbedder` fallback if `RAG_USE_MOCK_EMBEDDER=true` |
| `chunk_size` < `overlap` | `DocumentChunker.__init__` | Raise `ValueError` immediately with descriptive message |
| Adding duplicate `chunk_id` to `VectorStore` | `VectorStore.add` | Raise `ValueError("chunk_id {id} already exists; delete before re-adding")` |
| Empty `results` passed to `ContextManager` | `build_context` | Return `ContextBundle` with `context_str=""`, `included_chunk_ids=[]`, `total_tokens_used=0` — never raise |
| `budget_tokens=0` | `build_context` | Return empty `ContextBundle` — no error |
| BM25 index query when store is empty | `VectorStore.search` | Return empty list — BM25Okapi handles empty corpus gracefully |
| `tiktoken` encoding name not found | `ContextManager.__init__` | Raise `ValueError` with list of valid encoding names |
| Document with empty `description` | `CorpusGenerator` internal | Generation template always produces non-empty descriptions; validated in `ProductDocument.__post_init__` |
| `numpy` vector shape mismatch in `add` | `VectorStore.add` | Assert `len(chunks) == vectors.shape[0]`; raise `ValueError` with shapes in message |
| `filter_metadata` produces zero candidates | `VectorStore.search` | Return empty list — not an error |

### Logging

Use Python's standard `logging` module. Configure at `DEBUG` level in tests, `INFO` in notebook. Log format: `%(asctime)s [%(levelname)s] %(name)s: %(message)s`. Never use `print` in library code (`mock_llm.py` is the only exception).

---

## 11. Testing Strategy

### Unit Tests (`tests/`)

Each module has a dedicated test file. Tests are pure Python, use no network, and complete in < 30 seconds total.

| Test File | Key Scenarios |
|---|---|
| `test_corpus.py` | Count correctness, category distribution within ±20%, field validation, description word count bounds, spec/feature count bounds, reproducibility with same seed |
| `test_chunker.py` | Each strategy produces ≥ 1 chunk per doc, no empty chunks, overlap correctness for `fixed_overlap` (last N tokens of chunk i == first N tokens of chunk i+1), token counts within `chunk_size ± overlap` bounds, `chunk_batch` flat list length equals sum of individual results |
| `test_embeddings.py` | Output shapes, dtypes, L2 norm ≈ 1.0 (tolerance `1e-5`), `MockEmbedder` determinism, `SentenceTransformerEmbedder` loads without error (skip if model unavailable: `@pytest.mark.skipif`) |
| `test_vector_store.py` | `len` after add/delete, `get` returns correct chunk, `get` returns `None` for missing ID, `search` returns ≤ `top_k` results, `search` with `filter_metadata` excludes non-matching chunks, `delete` removes from both dense and sparse index, BM25 rebuild after delete |
| `test_context_manager.py` | Budget enforcement (total_tokens_used ≤ budget_tokens for all inputs), dedup removes duplicates at threshold=0.0, dedup keeps all at threshold=1.0, empty input returns empty bundle, ordering="document_order" sorts by chunk_index |

### Integration Tests

One end-to-end test in `tests/test_integration.py`:

```python
def test_full_pipeline():
    docs = CorpusGenerator(seed=0).generate(50)
    chunks = DocumentChunker().chunk_batch(docs)
    embedder = MockEmbedder()
    vectors = embedder.encode_batch([c.text for c in chunks])
    store = VectorStore(embedding_dim=embedder.dim)
    store.add(chunks, vectors)
    query = "wireless noise cancelling headphones"
    qvec = embedder.encode(query)
    results = store.search(qvec, query, top_k=5)
    bundle = ContextManager(budget_tokens=512).build_context(results, query)
    assert bundle.total_tokens_used <= 512
    assert len(bundle.included_chunk_ids) >= 1
```

### Notebook Execution Test

```bash
pytest --nbval rag_pipeline.ipynb
```

`nbval` replays the notebook and compares cell outputs. This guards against notebook-library divergence.

### Coverage

```bash
pytest tests/ --cov=rag --cov-report=term-missing --cov-fail-under=90
```

---

## 12. Security Considerations

### Input Validation

- All user-facing function arguments that accept strings are stripped and length-checked before processing. Strings exceeding 10,000 characters are truncated with a warning log rather than raising, to avoid DoS via malformed queries.
- `filter_metadata` dictionary keys and values are validated against an allowed character set (`[A-Za-z0-9_\- ]`) before use. Unrecognised keys are logged and ignored rather than passed to internal indexes.
- Chunk IDs generated by the library follow a deterministic format and are not accepted as user input directly (no lookup-by-arbitrary-string attack surface beyond `get(chunk_id: str)`).

### Data Protection

- No document data leaves the local process. `SentenceTransformerEmbedder` loads the model from disk; it does not transmit text to any remote service.
- The synthetic corpus contains no PII. If real product data containing PII is used, callers are responsible for anonymisation before ingestion.
- The `VectorStore` is not persisted to disk in this implementation. Restart clears all data. If persistence is added, the file must be stored with 600 permissions and excluded from version control.

### Dependency Security

- All dependencies are pinned in `requirements.txt`. A `pip audit` run is included in the CI workflow definition in `README.md`.
- `sentence-transformers` fetches the model from Hugging Face Hub on first use. In air-gapped environments, download the model separately and set `SENTENCE_TRANSFORMERS_HOME` to the local path.

### Notebook Security

- The notebook does not execute shell commands or import from untrusted paths.
- `eval()` and `exec()` are not used anywhere in `rag/`.

---

## 13. Performance & Scalability Notes

### Current System Characteristics (500 documents, ~1500 chunks)

| Operation | Expected Latency | Notes |
|---|---|---|
| Corpus generation | < 3 s | Single-threaded; `faker` is the bottleneck |
| Chunking 500 docs | < 2 s | NLTK `sent_tokenize` dominates for sentence strategy |
| Batch embedding 1500 chunks | < 60 s (CPU, real model) / < 1 s (mock) | `all-MiniLM-L6-v2` with batch_size=64 |
| `VectorStore.add` (1500 chunks) | < 2 s | BM25 rebuild is O(N·V) |
| Dense search | < 10 ms | `np.dot` over (1500, 384) |
| Sparse search (BM25) | < 5 ms | `rank_bm25` with 1500 docs |
| Hybrid RRF fusion | < 1 ms | Pure Python rank merge |
| `ContextManager.build_context` | < 5 ms | Linear dedup pass + tiktoken encoding |

### Scalability Bottlenecks at 500,000 Documents

At 500,000 documents (approximately 1.5M chunks at 3 chunks/doc):

**What fails first — Dense Vector Search:**

The in-memory NumPy dense index stores 1.5M × 384 × 4 bytes ≈ **2.3 GB** of float32 vectors. This exceeds the working RAM of most laptops. Beyond memory, the O(N · D) brute-force cosine search over 1.5M vectors takes approximately 4 seconds per query on a single CPU core — far above the 500 ms interactive threshold.

**What becomes inefficient — BM25 rebuild:**

`BM25Okapi` rebuilds the full inverted index on every `add` or `delete`. At 1.5M chunks, each rebuild takes tens of seconds. The `add` path becomes the write bottleneck before query latency does.

**Redesign at 500,000 documents:**

1. **Replace NumPy dense index with an approximate nearest-neighbour (ANN) library.** `faiss` (Facebook AI Similarity Search) with an `IVF4096,Flat` index reduces search time to < 50 ms at 1.5M vectors with < 1% recall loss. Alternatively, `hnswlib` provides an HNSW graph index with even faster queries.

2. **Replace in-process BM25 with an inverted-index service.** `Elasticsearch` or `OpenSearch` maintain inverted indexes incrementally, eliminating the full rebuild on each mutation. For a fully local option, `Tantivy` (via `tantivy-py`) provides a Rust-backed incremental index.

3. **Introduce batched incremental indexing.** Buffer writes in a pending list; re-index in background batches of 10,000 chunks rather than on every write.

4. **Shard the corpus.** Partition documents by category (or hashed doc_id) across worker processes. Each shard searches independently; results are merged at the coordinator. This distributes both memory and CPU.

5. **Cache query vectors.** Many repeated queries (e.g., "price range", "battery life") will share encoded vectors. An LRU cache over `(query_text → query_vector)` eliminates repeat embedding calls.

### When Combining Dense and Sparse Retrieval Reduces Performance

Hybrid retrieval (dense + sparse) can hurt compared to either alone in these conditions:

1. **Sparse retrieval is clearly superior and dense adds noise.** For exact product-name or SKU queries (e.g., "Sony WH-1000XM5"), BM25 will rank the exact match first. Dense retrieval introduces semantically similar but lexically different results that lower the precision of the top-3 results after RRF fusion. The RRF weight given to the BM25 rank-1 result is diluted by high-scoring dense candidates.

2. **The embedding model's vocabulary mismatch with the corpus.** If the corpus uses highly specialised technical terminology that was underrepresented in the embedding model's training data (e.g., proprietary spec codes), the dense retrieval returns poor candidates whose high semantic similarity scores (relative to other low-quality candidates) inflate the RRF score of irrelevant results above relevant sparse-only matches.

3. **Short, high-precision queries in a large corpus.** When a query is two words with a clear exact match in the corpus, BM25 retrieves it with near-perfect precision. Dense retrieval over a large corpus returns dozens of semantically adjacent but factually wrong results, and after RRF fusion the correct document's rank can fall if many semantically similar documents have high dense scores.

4. **Degenerate embeddings (mock or poorly trained model).** When the dense retrieval component produces random or low-quality vectors, its rankings are noise. RRF fusion dilutes the signal from the high-quality sparse component with noise from the dense component, degrading overall performance below sparse-only.

**Mitigation:** Expose `mode` parameter on `VectorStore.search` so callers can select `"sparse"` or `"dense"` alone when hybrid is known to underperform for a query type.

---

## 14. Open Questions / Assumptions

### Assumptions (stated, not buried)

| # | Assumption | Consequence if wrong |
|---|---|---|
| A1 | The synthetic corpus (generated by `CorpusGenerator`) is sufficient to demonstrate pipeline correctness and retrieve meaningful results. | If real corpus is provided later, re-index required; no code changes needed. |
| A2 | `all-MiniLM-L6-v2` is acceptable as the production-path embedding model. It produces 384-dimensional vectors. | If a different dimension is required, update `EmbeddingWrapper.dim` and rebuild the store. |
| A3 | Token counting uses `cl100k_base` (GPT-4 family) encoding. This is an approximation for models that use different tokenisers. | If the downstream LLM uses a different tokeniser, swap `encoding_name` in `RagConfig`. |
| A4 | The vector store is ephemeral (in-memory only). Data does not survive process restart. | If persistence is required, serialise with `pickle` or `numpy.savez` and add a `save/load` interface. |
| A5 | RRF with `k=60` is an acceptable fusion function. No relevance labels are available to tune fusion weights. | If labelled queries are available, replace RRF with a trained linear combination of dense and sparse scores. |
| A6 | The BM25 tokeniser splits on whitespace and lowercases. No stemming or stopword removal is applied. | For languages other than English, a language-aware tokeniser must be substituted. |
| A7 | The notebook is the primary deliverable. The `rag/` package exists to make the notebook readable and the code testable, not as a standalone library for external consumption. | If `rag/` is to be published as a package, add `pyproject.toml`, versioning, and a public API stability guarantee. |

### Open Questions

| # | Question | Impact | Proposed Resolution |
|---|---|---|---|
| OQ1 | Should the notebook support GPU acceleration for embedding? | Reduces 500-doc indexing from ~60 s to ~5 s. Low impact for 500 docs; high impact for demos. | Add `device="cuda"` param to `SentenceTransformerEmbedder`; auto-detect via `torch.cuda.is_available()`. |
| OQ2 | Is a specific context format required by the downstream LLM consumer? | Context string format may need to change. | The current format (numbered citations with score and metadata) is a standard RAG convention. Confirm with LLM integration team before finalising. |
| OQ3 | Should the notebook include a relevance evaluation (e.g., MRR, Recall@k) against synthetic ground-truth queries? | Adds rigour but requires ground-truth generation (~1 day). | Recommend adding in a follow-on iteration. Not required for initial delivery. |
| OQ4 | Should `DocumentChunker` include the product `name`, `category`, and `features` in every chunk, or only in the first chunk? | Including them in every chunk improves retrieval recall at the cost of redundancy. | Default: include name and category in every chunk's `metadata` field; include in text only for the first chunk. Configurable via `prepend_metadata: bool` parameter. |
