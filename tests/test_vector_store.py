"""tests/test_vector_store.py — Unit tests for VectorStore."""
import numpy as np
import pytest

from rag.chunker import DocumentChunker
from rag.corpus import CorpusGenerator
from rag.embeddings import MockEmbedder
from rag.models import SearchResult
from rag.vector_store import VectorStore


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def embedder():
    return MockEmbedder(dim=64, seed=42)


@pytest.fixture(scope="module")
def small_store(embedder):
    """A store with 30 documents chunked and indexed."""
    docs = CorpusGenerator(seed=42).generate(30)
    chunks = DocumentChunker().chunk_batch(docs)
    vectors = embedder.encode_batch([c.text for c in chunks])
    store = VectorStore(embedding_dim=embedder.dim)
    store.add(chunks, vectors)
    return store, chunks, vectors, embedder


# ---------------------------------------------------------------------------
# Initialisation
# ---------------------------------------------------------------------------


class TestVectorStoreInit:
    def test_empty_store_len(self):
        store = VectorStore(embedding_dim=64)
        assert len(store) == 0

    def test_empty_store_search_returns_empty(self, embedder):
        store = VectorStore(embedding_dim=64)
        qv = embedder.encode("test query")
        results = store.search(qv, "test query", top_k=5)
        assert results == []


# ---------------------------------------------------------------------------
# Add
# ---------------------------------------------------------------------------


class TestAdd:
    def test_len_after_add(self, small_store):
        store, chunks, *_ = small_store
        assert len(store) == len(chunks)

    def test_duplicate_chunk_id_raises(self, embedder):
        docs = CorpusGenerator(seed=1).generate(2)
        chunks = DocumentChunker().chunk_batch(docs)
        vectors = embedder.encode_batch([c.text for c in chunks])
        store = VectorStore(embedding_dim=embedder.dim)
        store.add(chunks, vectors)
        with pytest.raises(ValueError, match="already exists"):
            store.add([chunks[0]], vectors[[0]])

    def test_mismatched_lengths_raises(self, embedder):
        docs = CorpusGenerator(seed=2).generate(2)
        chunks = DocumentChunker().chunk_batch(docs)
        vectors = embedder.encode_batch([c.text for c in chunks])
        store = VectorStore(embedding_dim=embedder.dim)
        with pytest.raises(ValueError):
            store.add(chunks, vectors[:-1])

    def test_wrong_dim_raises(self, embedder):
        docs = CorpusGenerator(seed=3).generate(1)
        chunks = DocumentChunker().chunk_batch(docs)
        wrong_vectors = np.random.rand(len(chunks), 32).astype(np.float32)
        store = VectorStore(embedding_dim=64)
        with pytest.raises(ValueError):
            store.add(chunks, wrong_vectors)

    def test_empty_add_is_noop(self, embedder):
        store = VectorStore(embedding_dim=64)
        store.add([], np.empty((0, 64), dtype=np.float32))
        assert len(store) == 0


# ---------------------------------------------------------------------------
# Get
# ---------------------------------------------------------------------------


class TestGet:
    def test_get_existing_chunk(self, small_store):
        store, chunks, *_ = small_store
        chunk = chunks[0]
        result = store.get(chunk.chunk_id)
        assert result is not None
        assert result.chunk_id == chunk.chunk_id

    def test_get_missing_returns_none(self, small_store):
        store, *_ = small_store
        assert store.get("prod_9999_c999") is None


# ---------------------------------------------------------------------------
# Delete
# ---------------------------------------------------------------------------


class TestDelete:
    def test_delete_reduces_len(self, embedder):
        docs = CorpusGenerator(seed=10).generate(5)
        chunks = DocumentChunker().chunk_batch(docs)
        vectors = embedder.encode_batch([c.text for c in chunks])
        store = VectorStore(embedding_dim=embedder.dim)
        store.add(chunks, vectors)
        before = len(store)
        to_delete = [chunks[0].chunk_id]
        deleted = store.delete(to_delete)
        assert deleted == 1
        assert len(store) == before - 1

    def test_deleted_chunk_not_gettable(self, embedder):
        docs = CorpusGenerator(seed=11).generate(3)
        chunks = DocumentChunker().chunk_batch(docs)
        vectors = embedder.encode_batch([c.text for c in chunks])
        store = VectorStore(embedding_dim=embedder.dim)
        store.add(chunks, vectors)
        cid = chunks[0].chunk_id
        store.delete([cid])
        assert store.get(cid) is None

    def test_delete_missing_id_skipped(self, embedder):
        docs = CorpusGenerator(seed=12).generate(2)
        chunks = DocumentChunker().chunk_batch(docs)
        vectors = embedder.encode_batch([c.text for c in chunks])
        store = VectorStore(embedding_dim=embedder.dim)
        store.add(chunks, vectors)
        deleted = store.delete(["prod_9999_c999"])
        assert deleted == 0

    def test_deleted_chunk_not_in_search(self, embedder):
        docs = CorpusGenerator(seed=13).generate(5)
        chunks = DocumentChunker().chunk_batch(docs)
        vectors = embedder.encode_batch([c.text for c in chunks])
        store = VectorStore(embedding_dim=embedder.dim)
        store.add(chunks, vectors)
        target_id = chunks[0].chunk_id
        store.delete([target_id])
        qv = embedder.encode("product query")
        results = store.search(qv, "product query", top_k=50)
        result_ids = [r.chunk.chunk_id for r in results]
        assert target_id not in result_ids


# ---------------------------------------------------------------------------
# Search
# ---------------------------------------------------------------------------


class TestSearch:
    def test_returns_search_results(self, small_store):
        store, _, __, embedder = small_store
        qv = embedder.encode("wireless headphones")
        results = store.search(qv, "wireless headphones", top_k=5)
        assert all(isinstance(r, SearchResult) for r in results)

    def test_top_k_limit(self, small_store):
        store, _, __, embedder = small_store
        qv = embedder.encode("product")
        results = store.search(qv, "product", top_k=5)
        assert len(results) <= 5

    def test_sorted_descending_rrf(self, small_store):
        store, _, __, embedder = small_store
        qv = embedder.encode("ergonomic chair")
        results = store.search(qv, "ergonomic chair", top_k=10)
        scores = [r.rrf_score for r in results]
        assert scores == sorted(scores, reverse=True)

    def test_dense_mode(self, small_store):
        store, _, __, embedder = small_store
        qv = embedder.encode("kitchen blender")
        results = store.search(qv, "kitchen blender", top_k=5, mode="dense")
        assert len(results) <= 5
        for r in results:
            assert 0.0 <= r.dense_score <= 1.0

    def test_sparse_mode(self, small_store):
        store, _, __, embedder = small_store
        qv = embedder.encode("kitchen blender")
        results = store.search(qv, "kitchen blender", top_k=5, mode="sparse")
        assert len(results) <= 5

    def test_filter_metadata_restricts(self, small_store):
        store, _, __, embedder = small_store
        qv = embedder.encode("product")
        results = store.search(
            qv, "product", top_k=50, filter_metadata={"category": "electronics"}
        )
        for r in results:
            assert r.chunk.metadata.get("category") == "electronics"

    def test_filter_no_match_returns_empty(self, small_store):
        store, _, __, embedder = small_store
        qv = embedder.encode("product")
        results = store.search(
            qv, "product", top_k=10, filter_metadata={"category": "nonexistent_cat"}
        )
        assert results == []

    def test_invalid_mode_raises(self, small_store):
        store, _, __, embedder = small_store
        qv = embedder.encode("test")
        with pytest.raises(ValueError):
            store.search(qv, "test", mode="invalid_mode")  # type: ignore


# ---------------------------------------------------------------------------
# Metadata
# ---------------------------------------------------------------------------


class TestMetadata:
    def test_list_metadata_values(self, small_store):
        store, *_ = small_store
        values = store.list_metadata_values("category")
        assert len(values) > 0
        assert all(isinstance(v, str) for v in values)

    def test_missing_key_returns_empty(self, small_store):
        store, *_ = small_store
        assert store.list_metadata_values("nonexistent_key") == []
