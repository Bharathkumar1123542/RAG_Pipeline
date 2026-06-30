"""tests/test_context_manager.py — Unit tests for ContextManager."""
import numpy as np
import pytest

from rag.context_manager import ContextManager
from rag.chunker import DocumentChunker
from rag.corpus import CorpusGenerator
from rag.embeddings import MockEmbedder
from rag.models import Chunk, ContextBundle, SearchResult
from rag.vector_store import VectorStore


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_search_result(
    chunk_id: str = "prod_0001_c000",
    doc_id: str = "prod_0001",
    text: str = "This is a test chunk about wireless headphones with noise cancellation.",
    category: str = "electronics",
    dense_score: float = 0.9,
    sparse_score: float = 5.0,
    rrf_score: float = 0.02,
    vector: np.ndarray | None = None,
    chunk_index: int = 0,
) -> SearchResult:
    import tiktoken
    enc = tiktoken.get_encoding("cl100k_base")
    token_count = len(enc.encode(text))
    chunk = Chunk(
        chunk_id=chunk_id,
        doc_id=doc_id,
        text=text,
        strategy="fixed_overlap",
        chunk_index=chunk_index,
        metadata={"category": category, "doc_id": doc_id},
        token_count=token_count,
    )
    if vector is None:
        rng = np.random.default_rng(abs(hash(chunk_id)) % (2**31))
        v = rng.standard_normal(64).astype(np.float32)
        vector = v / np.linalg.norm(v)
    return SearchResult(
        chunk=chunk,
        dense_score=dense_score,
        sparse_score=sparse_score,
        rrf_score=rrf_score,
        vector=vector,
    )


def make_identical_vector() -> np.ndarray:
    v = np.ones(64, dtype=np.float32)
    return v / np.linalg.norm(v)


# ---------------------------------------------------------------------------
# Initialisation
# ---------------------------------------------------------------------------


class TestContextManagerInit:
    def test_default_params(self):
        cm = ContextManager()
        assert cm.budget_tokens == 2048
        assert cm.dedup_threshold == 0.85
        assert cm.ordering == "score"

    def test_invalid_ordering_raises(self):
        with pytest.raises(ValueError, match="ordering"):
            ContextManager(ordering="random")  # type: ignore

    def test_invalid_encoding_raises(self):
        with pytest.raises(ValueError):
            ContextManager(encoding_name="not_a_real_encoding")


# ---------------------------------------------------------------------------
# Budget enforcement
# ---------------------------------------------------------------------------


class TestBudgetEnforcement:
    def test_total_tokens_within_budget(self):
        cm = ContextManager(budget_tokens=50, dedup_threshold=1.0)
        # Create 10 results with short texts
        results = [
            make_search_result(
                chunk_id=f"prod_{i:04d}_c000",
                doc_id=f"prod_{i:04d}",
                text=f"Product {i} description with some keywords.",
                rrf_score=1.0 / (i + 1),
            )
            for i in range(1, 11)
        ]
        bundle = cm.build_context(results, "test query")
        assert bundle.total_tokens_used <= 50

    def test_budget_zero_returns_empty(self):
        cm = ContextManager(budget_tokens=0)
        results = [make_search_result(rrf_score=0.9)]
        bundle = cm.build_context(results, "query")
        assert bundle.context_str == ""
        assert bundle.included_chunk_ids == []
        assert bundle.total_tokens_used == 0

    def test_budget_enforced_across_all_inputs(self):
        """Budget holds for a realistic set of results."""
        docs = CorpusGenerator(seed=42).generate(30)
        chunks = DocumentChunker().chunk_batch(docs)
        embedder = MockEmbedder(dim=64)
        vectors = embedder.encode_batch([c.text for c in chunks])
        store = VectorStore(embedding_dim=64)
        store.add(chunks, vectors)
        qv = embedder.encode("product")
        results = store.search(qv, "product", top_k=20)

        for budget in [128, 256, 512, 1024]:
            cm = ContextManager(budget_tokens=budget, dedup_threshold=1.0)
            bundle = cm.build_context(results, "product")
            assert bundle.total_tokens_used <= budget, (
                f"Budget={budget}: used {bundle.total_tokens_used} tokens"
            )


# ---------------------------------------------------------------------------
# Deduplication
# ---------------------------------------------------------------------------


class TestDeduplication:
    def test_dedup_at_zero_keeps_one(self):
        """With threshold=0.0, all chunks are duplicates of the first — keep only one."""
        # Create two results with identical (cosine sim = 1.0) vectors
        v = make_identical_vector()
        results = [
            make_search_result(
                chunk_id="prod_0001_c000", rrf_score=0.9, vector=v.copy()
            ),
            make_search_result(
                chunk_id="prod_0002_c000", rrf_score=0.8, vector=v.copy()
            ),
        ]
        cm = ContextManager(budget_tokens=2048, dedup_threshold=0.0)
        bundle = cm.build_context(results, "query")
        # Only the first (highest-score) should be included
        assert len(bundle.included_chunk_ids) == 1

    def test_dedup_at_one_keeps_all(self):
        """With threshold=1.0, nothing is a duplicate — keep all chunks."""
        results = [
            make_search_result(
                chunk_id=f"prod_{i:04d}_c000",
                doc_id=f"prod_{i:04d}",
                text=f"Unique text about product variant {i} with distinct features.",
                rrf_score=1.0 / (i + 1),
            )
            for i in range(1, 6)
        ]
        cm = ContextManager(budget_tokens=2048, dedup_threshold=1.0)
        bundle = cm.build_context(results, "query")
        assert len(bundle.included_chunk_ids) == len(results)

    def test_identical_vectors_deduplicated(self):
        """Two chunks with identical vectors and threshold < 1.0: only one kept."""
        v = make_identical_vector()
        results = [
            make_search_result(
                chunk_id="prod_0001_c000",
                text="Wireless headphones with noise cancelling.",
                rrf_score=0.9,
                vector=v.copy(),
            ),
            make_search_result(
                chunk_id="prod_0002_c000",
                text="Wireless headphones with active noise cancelling.",
                rrf_score=0.8,
                vector=v.copy(),
            ),
        ]
        cm = ContextManager(budget_tokens=2048, dedup_threshold=0.85)
        bundle = cm.build_context(results, "headphones")
        assert len(bundle.included_chunk_ids) == 1
        assert bundle.included_chunk_ids[0] == "prod_0001_c000"


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    def test_empty_results_returns_empty_bundle(self):
        cm = ContextManager()
        bundle = cm.build_context([], "query")
        assert isinstance(bundle, ContextBundle)
        assert bundle.context_str == ""
        assert bundle.included_chunk_ids == []
        assert bundle.total_tokens_used == 0

    def test_single_result(self):
        cm = ContextManager(budget_tokens=2048)
        results = [make_search_result(rrf_score=0.5)]
        bundle = cm.build_context(results, "query")
        assert len(bundle.included_chunk_ids) == 1

    def test_query_preserved_in_bundle(self):
        cm = ContextManager()
        query = "wireless headphones under $100"
        bundle = cm.build_context([make_search_result()], query)
        assert bundle.query == query

    def test_budget_tokens_preserved_in_bundle(self):
        cm = ContextManager(budget_tokens=512)
        bundle = cm.build_context([make_search_result()], "query")
        assert bundle.budget_tokens == 512


# ---------------------------------------------------------------------------
# Ordering
# ---------------------------------------------------------------------------


class TestOrdering:
    def test_score_ordering_descending(self):
        """Included chunks should appear in descending score order."""
        results = [
            make_search_result(
                chunk_id=f"prod_{i:04d}_c000",
                doc_id=f"prod_{i:04d}",
                text=f"Product {i}: unique text about its features and benefits.",
                rrf_score=float(i) / 10.0,
                chunk_index=i,
            )
            for i in range(1, 6)
        ]
        cm = ContextManager(budget_tokens=2048, dedup_threshold=1.0, ordering="score")
        bundle = cm.build_context(results, "query")
        # The first included chunk should have the highest rrf_score
        first_id = bundle.included_chunk_ids[0]
        # prod_0005 has the highest score
        assert first_id == "prod_0005_c000"

    def test_document_order_sorting(self):
        """With document_order, chunks are sorted by (doc_id, chunk_index)."""
        results = [
            make_search_result(
                chunk_id="prod_0003_c000",
                doc_id="prod_0003",
                text="Third doc first chunk content about quality products.",
                rrf_score=0.9,
                chunk_index=0,
            ),
            make_search_result(
                chunk_id="prod_0001_c000",
                doc_id="prod_0001",
                text="First doc first chunk content about innovative design.",
                rrf_score=0.5,
                chunk_index=0,
            ),
            make_search_result(
                chunk_id="prod_0002_c000",
                doc_id="prod_0002",
                text="Second doc first chunk content about high performance.",
                rrf_score=0.7,
                chunk_index=0,
            ),
        ]
        cm = ContextManager(
            budget_tokens=2048, dedup_threshold=1.0, ordering="document_order"
        )
        bundle = cm.build_context(results, "query")
        # Should be ordered by doc_id: 0001, 0002, 0003
        assert bundle.included_chunk_ids == [
            "prod_0001_c000",
            "prod_0002_c000",
            "prod_0003_c000",
        ]
