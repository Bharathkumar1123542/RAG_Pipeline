"""tests/test_integration.py — End-to-end integration test for the full RAG pipeline."""
import time

import numpy as np
import pytest

from rag.chunker import DocumentChunker
from rag.context_manager import ContextManager
from rag.corpus import CorpusGenerator
from rag.embeddings import MockEmbedder
from rag.mock_llm import MockLLM
from rag.models import ContextBundle
from rag.vector_store import VectorStore


class TestFullPipeline:
    """End-to-end test using MockEmbedder (no network, no model download)."""

    @pytest.fixture(scope="class")
    def pipeline(self):
        docs = CorpusGenerator(seed=0).generate(50)
        chunker = DocumentChunker(strategy="fixed_overlap", chunk_size=128, overlap=25)
        chunks = chunker.chunk_batch(docs)
        embedder = MockEmbedder(dim=64, seed=0)
        vectors = embedder.encode_batch([c.text for c in chunks])
        store = VectorStore(embedding_dim=embedder.dim)
        store.add(chunks, vectors)
        return store, embedder, chunks

    def test_pipeline_smoke(self, pipeline):
        store, embedder, _ = pipeline
        query = "wireless noise cancelling headphones"
        qvec = embedder.encode(query)
        results = store.search(qvec, query, top_k=5)
        bundle = ContextManager(budget_tokens=512).build_context(results, query)
        assert bundle.total_tokens_used <= 512
        assert len(bundle.included_chunk_ids) >= 1

    def test_mock_llm_returns_string(self, pipeline, capsys):
        store, embedder, _ = pipeline
        query = "best ergonomic office chair"
        qvec = embedder.encode(query)
        results = store.search(qvec, query, top_k=5)
        bundle = ContextManager(budget_tokens=512).build_context(results, query)
        llm = MockLLM(verbose=True)
        answer = llm.answer(query, bundle)
        assert isinstance(answer, str)
        assert len(answer) > 0
        captured = capsys.readouterr()
        assert "SYSTEM" in captured.out
        assert "USER QUERY" in captured.out

    def test_end_to_end_budget_enforced(self, pipeline):
        store, embedder, _ = pipeline
        for budget in [64, 128, 256, 512]:
            qvec = embedder.encode("product features")
            results = store.search(qvec, "product features", top_k=20)
            bundle = ContextManager(budget_tokens=budget, dedup_threshold=1.0).build_context(
                results, "product features"
            )
            assert bundle.total_tokens_used <= budget, (
                f"Budget={budget}: used {bundle.total_tokens_used}"
            )

    def test_metadata_filter_in_pipeline(self, pipeline):
        store, embedder, _ = pipeline
        qvec = embedder.encode("comfortable design")
        results = store.search(
            qvec, "comfortable design", top_k=20,
            filter_metadata={"category": "electronics"}
        )
        for r in results:
            assert r.chunk.metadata["category"] == "electronics"

    def test_all_retrieval_modes(self, pipeline):
        store, embedder, _ = pipeline
        query = "lightweight portable bluetooth"
        qvec = embedder.encode(query)
        for mode in ("dense", "sparse", "hybrid"):
            results = store.search(qvec, query, top_k=5, mode=mode)
            assert len(results) <= 5

    def test_document_order_context(self, pipeline):
        store, embedder, _ = pipeline
        qvec = embedder.encode("product review")
        results = store.search(qvec, "product review", top_k=10)
        bundle = ContextManager(
            budget_tokens=1024, dedup_threshold=1.0, ordering="document_order"
        ).build_context(results, "product review")
        # Verify order is monotonically non-decreasing by (doc_id, chunk_index)
        included = [r for r in results if r.chunk.chunk_id in bundle.included_chunk_ids]
        keys = [(r.chunk.doc_id, r.chunk.chunk_index) for r in included]
        assert keys == sorted(keys)

    def test_empty_corpus_graceful(self):
        store = VectorStore(embedding_dim=64)
        embedder = MockEmbedder(dim=64)
        qvec = embedder.encode("any query")
        results = store.search(qvec, "any query", top_k=5)
        bundle = ContextManager().build_context(results, "any query")
        assert bundle.context_str == ""
        assert bundle.included_chunk_ids == []

    def test_no_results_context(self):
        bundle = ContextManager(budget_tokens=512).build_context([], "query")
        assert isinstance(bundle, ContextBundle)
        assert bundle.total_tokens_used == 0

    def test_pipeline_latency(self, pipeline):
        """Query → context should complete in < 5 seconds."""
        store, embedder, _ = pipeline
        query = "waterproof sports equipment"
        start = time.perf_counter()
        qvec = embedder.encode(query)
        results = store.search(qvec, query, top_k=20)
        ContextManager(budget_tokens=2048).build_context(results, query)
        elapsed = time.perf_counter() - start
        assert elapsed < 5.0, f"Pipeline took {elapsed:.2f}s (> 5s)"

    def test_store_stats_after_full_index(self, pipeline):
        store, _, chunks = pipeline
        assert len(store) == len(chunks)
        assert len(store) > 0
