"""tests/test_chunker.py — Unit tests for DocumentChunker."""
import pytest

from rag.chunker import DocumentChunker
from rag.corpus import CorpusGenerator
from rag.models import Chunk


def make_docs(n=10, seed=42):
    return CorpusGenerator(seed=seed).generate(n)


class TestDocumentChunkerInit:
    def test_default_strategy(self):
        c = DocumentChunker()
        assert c.strategy == "fixed_overlap"

    def test_invalid_overlap_raises(self):
        with pytest.raises(ValueError, match="overlap"):
            DocumentChunker(chunk_size=128, overlap=128)

    def test_negative_overlap_raises(self):
        with pytest.raises(ValueError):
            DocumentChunker(overlap=-1)

    def test_zero_chunk_size_raises(self):
        with pytest.raises(ValueError):
            DocumentChunker(chunk_size=0)


class TestFixedOverlapStrategy:
    def test_at_least_one_chunk(self):
        chunker = DocumentChunker(strategy="fixed_overlap")
        for doc in make_docs(20):
            chunks = chunker.chunk(doc)
            assert len(chunks) >= 1, f"No chunks for {doc.doc_id}"

    def test_no_empty_text_chunks(self):
        chunker = DocumentChunker(strategy="fixed_overlap")
        for doc in make_docs(20):
            for ch in chunker.chunk(doc):
                assert ch.text.strip(), f"Empty chunk text in {ch.chunk_id}"

    def test_chunk_ids_unique_per_doc(self):
        chunker = DocumentChunker(strategy="fixed_overlap")
        for doc in make_docs(10):
            chunks = chunker.chunk(doc)
            ids = [c.chunk_id for c in chunks]
            assert len(ids) == len(set(ids))

    def test_chunk_id_format(self):
        import re
        chunker = DocumentChunker(strategy="fixed_overlap")
        pattern = re.compile(r"^prod_\d{4}_c\d{3}$")
        for doc in make_docs(5):
            for ch in chunker.chunk(doc):
                assert pattern.match(ch.chunk_id), f"Bad chunk_id: {ch.chunk_id!r}"

    def test_token_count_precomputed_matches_text(self):
        import tiktoken
        enc = tiktoken.get_encoding("cl100k_base")
        chunker = DocumentChunker(strategy="fixed_overlap", chunk_size=64, overlap=10)
        for doc in make_docs(5):
            for ch in chunker.chunk(doc):
                actual = len(enc.encode(ch.text))
                assert ch.token_count == actual, (
                    f"{ch.chunk_id}: stored={ch.token_count}, actual={actual}"
                )

    def test_chunk_tokens_within_bounds(self):
        chunk_size = 64
        chunker = DocumentChunker(strategy="fixed_overlap", chunk_size=chunk_size, overlap=10)
        for doc in make_docs(10):
            for ch in chunker.chunk(doc):
                assert ch.token_count <= chunk_size + 5, (  # small tolerance for tiktoken decode
                    f"{ch.chunk_id}: token_count={ch.token_count} > chunk_size={chunk_size}"
                )

    def test_metadata_contains_category(self):
        chunker = DocumentChunker(strategy="fixed_overlap")
        for doc in make_docs(5):
            for ch in chunker.chunk(doc):
                assert "category" in ch.metadata
                assert ch.metadata["category"] == doc.category

    def test_strategy_field_set(self):
        chunker = DocumentChunker(strategy="fixed_overlap")
        for ch in chunker.chunk(make_docs(1)[0]):
            assert ch.strategy == "fixed_overlap"


class TestSentenceStrategy:
    def test_at_least_one_chunk(self):
        chunker = DocumentChunker(strategy="sentence", chunk_size=3, overlap=1)
        for doc in make_docs(10):
            chunks = chunker.chunk(doc)
            assert len(chunks) >= 1

    def test_no_empty_chunks(self):
        chunker = DocumentChunker(strategy="sentence", chunk_size=3, overlap=1)
        for doc in make_docs(10):
            for ch in chunker.chunk(doc):
                assert ch.text.strip()

    def test_strategy_field(self):
        chunker = DocumentChunker(strategy="sentence", chunk_size=3, overlap=0)
        for ch in chunker.chunk(make_docs(1)[0]):
            assert ch.strategy == "sentence"


class TestParagraphStrategy:
    def test_at_least_one_chunk(self):
        chunker = DocumentChunker(strategy="paragraph")
        for doc in make_docs(10):
            assert len(chunker.chunk(doc)) >= 1

    def test_no_empty_chunks(self):
        chunker = DocumentChunker(strategy="paragraph")
        for doc in make_docs(10):
            for ch in chunker.chunk(doc):
                assert ch.text.strip()


class TestRecursiveStrategy:
    def test_at_least_one_chunk(self):
        chunker = DocumentChunker(strategy="recursive")
        for doc in make_docs(10):
            assert len(chunker.chunk(doc)) >= 1

    def test_no_empty_chunks(self):
        chunker = DocumentChunker(strategy="recursive")
        for doc in make_docs(10):
            for ch in chunker.chunk(doc):
                assert ch.text.strip()


class TestChunkBatch:
    def test_flat_list_length(self):
        chunker = DocumentChunker(strategy="fixed_overlap")
        docs = make_docs(20)
        batch_chunks = chunker.chunk_batch(docs)
        individual_count = sum(len(chunker.chunk(d)) for d in docs)
        assert len(batch_chunks) == individual_count

    def test_all_chunks_are_chunk_instances(self):
        chunker = DocumentChunker()
        for ch in chunker.chunk_batch(make_docs(5)):
            assert isinstance(ch, Chunk)

    def test_empty_docs_returns_empty(self):
        chunker = DocumentChunker()
        assert chunker.chunk_batch([]) == []


class TestPrependMetadata:
    def test_prepend_includes_name(self):
        chunker = DocumentChunker(strategy="fixed_overlap", prepend_metadata=True)
        doc = make_docs(1)[0]
        chunks = chunker.chunk(doc)
        assert all(doc.name in ch.text for ch in chunks)
