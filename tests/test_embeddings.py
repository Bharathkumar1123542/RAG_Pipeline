"""tests/test_embeddings.py — Unit tests for EmbeddingWrapper implementations."""
import numpy as np
import pytest

from rag.embeddings import MockEmbedder, SentenceTransformerEmbedder


class TestMockEmbedder:
    def test_output_shape_single(self):
        emb = MockEmbedder(dim=64)
        vec = emb.encode("hello world")
        assert vec.shape == (64,)

    def test_output_shape_batch(self):
        emb = MockEmbedder(dim=64)
        vecs = emb.encode_batch(["hello", "world", "test"])
        assert vecs.shape == (3, 64)

    def test_dtype_float32(self):
        emb = MockEmbedder(dim=64)
        assert emb.encode("text").dtype == np.float32
        assert emb.encode_batch(["a", "b"]).dtype == np.float32

    def test_l2_norm_single(self):
        emb = MockEmbedder(dim=64)
        vec = emb.encode("some text here")
        norm = float(np.linalg.norm(vec))
        assert abs(norm - 1.0) < 1e-5, f"L2 norm = {norm}, expected ~1.0"

    def test_l2_norm_batch(self):
        emb = MockEmbedder(dim=64)
        texts = [f"product number {i}" for i in range(50)]
        vecs = emb.encode_batch(texts)
        norms = np.linalg.norm(vecs, axis=1)
        assert np.all(np.abs(norms - 1.0) < 1e-5), "Some row norms deviate from 1.0"

    def test_determinism_same_text(self):
        emb = MockEmbedder(dim=64, seed=42)
        v1 = emb.encode("wireless headphones")
        v2 = emb.encode("wireless headphones")
        np.testing.assert_array_equal(v1, v2)

    def test_determinism_batch(self):
        emb = MockEmbedder(dim=32, seed=7)
        texts = ["alpha", "beta", "gamma"]
        v1 = emb.encode_batch(texts)
        v2 = emb.encode_batch(texts)
        np.testing.assert_array_equal(v1, v2)

    def test_different_texts_different_vectors(self):
        emb = MockEmbedder(dim=64)
        v1 = emb.encode("wireless headphones")
        v2 = emb.encode("wooden bookshelf")
        assert not np.allclose(v1, v2), "Different texts produced identical vectors"

    def test_dim_property(self):
        emb = MockEmbedder(dim=128)
        assert emb.dim == 128

    def test_empty_batch_returns_correct_shape(self):
        emb = MockEmbedder(dim=64)
        vecs = emb.encode_batch([])
        assert vecs.shape == (0, 64)

    def test_custom_dim(self):
        emb = MockEmbedder(dim=256)
        assert emb.encode("test").shape == (256,)
        assert emb.encode_batch(["a", "b"]).shape == (2, 256)

    def test_large_batch(self):
        emb = MockEmbedder(dim=64)
        texts = [f"item {i}" for i in range(200)]
        vecs = emb.encode_batch(texts, batch_size=64)
        assert vecs.shape == (200, 64)
        norms = np.linalg.norm(vecs, axis=1)
        assert np.all(np.abs(norms - 1.0) < 1e-5)


class TestSentenceTransformerEmbedder:
    """Tests for the production embedder — skipped if model unavailable."""

    @pytest.fixture(scope="class")
    def embedder(self):
        try:
            return SentenceTransformerEmbedder()
        except (ImportError, OSError):
            pytest.skip("sentence-transformers model not available in this environment")

    def test_loads_without_error(self, embedder):
        assert embedder is not None

    def test_dim_is_384(self, embedder):
        assert embedder.dim == 384

    def test_encode_shape(self, embedder):
        vec = embedder.encode("wireless noise-cancelling headphones")
        assert vec.shape == (384,)

    def test_encode_dtype(self, embedder):
        vec = embedder.encode("test product")
        assert vec.dtype == np.float32

    def test_l2_norm_single(self, embedder):
        vec = embedder.encode("ergonomic office chair")
        norm = float(np.linalg.norm(vec))
        assert abs(norm - 1.0) < 1e-5

    def test_encode_batch_shape(self, embedder):
        texts = ["product a", "product b", "product c"]
        vecs = embedder.encode_batch(texts)
        assert vecs.shape == (3, 384)

    def test_encode_batch_norms(self, embedder):
        texts = [f"product {i}" for i in range(10)]
        vecs = embedder.encode_batch(texts)
        norms = np.linalg.norm(vecs, axis=1)
        assert np.all(np.abs(norms - 1.0) < 1e-5)

    def test_semantic_similarity(self, embedder):
        """Similar texts should have higher cosine similarity than unrelated ones."""
        v_headphones = embedder.encode("wireless noise-cancelling headphones")
        v_earbuds = embedder.encode("bluetooth earbuds with noise cancellation")
        v_chair = embedder.encode("solid oak ergonomic office chair")
        sim_similar = float(np.dot(v_headphones, v_earbuds))
        sim_different = float(np.dot(v_headphones, v_chair))
        assert sim_similar > sim_different, (
            f"Expected sim(headphones, earbuds)={sim_similar:.3f} > "
            f"sim(headphones, chair)={sim_different:.3f}"
        )
