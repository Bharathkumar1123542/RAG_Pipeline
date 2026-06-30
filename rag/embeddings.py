"""
rag/embeddings.py
Embedding wrapper interface and two concrete implementations:
  - SentenceTransformerEmbedder  (production path, all-MiniLM-L6-v2)
  - MockEmbedder                 (test/CI path, deterministic random vectors)
"""
from __future__ import annotations

import hashlib
import logging
import os
from abc import ABC, abstractmethod

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------


class EmbeddingWrapper(ABC):
    """Uniform interface for text → dense vector conversion."""

    @property
    @abstractmethod
    def dim(self) -> int:
        """Return embedding dimension."""

    @abstractmethod
    def encode(self, text: str) -> np.ndarray:
        """
        Encode a single text string.

        Returns
        -------
        np.ndarray
            Shape (dim,), dtype float32, L2-normalised.
        """

    @abstractmethod
    def encode_batch(self, texts: list[str], batch_size: int = 64) -> np.ndarray:
        """
        Encode a list of text strings.

        Returns
        -------
        np.ndarray
            Shape (len(texts), dim), dtype float32, L2-normalised.
        """

    # ------------------------------------------------------------------
    # Shared utility
    # ------------------------------------------------------------------

    @staticmethod
    def _l2_normalise(matrix: np.ndarray) -> np.ndarray:
        """Row-wise L2 normalisation. Avoids division by zero."""
        norms = np.linalg.norm(matrix, axis=-1, keepdims=True)
        norms = np.where(norms == 0, 1.0, norms)
        return (matrix / norms).astype(np.float32)


# ---------------------------------------------------------------------------
# Production embedder
# ---------------------------------------------------------------------------


class SentenceTransformerEmbedder(EmbeddingWrapper):
    """
    Local sentence-transformers embedder using all-MiniLM-L6-v2.

    Parameters
    ----------
    model_name : str
        HuggingFace model name or local path.
        Override via RAG_EMBEDDING_MODEL env var.
    device : str | None
        "cuda", "cpu", or None (auto-detect via torch).
    """

    def __init__(
        self,
        model_name: str = os.getenv("RAG_EMBEDDING_MODEL", "all-MiniLM-L6-v2"),
        device: str | None = None,
    ) -> None:
        try:
            from sentence_transformers import SentenceTransformer  # type: ignore
        except ImportError as exc:
            raise ImportError(
                "sentence-transformers is not installed. "
                "Run: pip install sentence-transformers"
            ) from exc

        try:
            self._model = SentenceTransformer(model_name, device=device)
            logger.info("Loaded embedding model: %s", model_name)
        except OSError as exc:
            raise OSError(
                f"Could not load model '{model_name}'. "
                "Ensure the model has been downloaded or set "
                "RAG_USE_MOCK_EMBEDDER=true to use MockEmbedder."
            ) from exc

        self._model_name = model_name
        self._dim: int = self._model.get_sentence_embedding_dimension()  # type: ignore

    @property
    def dim(self) -> int:
        return self._dim

    def encode(self, text: str) -> np.ndarray:
        vec = self._model.encode(
            [text],
            batch_size=1,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return vec[0].astype(np.float32)

    def encode_batch(self, texts: list[str], batch_size: int = 64) -> np.ndarray:
        if not texts:
            return np.empty((0, self._dim), dtype=np.float32)
        vecs = self._model.encode(
            texts,
            batch_size=batch_size,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return vecs.astype(np.float32)


# ---------------------------------------------------------------------------
# Mock embedder (deterministic, no model download)
# ---------------------------------------------------------------------------


class MockEmbedder(EmbeddingWrapper):
    """
    Deterministic random embedder for testing and CI environments.

    Produces hash-seeded pseudo-random L2-normalised vectors.
    The same text always produces the same vector within one process.

    .. warning::
        Vectors are semantically meaningless. Retrieval quality degrades
        to random but pipeline correctness is fully exercised.

    Parameters
    ----------
    dim : int
        Embedding dimension (default 64).
    seed : int
        Global random seed (default 42).
    """

    def __init__(self, dim: int = 64, seed: int = 42) -> None:
        self._dim = dim
        self._seed = seed

    @property
    def dim(self) -> int:
        return self._dim

    def encode(self, text: str) -> np.ndarray:
        vec = self._text_to_vector(text)
        return self._l2_normalise(vec.reshape(1, -1))[0]

    def encode_batch(self, texts: list[str], batch_size: int = 64) -> np.ndarray:
        if not texts:
            return np.empty((0, self._dim), dtype=np.float32)
        vecs = np.stack([self._text_to_vector(t) for t in texts], axis=0)
        return self._l2_normalise(vecs)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _text_to_vector(self, text: str) -> np.ndarray:
        """Deterministically map text to a float32 vector via SHA-256 seeding."""
        digest = hashlib.sha256(text.encode("utf-8")).digest()
        # Use first 4 bytes of digest + global seed to seed a local RNG
        int_seed = int.from_bytes(digest[:4], "big") ^ self._seed
        rng = np.random.default_rng(int_seed)
        return rng.standard_normal(self._dim).astype(np.float32)


# ---------------------------------------------------------------------------
# Factory helper
# ---------------------------------------------------------------------------


def get_embedder(use_mock: bool = False, **kwargs: object) -> EmbeddingWrapper:
    """
    Return the appropriate embedder based on the `use_mock` flag or
    the RAG_USE_MOCK_EMBEDDER environment variable.
    """
    force_mock = os.getenv("RAG_USE_MOCK_EMBEDDER", "false").lower() == "true"
    if use_mock or force_mock:
        logger.info("Using MockEmbedder (semantic quality not guaranteed)")
        return MockEmbedder(**{k: v for k, v in kwargs.items() if k in ("dim", "seed")})
    return SentenceTransformerEmbedder(**{k: v for k, v in kwargs.items() if k in ("model_name", "device")})
