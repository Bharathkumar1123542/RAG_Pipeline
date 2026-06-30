"""
rag/models.py
Canonical data structures shared across all RAG pipeline modules.
All dataclasses are frozen (immutable) and validate their fields on construction.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

import numpy as np

# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------

VALID_CATEGORIES: frozenset[str] = frozenset(
    {
        "electronics",
        "furniture",
        "clothing",
        "sports",
        "kitchen",
        "beauty",
        "toys",
        "automotive",
        "books",
        "office",
    }
)

_DOC_ID_RE = re.compile(r"^prod_\d{4}$")
_CHUNK_ID_RE = re.compile(r"^prod_\d{4}_c\d{3}$")


# ---------------------------------------------------------------------------
# ProductDocument
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ProductDocument:
    """
    A single synthetic product document.

    Fields
    ------
    doc_id      : Unique identifier, format "prod_{n:04d}".
    name        : Product name, 3–8 words.
    category    : One of VALID_CATEGORIES.
    features    : 3–7 bullet-point feature strings.
    specs       : Key-value specification pairs, e.g. {"Weight": "1.2kg"}.
    description : Free-text body, 50–300 words.
    created_at  : ISO-8601 date string, e.g. "2024-03-15".
    """

    doc_id: str
    name: str
    category: str
    features: list[str]
    specs: dict[str, str]
    description: str
    created_at: str

    def __post_init__(self) -> None:
        # Strip strings
        object.__setattr__(self, "name", self.name.strip())
        object.__setattr__(self, "description", self.description.strip())
        object.__setattr__(self, "created_at", self.created_at.strip())

        # Validate doc_id
        if not _DOC_ID_RE.match(self.doc_id):
            raise ValueError(
                f"doc_id must match ^prod_\\d{{4}}$, got: {self.doc_id!r}"
            )

        # Validate category
        if self.category not in VALID_CATEGORIES:
            raise ValueError(
                f"category must be one of {sorted(VALID_CATEGORIES)}, "
                f"got: {self.category!r}"
            )

        # Validate features length
        if not (3 <= len(self.features) <= 7):
            raise ValueError(
                f"features must have 3–7 items, got {len(self.features)}"
            )

        # Validate description word count
        word_count = len(self.description.split())
        if not (50 <= word_count <= 300):
            raise ValueError(
                f"description must have 50–300 words, got {word_count}"
            )

        # Validate non-empty specs
        if not self.specs:
            raise ValueError("specs must have at least one key-value pair")


# ---------------------------------------------------------------------------
# Chunk
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Chunk:
    """
    A text chunk derived from a ProductDocument.

    Fields
    ------
    chunk_id    : Unique identifier, format "{doc_id}_c{chunk_index:03d}".
    doc_id      : Parent document ID.
    text        : The chunk's text content (non-empty).
    strategy    : Chunking strategy used ("fixed_overlap", "sentence", etc.).
    chunk_index : 0-based position within the source document.
    metadata    : Arbitrary key-value metadata (e.g. category, doc_id).
    token_count : Precomputed token count via tiktoken.
    """

    chunk_id: str
    doc_id: str
    text: str
    strategy: str
    chunk_index: int
    metadata: dict[str, str]
    token_count: int

    def __post_init__(self) -> None:
        if not _CHUNK_ID_RE.match(self.chunk_id):
            raise ValueError(
                f"chunk_id must match ^prod_\\d{{4}}_c\\d{{3}}$, got: {self.chunk_id!r}"
            )
        if not self.text.strip():
            raise ValueError("Chunk text must be non-empty")
        if self.token_count < 1:
            raise ValueError(
                f"token_count must be >= 1, got {self.token_count}"
            )


# ---------------------------------------------------------------------------
# SearchResult
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SearchResult:
    """
    A retrieved chunk with associated retrieval scores.

    Fields
    ------
    chunk        : The retrieved Chunk.
    dense_score  : Cosine similarity ∈ [0, 1] after L2 normalisation.
    sparse_score : BM25 score ∈ [0, ∞).
    rrf_score    : Reciprocal Rank Fusion score ∈ (0, 1].
    vector       : The chunk's embedding, shape (dim,). Used for dedup.
    """

    chunk: Chunk
    dense_score: float
    sparse_score: float
    rrf_score: float
    vector: np.ndarray

    def __eq__(self, other: Any) -> bool:  # type: ignore[override]
        if not isinstance(other, SearchResult):
            return NotImplemented
        return self.chunk.chunk_id == other.chunk.chunk_id

    def __hash__(self) -> int:
        return hash(self.chunk.chunk_id)


# ---------------------------------------------------------------------------
# ContextBundle
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ContextBundle:
    """
    The assembled context string and accounting information.

    Fields
    ------
    context_str        : Formatted context ready for injection into an LLM prompt.
    included_chunk_ids : IDs of chunks included (in order of appearance).
    total_tokens_used  : Actual token count of context_str.
    budget_tokens      : The token budget that was applied.
    query              : The original user query.
    """

    context_str: str
    included_chunk_ids: list[str]
    total_tokens_used: int
    budget_tokens: int
    query: str
