"""
rag/context_manager.py
ContextManager: select, deduplicate, order, and truncate SearchResults
into a single context string that fits within a token budget.
"""
from __future__ import annotations

import logging
from typing import Literal

import numpy as np
import tiktoken

from rag.models import ContextBundle, SearchResult

logger = logging.getLogger(__name__)

OrderingType = Literal["score", "document_order"]


class ContextManager:
    """
    Assemble a ranked, deduplicated, budget-capped context string.

    Parameters
    ----------
    budget_tokens    : int
        Maximum number of tokens in the assembled context string.
    dedup_threshold  : float
        Cosine similarity above which two chunks are considered duplicates.
        The lower-scoring duplicate is dropped.
    ordering         : OrderingType
        "score"          — order included chunks by descending RRF score.
        "document_order" — order by (doc_id, chunk_index).
    encoding_name    : str
        tiktoken encoding to use for token counting.
    """

    def __init__(
        self,
        budget_tokens: int = 2048,
        dedup_threshold: float = 0.85,
        ordering: OrderingType = "score",
        encoding_name: str = "cl100k_base",
    ) -> None:
        if ordering not in ("score", "document_order"):
            raise ValueError(f"ordering must be 'score' or 'document_order', got {ordering!r}")
        try:
            self._enc = tiktoken.get_encoding(encoding_name)
        except Exception as exc:
            valid = ["cl100k_base", "p50k_base", "r50k_base", "gpt2"]
            raise ValueError(
                f"Unknown encoding name {encoding_name!r}. Valid options: {valid}"
            ) from exc

        self.budget_tokens = budget_tokens
        self.dedup_threshold = dedup_threshold
        self.ordering = ordering

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def build_context(
        self,
        results: list[SearchResult],
        query: str,
    ) -> ContextBundle:
        """
        Select, deduplicate, order, and truncate results into a context string.

        Deduplication
        -------------
        Chunks whose cosine similarity to any already-included chunk exceeds
        dedup_threshold are dropped (lower-scoring chunk is discarded).

        Truncation
        ----------
        Chunks are added greedily in score order until budget_tokens is reached.
        A chunk is included in full or not at all (no mid-chunk truncation).

        Returns
        -------
        ContextBundle
            Contains formatted context_str and accounting fields.
        """
        if not results or self.budget_tokens <= 0:
            return ContextBundle(
                context_str="",
                included_chunk_ids=[],
                total_tokens_used=0,
                budget_tokens=self.budget_tokens,
                query=query,
            )

        # Step 1: sort by descending RRF score for greedy selection
        sorted_results = sorted(results, key=lambda r: r.rrf_score, reverse=True)

        # Step 2: greedy deduplication + budget enforcement
        included: list[SearchResult] = []
        included_vectors: list[np.ndarray] = []
        tokens_used = 0

        for result in sorted_results:
            chunk_tokens = self._count_tokens(result.chunk.text)
            if tokens_used + chunk_tokens > self.budget_tokens:
                continue  # Would exceed budget; try next candidate

            # Deduplication check
            if included_vectors and self._is_duplicate(result.vector, included_vectors):
                logger.debug(
                    "Dropping duplicate chunk %s (similarity above threshold %.2f)",
                    result.chunk.chunk_id,
                    self.dedup_threshold,
                )
                continue

            included.append(result)
            included_vectors.append(result.vector)
            tokens_used += chunk_tokens

        # Step 3: re-order for final context string
        if self.ordering == "document_order":
            included = sorted(
                included,
                key=lambda r: (r.chunk.doc_id, r.chunk.chunk_index),
            )
        # else: already in score order

        # Step 4: format context string
        context_str = self._format_context(included)
        total_tokens = self._count_tokens(context_str)

        return ContextBundle(
            context_str=context_str,
            included_chunk_ids=[r.chunk.chunk_id for r in included],
            total_tokens_used=total_tokens,
            budget_tokens=self.budget_tokens,
            query=query,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _count_tokens(self, text: str) -> int:
        return len(self._enc.encode(text))

    def _is_duplicate(
        self, vector: np.ndarray, included_vectors: list[np.ndarray]
    ) -> bool:
        """Return True if vector is too similar to any already-included vector."""
        v = vector.astype(np.float32)
        norm_v = np.linalg.norm(v)
        if norm_v == 0:
            return False
        v_normalised = v / norm_v

        for inc_vec in included_vectors:
            iv = inc_vec.astype(np.float32)
            norm_iv = np.linalg.norm(iv)
            if norm_iv == 0:
                continue
            iv_normalised = iv / norm_iv
            similarity = float(np.dot(v_normalised, iv_normalised))
            if similarity >= self.dedup_threshold:
                return True
        return False

    @staticmethod
    def _format_context(results: list[SearchResult]) -> str:
        """
        Format results into a numbered citation string.

        Example output:
            [1] (score=0.847) [Category: Electronics | Doc: prod_0042]
            Wireless noise-cancelling headphones with 30-hour battery life...

            [2] (score=0.791) [Category: Electronics | Doc: prod_0017]
            Over-ear studio monitor headphones with flat frequency response...
        """
        if not results:
            return ""
        lines: list[str] = []
        for i, result in enumerate(results, start=1):
            meta = result.chunk.metadata
            category = meta.get("category", "Unknown")
            doc_id = meta.get("doc_id", result.chunk.doc_id)
            header = (
                f"[{i}] (score={result.rrf_score:.3f}) "
                f"[Category: {category.title()} | Doc: {doc_id}]"
            )
            lines.append(header)
            lines.append(result.chunk.text)
            lines.append("")  # blank separator
        return "\n".join(lines).rstrip()
