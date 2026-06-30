"""
rag/vector_store.py
In-memory vector store with CRUD and dual-mode (dense + sparse) retrieval.

Dense search  : Cosine similarity via np.dot over L2-normalised vectors.
Sparse search : BM25Okapi (rank_bm25).
Hybrid        : Reciprocal Rank Fusion (RRF) with k=60.
"""
from __future__ import annotations

import logging
from typing import Literal

import numpy as np
from rank_bm25 import BM25Okapi  # type: ignore

from rag.models import Chunk, SearchResult

logger = logging.getLogger(__name__)

SearchMode = Literal["dense", "sparse", "hybrid"]

_RRF_K = 60  # Default RRF constant; robust across most retrieval tasks


class VectorStore:
    """
    In-memory store of Chunk objects indexed by dense vectors and BM25.

    Parameters
    ----------
    embedding_dim : int
        Dimensionality of the dense embedding vectors.
    rrf_k : int
        Reciprocal Rank Fusion constant (default 60).
    """

    def __init__(self, embedding_dim: int, rrf_k: int = _RRF_K) -> None:
        self._dim = embedding_dim
        self._rrf_k = rrf_k

        # Core storage
        self._chunks: dict[str, Chunk] = {}        # chunk_id → Chunk
        self._vectors: np.ndarray = np.empty((0, embedding_dim), dtype=np.float32)
        self._ids: list[str] = []                  # parallel to _vectors rows

        # BM25 index (rebuilt on every add/delete)
        self._bm25: BM25Okapi | None = None
        self._bm25_ids: list[str] = []             # parallel to BM25 corpus

        # Metadata reverse index: key → {value → set[chunk_id]}
        self._metadata_index: dict[str, dict[str, set[str]]] = {}

    # ------------------------------------------------------------------
    # CRUD
    # ------------------------------------------------------------------

    def add(self, chunks: list[Chunk], vectors: np.ndarray) -> None:
        """
        Add chunks and their corresponding L2-normalised vectors.

        Parameters
        ----------
        chunks  : list[Chunk]
            Chunks to add.
        vectors : np.ndarray
            Shape (len(chunks), embedding_dim), L2-normalised float32.

        Raises
        ------
        ValueError
            If chunk count and vector count differ, if embedding_dim mismatches,
            or if a duplicate chunk_id is detected.
        """
        if len(chunks) == 0:
            return
        if len(chunks) != vectors.shape[0]:
            raise ValueError(
                f"chunks length ({len(chunks)}) != vectors.shape[0] ({vectors.shape[0]})"
            )
        if vectors.shape[1] != self._dim:
            raise ValueError(
                f"Vector dim {vectors.shape[1]} != store dim {self._dim}"
            )

        for chunk in chunks:
            if chunk.chunk_id in self._chunks:
                raise ValueError(
                    f"chunk_id {chunk.chunk_id!r} already exists; "
                    "delete before re-adding."
                )

        # Append to dense index
        self._vectors = np.vstack([self._vectors, vectors.astype(np.float32)]) if len(self._ids) else vectors.astype(np.float32)
        for chunk in chunks:
            self._ids.append(chunk.chunk_id)
            self._chunks[chunk.chunk_id] = chunk
            self._index_metadata(chunk)

        self._rebuild_bm25()
        logger.debug("Added %d chunks; store size: %d", len(chunks), len(self))

    def delete(self, chunk_ids: list[str]) -> int:
        """
        Remove chunks by ID.

        Parameters
        ----------
        chunk_ids : list[str]
            IDs to remove. Non-existent IDs are silently skipped.

        Returns
        -------
        int
            Number of chunks successfully deleted.
        """
        to_delete = set(chunk_ids) & set(self._chunks.keys())
        if not to_delete:
            return 0

        # Rebuild dense index excluding deleted IDs
        keep_mask = np.array(
            [cid not in to_delete for cid in self._ids], dtype=bool
        )
        self._vectors = self._vectors[keep_mask]
        self._ids = [cid for cid in self._ids if cid not in to_delete]

        for cid in to_delete:
            chunk = self._chunks.pop(cid)
            self._deindex_metadata(chunk)

        self._rebuild_bm25()
        logger.debug("Deleted %d chunks; store size: %d", len(to_delete), len(self))
        return len(to_delete)

    def get(self, chunk_id: str) -> Chunk | None:
        """Return chunk by ID, or None if not found."""
        return self._chunks.get(chunk_id)

    def __len__(self) -> int:
        """Return number of chunks currently stored."""
        return len(self._chunks)

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    def search(
        self,
        query_vector: np.ndarray,
        query_text: str,
        top_k: int = 10,
        filter_metadata: dict[str, str] | None = None,
        mode: SearchMode = "hybrid",
    ) -> list[SearchResult]:
        """
        Retrieve the top_k most relevant chunks.

        Parameters
        ----------
        query_vector    : np.ndarray, shape (dim,), L2-normalised.
        query_text      : Raw string for BM25 tokenisation.
        top_k           : Maximum results to return.
        filter_metadata : If provided, restrict to chunks matching all key-values.
        mode            : "dense" | "sparse" | "hybrid"

        Returns
        -------
        list[SearchResult]
            Sorted by descending score (rrf_score for hybrid, else dense/sparse).
            Length <= top_k.
        """
        if len(self._ids) == 0:
            return []

        # Determine candidate set after metadata filtering
        candidate_ids = self._apply_metadata_filter(filter_metadata)
        if not candidate_ids:
            return []

        # Build index mapping chunk_id → row position in _vectors / BM25
        id_to_row = {cid: i for i, cid in enumerate(self._ids)}
        candidate_rows = [id_to_row[cid] for cid in candidate_ids if cid in id_to_row]
        candidate_ids_ordered = [self._ids[r] for r in candidate_rows]

        if not candidate_rows:
            return []

        candidate_vectors = self._vectors[candidate_rows]  # (C, D)

        dense_scores = self._dense_search(query_vector, candidate_vectors)
        sparse_scores = self._sparse_search(query_text, candidate_rows)

        results = self._fuse_and_rank(
            candidate_ids=candidate_ids_ordered,
            candidate_vectors=candidate_vectors,
            dense_scores=dense_scores,
            sparse_scores=sparse_scores,
            top_k=top_k,
            mode=mode,
        )
        return results

    def list_metadata_values(self, key: str) -> list[str]:
        """Return all distinct values for a metadata key across all chunks."""
        return sorted(self._metadata_index.get(key, {}).keys())

    # ------------------------------------------------------------------
    # Dense search
    # ------------------------------------------------------------------

    def _dense_search(
        self, query_vector: np.ndarray, candidate_vectors: np.ndarray
    ) -> np.ndarray:
        """Cosine similarity via dot product (vectors are L2-normalised)."""
        qv = query_vector.astype(np.float32).reshape(1, -1)
        scores = np.dot(candidate_vectors, qv.T).flatten()
        # Clip to [0, 1] — floating-point precision can produce values slightly > 1
        return np.clip(scores, 0.0, 1.0)

    # ------------------------------------------------------------------
    # Sparse (BM25) search
    # ------------------------------------------------------------------

    def _sparse_search(
        self, query_text: str, candidate_rows: list[int]
    ) -> np.ndarray:
        """BM25 scores for the candidate rows."""
        if self._bm25 is None or not self._bm25_ids:
            return np.zeros(len(candidate_rows), dtype=np.float32)

        tokens = query_text.lower().split()
        all_scores = self._bm25.get_scores(tokens)  # shape: (N_bm25,)

        # Map BM25 corpus positions back to candidate rows
        bm25_id_to_pos = {cid: i for i, cid in enumerate(self._bm25_ids)}
        scores = np.array(
            [
                all_scores[bm25_id_to_pos[self._ids[r]]]
                if self._ids[r] in bm25_id_to_pos
                else 0.0
                for r in candidate_rows
            ],
            dtype=np.float32,
        )
        return scores

    # ------------------------------------------------------------------
    # Score fusion
    # ------------------------------------------------------------------

    def _fuse_and_rank(
        self,
        candidate_ids: list[str],
        candidate_vectors: np.ndarray,
        dense_scores: np.ndarray,
        sparse_scores: np.ndarray,
        top_k: int,
        mode: SearchMode,
    ) -> list[SearchResult]:
        n = len(candidate_ids)
        if n == 0:
            return []

        # Compute RRF ranks (1-indexed, lower = better)
        dense_ranks = self._scores_to_ranks(dense_scores)
        sparse_ranks = self._scores_to_ranks(sparse_scores)

        rrf_scores = (
            1.0 / (self._rrf_k + dense_ranks) + 1.0 / (self._rrf_k + sparse_ranks)
        )

        # Select sort key based on mode
        match mode:
            case "dense":
                sort_scores = dense_scores
            case "sparse":
                sort_scores = sparse_scores
            case "hybrid":
                sort_scores = rrf_scores
            case _:
                raise ValueError(f"Unknown mode: {mode!r}")

        order = np.argsort(sort_scores)[::-1][:top_k]

        results: list[SearchResult] = []
        for i in order:
            cid = candidate_ids[i]
            chunk = self._chunks[cid]
            results.append(
                SearchResult(
                    chunk=chunk,
                    dense_score=float(dense_scores[i]),
                    sparse_score=float(sparse_scores[i]),
                    rrf_score=float(rrf_scores[i]),
                    vector=candidate_vectors[i],
                )
            )
        return results

    @staticmethod
    def _scores_to_ranks(scores: np.ndarray) -> np.ndarray:
        """Convert score array to 1-indexed ranks (higher score → lower rank)."""
        order = np.argsort(scores)[::-1]
        ranks = np.empty(len(scores), dtype=np.float32)
        ranks[order] = np.arange(1, len(scores) + 1, dtype=np.float32)
        return ranks

    # ------------------------------------------------------------------
    # BM25 index management
    # ------------------------------------------------------------------

    def _rebuild_bm25(self) -> None:
        """Rebuild the BM25 index from the current chunk corpus."""
        if not self._chunks:
            self._bm25 = None
            self._bm25_ids = []
            return
        self._bm25_ids = list(self._ids)
        corpus = [
            self._chunks[cid].text.lower().split() for cid in self._bm25_ids
        ]
        self._bm25 = BM25Okapi(corpus)

    # ------------------------------------------------------------------
    # Metadata index management
    # ------------------------------------------------------------------

    def _index_metadata(self, chunk: Chunk) -> None:
        for key, value in chunk.metadata.items():
            if key not in self._metadata_index:
                self._metadata_index[key] = {}
            if value not in self._metadata_index[key]:
                self._metadata_index[key][value] = set()
            self._metadata_index[key][value].add(chunk.chunk_id)

    def _deindex_metadata(self, chunk: Chunk) -> None:
        for key, value in chunk.metadata.items():
            if key in self._metadata_index and value in self._metadata_index[key]:
                self._metadata_index[key][value].discard(chunk.chunk_id)

    def _apply_metadata_filter(
        self, filter_metadata: dict[str, str] | None
    ) -> list[str]:
        """
        Return chunk IDs passing all filter_metadata key-value constraints.
        If filter_metadata is None, return all chunk IDs.
        """
        if not filter_metadata:
            return list(self._ids)

        # Validate filter keys/values (allowed chars: alphanumeric, _, -, space)
        import re

        allowed = re.compile(r"^[A-Za-z0-9_\- ]+$")
        matching: set[str] | None = None
        for key, value in filter_metadata.items():
            if not allowed.match(str(key)) or not allowed.match(str(value)):
                logger.warning(
                    "filter_metadata key/value contains disallowed characters; "
                    "ignoring key=%r value=%r",
                    key,
                    value,
                )
                continue
            ids = self._metadata_index.get(key, {}).get(value, set())
            matching = ids if matching is None else matching & ids

        if matching is None:
            return list(self._ids)
        # Preserve _ids order
        matching_set = matching
        return [cid for cid in self._ids if cid in matching_set]
