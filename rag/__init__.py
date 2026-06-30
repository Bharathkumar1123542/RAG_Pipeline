"""
rag/__init__.py
Central configuration dataclass for the RAG pipeline.
All tuneable parameters are exposed here and may be overridden via environment variables.
"""
from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass
class RagConfig:
    """Central configuration. Override any field via environment variables."""

    embedding_model: str = os.getenv("RAG_EMBEDDING_MODEL", "all-MiniLM-L6-v2")
    use_mock_embedder: bool = (
        os.getenv("RAG_USE_MOCK_EMBEDDER", "false").lower() == "true"
    )
    budget_tokens: int = int(os.getenv("RAG_BUDGET_TOKENS", "2048"))
    chunk_size: int = int(os.getenv("RAG_CHUNK_SIZE", "128"))
    chunk_overlap: int = int(os.getenv("RAG_CHUNK_OVERLAP", "25"))
    dedup_threshold: float = 0.85
    rrf_k: int = 60
    default_top_k: int = 20


# Module-level default config (can be replaced by users)
DEFAULT_CONFIG = RagConfig()
