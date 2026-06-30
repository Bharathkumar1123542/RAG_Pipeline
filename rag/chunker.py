"""
rag/chunker.py
DocumentChunker: splits ProductDocument objects into Chunk objects using
one of four configurable strategies.

Strategies
----------
fixed_overlap : Token-based with configurable overlap (DEFAULT)
sentence      : NLTK sentence-boundary chunking
paragraph     : Double-newline split
recursive     : Paragraph → sentence → token fallback
"""
from __future__ import annotations

import logging
import re
from typing import Literal

import tiktoken

from rag.models import Chunk, ProductDocument

logger = logging.getLogger(__name__)

_ENCODING = tiktoken.get_encoding("cl100k_base")

StrategyType = Literal["fixed_overlap", "sentence", "paragraph", "recursive"]


def _tokenise(text: str) -> list[int]:
    return _ENCODING.encode(text)


def _count_tokens(text: str) -> int:
    return len(_tokenise(text))


def _decode_tokens(tokens: list[int]) -> str:
    return _ENCODING.decode(tokens)


# ---------------------------------------------------------------------------
# DocumentChunker
# ---------------------------------------------------------------------------


class DocumentChunker:
    """
    Split a ProductDocument into Chunk objects.

    Parameters
    ----------
    strategy : StrategyType
        Chunking strategy to use (default "fixed_overlap").
    chunk_size : int
        For fixed/recursive: max tokens per chunk.
        For sentence: number of sentences per chunk.
    overlap : int
        For fixed/recursive: token overlap between consecutive chunks.
        For sentence: sentence overlap between consecutive chunks.
    prepend_metadata : bool
        If True, prepend "Name: {name} | Category: {category} |" to each
        chunk's text. Increases recall at the cost of token count.
    """

    def __init__(
        self,
        strategy: StrategyType = "fixed_overlap",
        chunk_size: int = 128,
        overlap: int = 25,
        prepend_metadata: bool = False,
    ) -> None:
        if chunk_size < 1:
            raise ValueError(f"chunk_size must be >= 1, got {chunk_size}")
        if overlap < 0:
            raise ValueError(f"overlap must be >= 0, got {overlap}")
        if overlap >= chunk_size:
            raise ValueError(
                f"overlap ({overlap}) must be < chunk_size ({chunk_size})"
            )

        self.strategy: StrategyType = strategy
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.prepend_metadata = prepend_metadata

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def chunk(self, doc: ProductDocument) -> list[Chunk]:
        """
        Split doc into Chunk objects using the configured strategy.

        Returns at least one Chunk per document. Never returns empty-text Chunks.
        """
        full_text = self._build_full_text(doc)
        raw_texts = self._split(full_text)

        # Guard: never return empty chunks
        raw_texts = [t.strip() for t in raw_texts if t.strip()]
        if not raw_texts:
            raw_texts = [full_text.strip() or doc.description.strip()]

        chunks: list[Chunk] = []
        for idx, text in enumerate(raw_texts):
            if self.prepend_metadata:
                text = f"Name: {doc.name} | Category: {doc.category} | {text}"
            chunk_text = text.strip()
            if not chunk_text:
                continue
            chunks.append(
                Chunk(
                    chunk_id=f"{doc.doc_id}_c{idx:03d}",
                    doc_id=doc.doc_id,
                    text=chunk_text,
                    strategy=self.strategy,
                    chunk_index=idx,
                    metadata={
                        "category": doc.category,
                        "doc_id": doc.doc_id,
                        "name": doc.name,
                    },
                    token_count=_count_tokens(chunk_text),
                )
            )
        return chunks

    def chunk_batch(self, docs: list[ProductDocument]) -> list[Chunk]:
        """Convenience method: chunk all docs and return flat list."""
        all_chunks: list[Chunk] = []
        for doc in docs:
            all_chunks.extend(self.chunk(doc))
        return all_chunks

    # ------------------------------------------------------------------
    # Strategy dispatcher
    # ------------------------------------------------------------------

    def _split(self, text: str) -> list[str]:
        match self.strategy:
            case "fixed_overlap":
                return self._fixed_overlap(text)
            case "sentence":
                return self._sentence(text)
            case "paragraph":
                return self._paragraph(text)
            case "recursive":
                return self._recursive(text)
            case _:
                raise ValueError(f"Unknown strategy: {self.strategy!r}")

    # ------------------------------------------------------------------
    # Strategy implementations
    # ------------------------------------------------------------------

    def _fixed_overlap(self, text: str) -> list[str]:
        """Token-based sliding window with overlap."""
        tokens = _tokenise(text)
        if not tokens:
            return [text]

        step = max(1, self.chunk_size - self.overlap)
        chunks: list[str] = []
        start = 0
        while start < len(tokens):
            end = min(start + self.chunk_size, len(tokens))
            chunk_tokens = tokens[start:end]
            decoded = _decode_tokens(chunk_tokens).strip()
            if decoded:
                chunks.append(decoded)
            if end >= len(tokens):
                break
            start += step
        return chunks

    def _sentence(self, text: str) -> list[str]:
        """NLTK sentence-boundary chunking."""
        try:
            import nltk  # type: ignore

            sentences = nltk.sent_tokenize(text)
        except LookupError:
            logger.warning(
                "NLTK punkt not downloaded. "
                "Run: python -m nltk.downloader punkt punkt_tab. "
                "Falling back to fixed_overlap."
            )
            return self._fixed_overlap(text)

        if not sentences:
            return [text]

        step = max(1, self.chunk_size - self.overlap)
        chunks: list[str] = []
        start = 0
        while start < len(sentences):
            end = min(start + self.chunk_size, len(sentences))
            chunk_text = " ".join(sentences[start:end]).strip()
            if chunk_text:
                chunks.append(chunk_text)
            if end >= len(sentences):
                break
            start += step
        return chunks

    def _paragraph(self, text: str) -> list[str]:
        """Split on double-newline; keep paragraphs intact."""
        raw = re.split(r"\n\s*\n", text)
        chunks = [p.strip() for p in raw if p.strip()]
        # If a paragraph exceeds chunk_size tokens, sub-split it
        result: list[str] = []
        for para in chunks:
            if _count_tokens(para) > self.chunk_size:
                result.extend(self._fixed_overlap(para))
            else:
                result.append(para)
        return result if result else [text]

    def _recursive(self, text: str) -> list[str]:
        """
        Attempt paragraph → sentence → token splits in order.
        Falls back to the next level if a segment is still too large.
        """
        # Level 1: paragraph split
        paragraphs = self._paragraph(text)
        result: list[str] = []
        for para in paragraphs:
            if _count_tokens(para) <= self.chunk_size:
                result.append(para)
            else:
                # Level 2: sentence split
                sentences_chunks = self._sentence(para)
                for sc in sentences_chunks:
                    if _count_tokens(sc) <= self.chunk_size:
                        result.append(sc)
                    else:
                        # Level 3: token split
                        result.extend(self._fixed_overlap(sc))
        return result if result else [text]

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _build_full_text(self, doc: ProductDocument) -> str:
        """
        Assemble the document's text content for chunking.
        Includes name, category, features, specs, and description.
        """
        feature_text = " ".join(f"• {f}" for f in doc.features)
        spec_text = " ".join(f"{k}: {v}" for k, v in doc.specs.items())
        return (
            f"{doc.name}. Category: {doc.category}. "
            f"Features: {feature_text}. "
            f"Specifications: {spec_text}. "
            f"{doc.description}"
        )
