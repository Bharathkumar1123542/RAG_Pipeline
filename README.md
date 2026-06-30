# RAG Pipeline — Product Description Q&A

A self-contained Retrieval-Augmented Generation pipeline that answers natural-language questions over ~500 synthetic product-description documents. Runs entirely locally — no API keys, no cloud services.

## Quick Start

```bash
pip install -r requirements.txt
python -m nltk.downloader punkt punkt_tab
jupyter lab
```

Then open `rag_pipeline.ipynb` and click **Run All**.

## Architecture

```
CorpusGenerator → DocumentChunker → EmbeddingWrapper (all-MiniLM-L6-v2)
                                           ↓
                               VectorStore (dense cosine + BM25 hybrid)
                                           ↓
                             ContextManager (dedup + token budget)
                                           ↓
                                       MockLLM
```

## Components

| Module                   | Class                         | Role                                       |
| ------------------------ | ----------------------------- | ------------------------------------------ |
| `rag/corpus.py`          | `CorpusGenerator`             | Synthetic product docs (Faker + templates) |
| `rag/chunker.py`         | `DocumentChunker`             | 4 chunking strategies                      |
| `rag/embeddings.py`      | `SentenceTransformerEmbedder` | 384-dim local embeddings                   |
| `rag/vector_store.py`    | `VectorStore`                 | In-memory dense + BM25 hybrid search       |
| `rag/context_manager.py` | `ContextManager`              | Dedup + token-budget assembly              |
| `rag/mock_llm.py`        | `MockLLM`                     | Prompt printer (no API)                    |

## Running Tests

```bash
# Run all tests with coverage report
pytest tests/ --cov=rag --cov-report=term-missing --cov-fail-under=90

# Integration test only
pytest tests/test_integration.py -v

# Execute notebook as a test
jupyter nbconvert --to notebook --execute rag_pipeline.ipynb --output rag_pipeline_executed.ipynb
```

## Environment Variables (all optional)

| Variable                | Default            | Purpose                          |
| ----------------------- | ------------------ | -------------------------------- |
| `RAG_EMBEDDING_MODEL`   | `all-MiniLM-L6-v2` | Override embedding model         |
| `RAG_USE_MOCK_EMBEDDER` | `false`            | Force MockEmbedder (no download) |
| `RAG_BUDGET_TOKENS`     | `2048`             | Default context token budget     |
| `RAG_CHUNK_SIZE`        | `128`              | Default chunk size (tokens)      |
| `RAG_CHUNK_OVERLAP`     | `25`               | Default overlap (tokens)         |

## Security Notes

- No data leaves the local process
- No secrets required
- `.env` is gitignored (add API keys there if extending to real LLMs)
- Run `pip audit` to check for dependency vulnerabilities

## Extending to a Real LLM

Subclass `BaseLLM` in `rag/mock_llm.py`:

```python
from rag.mock_llm import BaseLLM

class MyLLM(BaseLLM):
    def answer(self, query: str, context) -> str:
        # Call your local llama.cpp, Ollama, or API here
        ...
```
