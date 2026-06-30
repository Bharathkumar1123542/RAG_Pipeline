"""
rag/mock_llm.py
MockLLM: simulates LLM completion without any API call.
Accepts a query and ContextBundle; prints the assembled prompt and
returns a placeholder answer string.

To replace with a real model, subclass BaseLLM and implement answer().
"""
from __future__ import annotations

from abc import ABC, abstractmethod

from rag.models import ContextBundle


class BaseLLM(ABC):
    """Abstract base class for LLM components in the RAG pipeline."""

    @abstractmethod
    def answer(self, query: str, context: ContextBundle) -> str:
        """
        Generate an answer given a query and a context bundle.

        Parameters
        ----------
        query   : str
            The user's natural-language question.
        context : ContextBundle
            Assembled context from ContextManager.

        Returns
        -------
        str
            The model's answer string.
        """


class MockLLM(BaseLLM):
    """
    Mock LLM that prints the assembled prompt and returns a placeholder.

    No network calls, no API keys, no model downloads.
    Use this to verify the pipeline produces correctly formatted prompts.

    Parameters
    ----------
    system_prompt : str | None
        Optional system prompt prefix. Defaults to a standard RAG instruction.
    verbose : bool
        If True (default), print the full assembled prompt to stdout.
    """

    _DEFAULT_SYSTEM = (
        "You are a helpful product information assistant. "
        "Answer the user's question using ONLY the information provided in the "
        "context below. If the answer cannot be found in the context, say so clearly. "
        "Do not fabricate information."
    )

    def __init__(
        self,
        system_prompt: str | None = None,
        verbose: bool = True,
    ) -> None:
        self._system = system_prompt or self._DEFAULT_SYSTEM
        self._verbose = verbose

    def answer(self, query: str, context: ContextBundle) -> str:
        """
        Print assembled prompt to stdout and return a placeholder answer.

        Never raises.
        """
        prompt = self._build_prompt(query, context)
        if self._verbose:
            print(prompt)
        return (
            "[MockLLM: answer would appear here. "
            "Replace MockLLM with a real LLM subclass to get actual answers.]\n"
            f"  → Query: {query!r}\n"
            f"  → Context chunks used: {len(context.included_chunk_ids)}\n"
            f"  → Tokens used: {context.total_tokens_used}/{context.budget_tokens}"
        )

    def _build_prompt(self, query: str, context: ContextBundle) -> str:
        separator = "=" * 60
        return (
            f"\n{separator}\n"
            f"SYSTEM\n{separator}\n"
            f"{self._system}\n\n"
            f"{separator}\n"
            f"CONTEXT ({context.total_tokens_used} tokens, "
            f"budget={context.budget_tokens})\n"
            f"{separator}\n"
            f"{context.context_str if context.context_str else '[No relevant context found]'}\n\n"
            f"{separator}\n"
            f"USER QUERY\n{separator}\n"
            f"{query}\n"
            f"{separator}\n"
        )


# ---------------------------------------------------------------------------
# Example real-LLM replacement (commented out — requires llama-cpp-python)
# ---------------------------------------------------------------------------
#
# from llama_cpp import Llama
#
# class LlamaCppLLM(BaseLLM):
#     def __init__(self, model_path: str, n_ctx: int = 4096, **kwargs):
#         self._llm = Llama(model_path=model_path, n_ctx=n_ctx, **kwargs)
#         self._mock = MockLLM(verbose=False)
#
#     def answer(self, query: str, context: ContextBundle) -> str:
#         prompt = self._mock._build_prompt(query, context)
#         output = self._llm(prompt, max_tokens=512, stop=["</s>"])
#         return output["choices"][0]["text"].strip()
