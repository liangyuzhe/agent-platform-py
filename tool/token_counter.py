"""Token counting utilities with tiktoken."""

from __future__ import annotations

from typing import Optional

import tiktoken


class TokenCounter:
    """Count tokens and fit text parts into a token budget.

    Uses ``tiktoken`` for fast, local token counting.  Defaults to the
    ``cl100k_base`` encoding used by GPT-4 / GPT-3.5-turbo.
    """

    def __init__(self, encoding_name: str = "cl100k_base") -> None:
        self._encoding: tiktoken.Encoding = tiktoken.get_encoding(encoding_name)

    def count(self, text: str) -> int:
        """Return the number of tokens in *text*."""
        return len(self._encoding.encode(text))

    def fit_to_budget(self, parts: list[str], max_tokens: int) -> list[str]:
        """Select the largest prefix of *parts* whose total tokens fit in *max_tokens*.

        Uses a greedy left-to-right strategy: keep appending parts until the
        next one would exceed the budget.

        Args:
            parts: Ordered text segments to consider.
            max_tokens: Maximum allowed total tokens.

        Returns:
            A (possibly shorter) list of parts that fit within the budget.
        """
        result: list[str] = []
        used = 0
        for part in parts:
            tokens = self.count(part)
            if used + tokens > max_tokens:
                break
            result.append(part)
            used += tokens
        return result
