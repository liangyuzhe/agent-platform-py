"""Token counting utilities with tiktoken."""

from __future__ import annotations

import logging
import re

import tiktoken

logger = logging.getLogger(__name__)


class TokenCounter:
    """Count tokens and fit text parts into a token budget.

    Uses ``tiktoken`` for fast, local token counting.  Defaults to the
    ``cl100k_base`` encoding used by GPT-4 / GPT-3.5-turbo.
    """

    def __init__(self, encoding_name: str = "cl100k_base") -> None:
        try:
            self._encoding: tiktoken.Encoding | None = tiktoken.get_encoding(encoding_name)
        except Exception as e:
            logger.warning("tiktoken encoding %s unavailable, using fallback counter: %s", encoding_name, e)
            self._encoding = None

    def count(self, text: str) -> int:
        """Return the number of tokens in *text*."""
        if self._encoding is None:
            return _fallback_count(text)
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


def _fallback_count(text: str) -> int:
    """Approximate token count without external tiktoken assets."""
    count = 0
    for token in re.findall(r"[\u4e00-\u9fff]|[A-Za-z0-9_]+|[^\s]", text or ""):
        if re.fullmatch(r"[A-Za-z0-9_]+", token):
            count += 1
        else:
            count += 1
    return count
