"""BM25 (Best Matching 25) ranking function.

Reference: Robertson et al., "Okapi at TREC-3", 1994.
"""

from __future__ import annotations

import math
from collections import Counter


class BM25:
    """Okapi BM25 scoring over a corpus of tokenized documents.

    Parameters
    ----------
    docs:
        A list of tokenized documents, where each document is a list of
        string tokens (words).
    k1:
        Term frequency saturation parameter.  Higher values give more weight
        to repeated terms.  Default ``1.5``.
    b:
        Document length normalisation parameter (0..1).  ``0`` disables
        length normalisation, ``1`` fully normalises.  Default ``0.75``.
    """

    def __init__(
        self,
        docs: list[list[str]],
        k1: float = 1.5,
        b: float = 0.75,
    ) -> None:
        self.k1 = k1
        self.b = b
        self.docs = docs
        self.n_docs = len(docs)

        # Document lengths and average
        self.doc_lens: list[int] = [len(doc) for doc in docs]
        self.avg_dl: float = (
            sum(self.doc_lens) / self.n_docs if self.n_docs else 0.0
        )

        # Inverted index: term -> number of documents containing it
        self.doc_freqs: Counter[str] = Counter()
        # Per-document term frequency maps
        self.term_freqs: list[Counter[str]] = []
        for doc in docs:
            tf = Counter(doc)
            self.term_freqs.append(tf)
            for term in tf:
                self.doc_freqs[term] += 1

    # -- IDF ----------------------------------------------------------------

    def _idf(self, term: str) -> float:
        """Inverse document frequency with floor at 0.

        Uses the formula::

            idf = log(1 + (N - n + 0.5) / (n + 0.5))

        where *N* is the total number of documents and *n* is the number of
        documents containing *term*.
        """
        n = self.doc_freqs.get(term, 0)
        return math.log(1.0 + (self.n_docs - n + 0.5) / (n + 0.5))

    # -- Scoring ------------------------------------------------------------

    def score(self, query: list[str], doc_index: int) -> float:
        """Compute the BM25 score of document *doc_index* against *query*.

        Parameters
        ----------
        query:
            Tokenised query (list of string tokens).
        doc_index:
            Index into ``self.docs`` of the document to score.

        Returns
        -------
        float
            BM25 relevance score (higher is more relevant).
        """
        tf = self.term_freqs[doc_index]
        dl = self.doc_lens[doc_index]
        score = 0.0

        for term in query:
            if term not in tf:
                continue
            term_freq = tf[term]
            idf = self._idf(term)
            numerator = term_freq * (self.k1 + 1.0)
            denominator = term_freq + self.k1 * (
                1.0 - self.b + self.b * dl / self.avg_dl
            )
            score += idf * numerator / denominator

        return score

    def get_scores(self, query: list[str]) -> list[float]:
        """Score every document in the corpus against *query*.

        Parameters
        ----------
        query:
            Tokenised query.

        Returns
        -------
        list[float]
            One score per document, in corpus order.
        """
        return [self.score(query, i) for i in range(self.n_docs)]
