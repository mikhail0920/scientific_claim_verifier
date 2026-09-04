"""BM25 retrieval over the SciFact document corpus."""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any

from rank_bm25 import BM25Okapi


_TOKEN_PATTERN = re.compile(r"[\w]+", flags=re.UNICODE)


def _tokenize(text: str) -> list[str]:
    """Tokenize text for BM25 while keeping scientific terms intact."""
    return _TOKEN_PATTERN.findall(text.lower())


def _abstract_to_text(abstract: Any) -> str:
    """Convert either a list of abstract sentences or a string to text."""
    if isinstance(abstract, str):
        return abstract
    return " ".join(str(sentence) for sentence in (abstract or []))


@dataclass(frozen=True)
class RetrievalResult:
    """A ranked corpus document returned by :class:`BM25Retriever`."""

    doc_id: int | str
    score: float
    title: str
    abstract: list[str] | str

    def as_dict(self) -> dict[str, Any]:
        """Return a dictionary suitable for serialisation or display."""
        return {
            "doc_id": self.doc_id,
            "score": self.score,
            "title": self.title,
            "abstract": self.abstract,
        }


class BM25Retriever:
    """Retrieve the most relevant documents using Okapi BM25.

    ``corpus`` can be a Hugging Face ``Dataset`` or any iterable of mappings
    with ``doc_id``, ``title``, and ``abstract`` fields. The indexed text is
    ``title + abstract``.
    """

    def __init__(self, corpus: Iterable[Mapping[str, Any]]) -> None:
        self.documents = [dict(document) for document in corpus]
        if not self.documents:
            raise ValueError("BM25Retriever requires a non-empty corpus")

        self._texts = [
            f"{document.get('title', '')} {_abstract_to_text(document.get('abstract', ''))}"
            for document in self.documents
        ]
        self._bm25 = BM25Okapi([_tokenize(text) for text in self._texts])

    def search(self, query: str, top_k: int = 10) -> list[dict[str, Any]]:
        """Return up to ``top_k`` ranked documents for ``query``.

        Each result contains ``doc_id``, BM25 ``score``, ``title``, and
        ``abstract``. Results are ordered from highest to lowest score.
        """
        if top_k <= 0:
            return []
        if not query.strip():
            return []

        scores = self._bm25.get_scores(_tokenize(query))
        ranked_indices = sorted(
            range(len(self.documents)),
            key=lambda index: (-float(scores[index]), index),
        )[:top_k]
        return [
            RetrievalResult(
                doc_id=document["doc_id"],
                score=float(scores[index]),
                title=str(document.get("title", "")),
                abstract=document.get("abstract", ""),
            ).as_dict()
            for index in ranked_indices
            for document in [self.documents[index]]
        ]
