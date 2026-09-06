"""Dense retrieval with a pretrained SentenceTransformer encoder."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

import numpy as np
from sentence_transformers import SentenceTransformer

from .bm25 import _abstract_to_text


DEFAULT_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


class DenseRetriever:
    """Retrieve corpus documents by cosine similarity to an encoded claim.

    Corpus documents are encoded once during construction. Both document and
    query embeddings are L2-normalized, so their dot product is cosine
    similarity.
    """

    def __init__(
        self,
        corpus: Iterable[Mapping[str, Any]],
        model_name: str = DEFAULT_MODEL_NAME,
        batch_size: int = 32,
        device: str | None = None,
    ) -> None:
        self.documents = [dict(document) for document in corpus]
        if not self.documents:
            raise ValueError("DenseRetriever requires a non-empty corpus")

        self.model_name = model_name
        self.model = SentenceTransformer(model_name, device=device)
        self._texts = [
            f"{document.get('title', '')} {_abstract_to_text(document.get('abstract', ''))}"
            for document in self.documents
        ]
        self.document_embeddings = self.model.encode(
            self._texts,
            batch_size=batch_size,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=True,
        )

    def search(self, query: str, top_k: int = 10) -> list[dict[str, Any]]:
        """Return up to ``top_k`` documents ranked by cosine similarity."""
        if top_k <= 0 or not query.strip():
            return []

        query_embedding = self.model.encode(
            [query],
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )[0]
        scores = np.asarray(self.document_embeddings @ query_embedding).ravel()
        ranked_indices = np.argsort(-scores, kind="stable")[:top_k]
        return [
            {
                "doc_id": self.documents[index]["doc_id"],
                "score": float(scores[index]),
                "title": str(self.documents[index].get("title", "")),
                "abstract": self.documents[index].get("abstract", ""),
            }
            for index in ranked_indices
        ]
