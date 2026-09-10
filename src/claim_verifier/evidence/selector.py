"""Select evidence sentences with a pretrained sentence embedding model."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
from sentence_transformers import SentenceTransformer

from claim_verifier.retrieval.dense import DEFAULT_MODEL_NAME


class EvidenceSelector:
    """Rank abstract sentences by cosine similarity to a scientific claim.

    The model is pretrained and is not fine-tuned. Sentence and claim
    embeddings are normalized, making their dot product cosine similarity.
    """

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL_NAME,
        device: str | None = None,
    ) -> None:
        self.model = SentenceTransformer(model_name, device=device)

    def select(
        self,
        claim: str,
        abstract: Sequence[str] | str,
        top_k: int = 3,
    ) -> list[dict[str, Any]]:
        """Return the ``top_k`` abstract sentences most similar to ``claim``.

        Each result has its zero-based ``sentence_id``, original ``sentence``,
        and cosine ``score``. Returned sentences are ordered by descending
        similarity, with original abstract order breaking ties.
        """
        if top_k <= 0 or not claim.strip():
            return []

        sentences = [abstract] if isinstance(abstract, str) else list(abstract)
        if not sentences:
            return []

        embeddings = self.model.encode(
            [claim, *sentences],
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        scores = np.asarray(embeddings[1:] @ embeddings[0]).ravel()
        ranked_indices = np.argsort(-scores, kind="stable")[:top_k]
        return [
            {
                "sentence_id": int(index),
                "sentence": str(sentences[index]),
                "score": float(scores[index]),
            }
            for index in ranked_indices
        ]
