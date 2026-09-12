"""Cross-encoder evidence sentence ranking with a pretrained relevance model."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
from sentence_transformers import CrossEncoder


DEFAULT_CROSS_ENCODER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"


class CrossEncoderEvidenceSelector:
    """Rank abstract sentences by jointly scoring each claim/sentence pair.

    The pretrained cross-encoder is used as-is. Its scores are only used for
    ranking sentences within an abstract; they are not interpreted as
    calibrated probabilities.
    """

    def __init__(
        self,
        model_name: str = DEFAULT_CROSS_ENCODER_MODEL,
        batch_size: int = 32,
        device: str | None = None,
    ) -> None:
        self.model = CrossEncoder(model_name, device=device)
        self.batch_size = batch_size

    def select(
        self,
        claim: str,
        abstract: Sequence[str] | str,
        top_k: int = 3,
    ) -> list[dict[str, Any]]:
        """Return the ``top_k`` jointly scored sentences from an abstract.

        Each result contains a zero-based ``sentence_id``, the original
        ``sentence``, and the model's relevance ``score``. Results are ordered
        by descending score, with original abstract order breaking ties.
        """
        if top_k <= 0 or not claim.strip():
            return []

        sentences = [abstract] if isinstance(abstract, str) else list(abstract)
        if not sentences:
            return []

        pairs = [(claim, str(sentence)) for sentence in sentences]
        scores = np.asarray(
            self.model.predict(
                pairs,
                batch_size=self.batch_size,
                show_progress_bar=False,
                convert_to_numpy=True,
            )
        ).reshape(-1)
        ranked_indices = np.argsort(-scores, kind="stable")[:top_k]
        return [
            {
                "sentence_id": int(index),
                "sentence": str(sentences[index]),
                "score": float(scores[index]),
            }
            for index in ranked_indices
        ]
