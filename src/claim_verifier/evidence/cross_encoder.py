"""Cross-encoder evidence sentence ranking with a pretrained relevance model."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
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

    def select_from_documents(
        self,
        claim: str,
        documents: Sequence[Mapping[str, Any]],
        top_k: int = 3,
    ) -> list[dict[str, Any]]:
        """Rank sentences jointly across retrieved documents in one batch."""
        if top_k <= 0 or not claim.strip():
            return []

        sentence_rows: list[dict[str, Any]] = []
        pairs: list[tuple[str, str]] = []
        for document_rank, document in enumerate(documents):
            abstract = document.get("abstract", [])
            sentences = [abstract] if isinstance(abstract, str) else list(abstract or [])
            for sentence_id, sentence in enumerate(sentences):
                text = str(sentence).strip()
                if not text:
                    continue
                pairs.append((claim, text))
                sentence_rows.append(
                    {
                        "doc_id": document["doc_id"],
                        "title": str(document.get("title", "")),
                        "document_rank": document_rank + 1,
                        "sentence_id": sentence_id,
                        "sentence": text,
                    }
                )
        if not pairs:
            return []

        scores = np.asarray(
            self.model.predict(
                pairs,
                batch_size=self.batch_size,
                show_progress_bar=False,
                convert_to_numpy=True,
            )
        ).reshape(-1)
        ranked_indices = sorted(
            range(len(sentence_rows)),
            key=lambda index: (
                -float(scores[index]),
                sentence_rows[index]["document_rank"],
                sentence_rows[index]["sentence_id"],
            ),
        )[:top_k]
        return [
            {**sentence_rows[index], "score": float(scores[index])}
            for index in ranked_indices
        ]
