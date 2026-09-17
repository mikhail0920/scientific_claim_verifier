"""Binary claim verification using a pretrained three-way NLI model."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from sentence_transformers import CrossEncoder


DEFAULT_NLI_MODEL = "cross-encoder/nli-deberta-v3-xsmall"


class ClaimVerifier:
    """Classify a claim against evidence as SUPPORT or CONTRADICT.

    Pretrained NLI checkpoints map entailment/contradiction logits to the
    binary SciFact labels. Fine-tuned checkpoints expose the binary labels
    directly. In either case, the returned score is the selected-class
    probability after softmax over the model's full output layer.
    """

    def __init__(
        self,
        model_name: str = DEFAULT_NLI_MODEL,
        device: str | None = None,
    ) -> None:
        self.model = CrossEncoder(model_name, device=device)
        id2label = self.model.model.config.id2label
        normalized = {
            str(label).strip().lower(): int(index)
            for index, label in id2label.items()
        }
        self.contradiction_index = self._find_label_index(
            normalized, {"contradiction", "contradict"}
        )
        self.support_index = self._find_label_index(
            normalized, {"entailment", "entail", "support"}
        )
        if self.contradiction_index is None or self.support_index is None:


            if model_name == DEFAULT_NLI_MODEL:
                self.contradiction_index, self.support_index = 0, 1
            else:
                raise ValueError(
                    "The selected model must expose contradiction and "
                    "entailment labels in its id2label configuration"
                )

    @staticmethod
    def _find_label_index(labels: dict[str, int], candidates: set[str]) -> int | None:
        for label, index in labels.items():
            normalized = label.replace("label_", "").strip()
            if normalized in candidates:
                return index
        return None

    def predict(
        self,
        claim: str,
        evidence: Sequence[str] | str,
    ) -> tuple[str, float]:
        """Return a binary label and its probability from the model output.

        Evidence is used as the premise and the claim as the hypothesis.
        For multiple gold sentences, they are joined into one premise.
        """
        if not claim.strip():
            raise ValueError("claim must not be empty")
        if isinstance(evidence, str):
            premise = evidence.strip()
        else:
            premise = " ".join(str(sentence).strip() for sentence in evidence).strip()
        if not premise:
            raise ValueError("evidence must contain at least one sentence")

        logits = np.asarray(
            self.model.predict(
                [(premise, claim)],
                show_progress_bar=False,
                convert_to_numpy=True,
                apply_softmax=False,
            )
        ).reshape(-1)
        shifted = logits - np.max(logits)
        probabilities = np.exp(shifted) / np.exp(shifted).sum()

        if logits[self.support_index] >= logits[self.contradiction_index]:
            predicted_index = self.support_index
            label = "SUPPORT"
        else:
            predicted_index = self.contradiction_index
            label = "CONTRADICT"
        return label, float(probabilities[predicted_index])
