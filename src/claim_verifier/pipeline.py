"""End-to-end scientific claim verification pipeline."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from .evidence.cross_encoder import (
    DEFAULT_CROSS_ENCODER_MODEL,
    CrossEncoderEvidenceSelector,
)
from .retrieval.dense import DEFAULT_MODEL_NAME
from .retrieval.hybrid import HybridRetriever
from .verification.verifier import ClaimVerifier


class ScientificClaimVerifier:
    """Retrieve papers, rank their evidence sentences, and classify a claim."""

    def __init__(
        self,
        corpus: Iterable[Mapping[str, Any]],
        verifier_model: str | Path = "checkpoints/verifier",
        retrieval_model: str = DEFAULT_MODEL_NAME,
        evidence_model: str = DEFAULT_CROSS_ENCODER_MODEL,
        top_documents: int = 10,
        top_evidence: int = 3,
        candidate_k: int = 100,
        rrf_k: int = 60,
        device: str | None = None,
    ) -> None:
        if top_documents <= 0 or top_evidence <= 0:
            raise ValueError("top_documents and top_evidence must be positive")

        model_path = Path(verifier_model)
        if model_path.exists():
            model_name = str(model_path.resolve())
        elif model_path.name == "verifier" and model_path.parent.name == "checkpoints":
            raise FileNotFoundError(
                "Fine-tuned verifier checkpoint is missing. Run "
                "`python scripts/train_verifier.py` first, or pass --verifier-model."
            )
        else:
            model_name = str(verifier_model)

        self.retriever = HybridRetriever(
            corpus,
            model_name=retrieval_model,
            candidate_k=candidate_k,
            rrf_k=rrf_k,
            device=device,
        )
        self.evidence_selector = CrossEncoderEvidenceSelector(
            model_name=evidence_model,
            device=device,
        )
        self.verifier = ClaimVerifier(model_name=model_name, device=device)
        self.top_documents = top_documents
        self.top_evidence = top_evidence

    def verify(self, claim: str) -> dict[str, Any]:
        """Return verdict, confidence, selected evidence, and retrieved papers."""
        if not claim.strip():
            raise ValueError("claim must not be empty")

        documents = self.retriever.search(claim, top_k=self.top_documents)
        evidence = self.evidence_selector.select_from_documents(
            claim,
            documents,
            top_k=self.top_evidence,
        )
        if not evidence:
            return {
                "claim": claim,
                "verdict": "NEI",
                "confidence": 0.0,
                "evidence": [],
                "documents": documents,
            }

        verdict, confidence = self.verifier.predict(
            claim,
            [item["sentence"] for item in evidence],
        )
        return {
            "claim": claim,
            "verdict": verdict,
            "confidence": confidence,
            "evidence": evidence,
            "documents": documents,
        }
