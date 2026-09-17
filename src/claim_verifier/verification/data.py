"""Prepare SciFact claims with their annotated evidence sentences."""

from __future__ import annotations

from collections import defaultdict
from typing import Any


LABELS = ["SUPPORT", "CONTRADICT"]


def build_oracle_examples(corpus: Any, claims: Any) -> list[dict[str, Any]]:
    """Build one example per rationale-bearing claim using all gold sentences."""
    documents = {str(row["doc_id"]): row for row in corpus}
    claim_data: dict[str, dict[str, str]] = {}
    sentence_refs: dict[str, set[tuple[str, int]]] = defaultdict(set)
    labels_by_claim: dict[str, set[str]] = defaultdict(set)

    for row in claims:
        claim_id = str(row["id"])
        doc_id = str(row["evidence_doc_id"])
        label = str(row["evidence_label"])
        if not doc_id or not label:
            continue
        if label not in LABELS:
            raise ValueError(f"Unexpected SciFact rationale label: {label}")

        claim_data.setdefault(claim_id, {"claim": str(row["claim"])})
        labels_by_claim[claim_id].add(label)
        sentence_refs[claim_id].update(
            (doc_id, int(sentence_id)) for sentence_id in row["evidence_sentences"]
        )

    examples = []
    for claim_id, data in claim_data.items():
        claim_labels = labels_by_claim[claim_id]
        if len(claim_labels) != 1:
            raise ValueError(
                f"Claim {claim_id} has conflicting rationale labels: {claim_labels}"
            )

        evidence_sentences = []
        for doc_id, sentence_id in sorted(
            sentence_refs[claim_id], key=lambda ref: (int(ref[0]), ref[1])
        ):
            if doc_id not in documents:
                raise ValueError(f"Evidence document {doc_id} is missing from corpus")
            abstract = documents[doc_id]["abstract"]
            if sentence_id < 0 or sentence_id >= len(abstract):
                raise ValueError(
                    f"Invalid evidence sentence index {sentence_id} for document {doc_id}"
                )
            evidence_sentences.append(str(abstract[sentence_id]))

        if evidence_sentences:
            examples.append(
                {
                    "claim_id": claim_id,
                    "claim": data["claim"],
                    "evidence": evidence_sentences,
                    "label": next(iter(claim_labels)),
                }
            )
    return examples
