"""Evaluate the NLI verifier with gold evidence sentences (oracle setting)."""

from __future__ import annotations

import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from datasets import load_from_disk


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data" / "scifact"
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from claim_verifier.verification import ClaimVerifier


LABELS = ["SUPPORT", "CONTRADICT"]


def _load_data() -> tuple[Any, Any]:
    """Load the SciFact corpus and validation claims."""
    try:
        corpus = load_from_disk(str(DATA_DIR / "corpus"))["train"]
        claims = load_from_disk(str(DATA_DIR / "claims"))["validation"]
    except FileNotFoundError as error:
        raise SystemExit(
            "SciFact is not downloaded. Run `python scripts/download_data.py` first."
        ) from error
    return corpus, claims


def _oracle_examples(corpus: Any, claims: Any) -> list[dict[str, Any]]:
    """Build one example per evidence-bearing claim using all gold sentences."""
    documents = {str(row["doc_id"]): row for row in corpus}
    claim_data: dict[str, dict[str, Any]] = {}
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


def evaluate(verifier: ClaimVerifier, examples: list[dict[str, Any]]) -> dict[str, Any]:
    """Compute accuracy, per-class F1, macro F1, and confusion matrix."""
    y_true = [example["label"] for example in examples]
    predictions = [
        verifier.predict(example["claim"], example["evidence"])[0]
        for example in examples
    ]
    per_class_f1 = f1_score(
        y_true,
        predictions,
        labels=LABELS,
        average=None,
        zero_division=0,
    )
    return {
        "accuracy": float(accuracy_score(y_true, predictions)),
        "macro_f1": float(f1_score(
            y_true,
            predictions,
            labels=LABELS,
            average="macro",
            zero_division=0,
        )),
        "f1_support": float(per_class_f1[0]),
        "f1_contradict": float(per_class_f1[1]),
        "confusion_matrix": confusion_matrix(y_true, predictions, labels=LABELS),
        "predictions": predictions,
    }


def main() -> None:
    """Run the binary oracle evidence evaluation and print metrics."""
    corpus, claims = _load_data()
    examples = _oracle_examples(corpus, claims)
    verifier = ClaimVerifier(model_name="cross-encoder/nli-deberta-v3-xsmall")
    results = evaluate(verifier, examples)

    print(
        "Claim verification (oracle gold evidence; "
        f"{len(examples)} claims, no NEI examples)"
    )
    print(f"Accuracy:       {results['accuracy']:.4f}")
    print(f"Macro F1:       {results['macro_f1']:.4f}")
    print(f"F1 SUPPORT:     {results['f1_support']:.4f}")
    print(f"F1 CONTRADICT:  {results['f1_contradict']:.4f}")
    print("Confusion matrix (rows=true, columns=predicted):")
    print("                 SUPPORT  CONTRADICT")
    matrix = results["confusion_matrix"]
    print(f"SUPPORT          {matrix[0, 0]:7d}  {matrix[0, 1]:10d}")
    print(f"CONTRADICT       {matrix[1, 0]:7d}  {matrix[1, 1]:10d}")


if __name__ == "__main__":
    main()
