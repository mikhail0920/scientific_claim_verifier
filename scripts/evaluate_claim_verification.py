"""Evaluate the NLI verifier with gold evidence sentences (oracle setting)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from datasets import load_from_disk


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data" / "scifact"
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from claim_verifier.verification import ClaimVerifier
from claim_verifier.verification.data import LABELS, build_oracle_examples


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
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        default="cross-encoder/nli-deberta-v3-xsmall",
        help="Hugging Face model id or local checkpoint directory",
    )
    args = parser.parse_args()
    corpus, claims = _load_data()
    examples = build_oracle_examples(corpus, claims)
    verifier = ClaimVerifier(model_name=args.model)
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
