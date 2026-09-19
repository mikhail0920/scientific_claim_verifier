"""Evaluate oracle and end-to-end verdicts on SciFact validation claims."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

from datasets import load_from_disk
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data" / "scifact"
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from claim_verifier.pipeline import ScientificClaimVerifier
from claim_verifier.verification.data import LABELS, build_oracle_examples
from evaluate_claim_verification import evaluate


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--verifier-model", default=str(PROJECT_ROOT / "checkpoints" / "verifier")
    )
    args = parser.parse_args()

    try:
        corpus = load_from_disk(str(DATA_DIR / "corpus"))["train"]
        claims = load_from_disk(str(DATA_DIR / "claims"))["validation"]
    except FileNotFoundError as error:
        raise SystemExit(
            "SciFact is not downloaded. Run `python scripts/download_data.py` first."
        ) from error

    examples = build_oracle_examples(corpus, claims)
    pipeline = ScientificClaimVerifier(corpus, verifier_model=args.verifier_model)
    oracle = evaluate(pipeline.verifier, examples)

    gold = [example["label"] for example in examples]
    predictions: list[str] = []
    no_evidence = 0
    for index, example in enumerate(examples, start=1):
        result = pipeline.verify(example["claim"])
        predictions.append(result["verdict"])
        no_evidence += not bool(result["evidence"])
        print(f"Evaluated {index}/{len(examples)} claims", end="\r", flush=True)

    predicted_metrics = {
        "accuracy": float(accuracy_score(gold, predictions)),
        "macro_f1": float(
            f1_score(gold, predictions, labels=LABELS, average="macro", zero_division=0)
        ),
        "confusion_matrix": confusion_matrix(gold, predictions, labels=LABELS),
    }
    print(" " * 60)
    print(f"Validation claims: {len(examples)}")
    print(f"Oracle evidence Macro F1:     {oracle['macro_f1']:.4f}")
    print(f"Predicted evidence Macro F1:  {predicted_metrics['macro_f1']:.4f}")
    print(f"Predicted evidence Accuracy: {predicted_metrics['accuracy']:.4f}")
    print(f"Claims with no selected evidence: {no_evidence}")
    matrix = predicted_metrics["confusion_matrix"]
    print("Confusion matrix (rows=true, columns=predicted):")
    print("                 SUPPORT  CONTRADICT")
    print(f"SUPPORT          {matrix[0, 0]:7d}  {matrix[0, 1]:10d}")
    print(f"CONTRADICT       {matrix[1, 0]:7d}  {matrix[1, 1]:10d}")


if __name__ == "__main__":
    main()
