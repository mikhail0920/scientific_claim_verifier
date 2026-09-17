"""Fine-tune the SciFact verifier using train and validation splits."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from datasets import load_from_disk


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data" / "scifact"
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from claim_verifier.verification.data import build_oracle_examples
from claim_verifier.verification.train import train_verifier


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="cross-encoder/nli-deberta-v3-xsmall")
    parser.add_argument("--output-dir", default=str(PROJECT_ROOT / "checkpoints" / "verifier"))
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    try:
        corpus = load_from_disk(str(DATA_DIR / "corpus"))["train"]
        claims = load_from_disk(str(DATA_DIR / "claims"))
        train_examples = build_oracle_examples(corpus, claims["train"])
        validation_examples = build_oracle_examples(corpus, claims["validation"])
    except FileNotFoundError as error:
        raise SystemExit(
            "SciFact is not downloaded. Run `python scripts/download_data.py` first."
        ) from error

    print(
        f"Training on {len(train_examples)} train claims; "
        f"selecting checkpoints on {len(validation_examples)} validation claims."
    )
    history = train_verifier(
        train_examples,
        validation_examples,
        output_dir=args.output_dir,
        model_name=args.model,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        max_length=args.max_length,
        seed=args.seed,
    )
    best = max(history, key=lambda metrics: metrics["macro_f1"])
    print(
        f"Best checkpoint: {args.output_dir} (epoch {int(best['epoch'])})\n"
        f"Validation accuracy: {best['accuracy']:.4f}\n"
        f"Validation Macro F1: {best['macro_f1']:.4f}"
    )


if __name__ == "__main__":
    main()
