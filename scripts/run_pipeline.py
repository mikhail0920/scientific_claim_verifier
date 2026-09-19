"""Run end-to-end claim verification on a single claim."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from datasets import load_from_disk


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data" / "scifact"
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from claim_verifier.pipeline import ScientificClaimVerifier


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("claim", help="Scientific claim to verify")
    parser.add_argument(
        "--verifier-model", default=str(PROJECT_ROOT / "checkpoints" / "verifier")
    )
    args = parser.parse_args()

    try:
        corpus = load_from_disk(str(DATA_DIR / "corpus"))["train"]
    except FileNotFoundError as error:
        raise SystemExit(
            "SciFact is not downloaded. Run `python scripts/download_data.py` first."
        ) from error

    pipeline = ScientificClaimVerifier(corpus, verifier_model=args.verifier_model)
    result = pipeline.verify(args.claim)

    print(f"Claim: {result['claim']}\n")
    print(f"Verdict: {result['verdict']}")
    print(f"Confidence: {result['confidence']:.2f}\n")
    print("Evidence:")
    if not result["evidence"]:
        print("No evidence sentences were found.")
    for index, evidence in enumerate(result["evidence"], start=1):
        print(f'{index}. "{evidence["sentence"]}"')
        print(f'   Paper: {evidence["title"]} (doc_id: {evidence["doc_id"]})')


if __name__ == "__main__":
    main()
