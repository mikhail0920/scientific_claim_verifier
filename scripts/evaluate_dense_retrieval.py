"""Evaluate the dense retrieval baseline on SciFact validation claims."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from claim_verifier.retrieval import DenseRetriever
from evaluate_retrieval import _load_data, evaluate


def main() -> None:
    """Encode the corpus once, evaluate all claims, and print the metrics."""
    corpus, claims = _load_data()
    retriever = DenseRetriever(
        corpus,
        model_name="sentence-transformers/all-MiniLM-L6-v2",
    )
    metrics = evaluate(retriever, claims)

    print("Dense retrieval baseline")
    print(f"Recall@5:  {metrics['Recall@5']:.4f}")
    print(f"Recall@10: {metrics['Recall@10']:.4f}")
    print(f"MRR:       {metrics['MRR']:.4f}")


if __name__ == "__main__":
    main()
