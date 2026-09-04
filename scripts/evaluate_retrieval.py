"""Evaluate the BM25 retrieval baseline on SciFact validation claims."""

from __future__ import annotations

import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

from datasets import load_from_disk


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data" / "scifact"
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from claim_verifier.retrieval import BM25Retriever


def _load_data() -> tuple[Any, Any]:
    """Load the saved SciFact corpus and validation claims."""
    try:
        corpus = load_from_disk(str(DATA_DIR / "corpus"))["train"]
        claims = load_from_disk(str(DATA_DIR / "claims"))["validation"]
    except FileNotFoundError as error:
        raise SystemExit(
            "SciFact is not downloaded. Run `python scripts/download_data.py` first."
        ) from error
    return corpus, claims


def _gold_documents(claims: Any) -> dict[str, set[str]]:
    """Group all annotated evidence document IDs by claim ID."""
    gold: dict[str, set[str]] = defaultdict(set)
    for row in claims:
        doc_id = str(row["evidence_doc_id"])
        if doc_id:
            gold[str(row["id"])].add(doc_id)
    return dict(gold)


def evaluate(retriever: BM25Retriever, claims: Any, top_k: int = 10) -> dict[str, float]:
    """Compute Recall@5, Recall@10, and MRR for evidence-bearing claims."""
    gold = _gold_documents(claims)
    if not gold:
        raise ValueError("No evidence-bearing claims found")

    hits_at_5 = 0
    hits_at_10 = 0
    reciprocal_rank_sum = 0.0
    for claim_id, relevant_doc_ids in gold.items():
        claim_rows = [row for row in claims if str(row["id"]) == claim_id]
        query = str(claim_rows[0]["claim"])
        results = retriever.search(query, top_k=top_k)
        ranked_ids = [str(result["doc_id"]) for result in results]
        hits_at_5 += bool(set(ranked_ids[:5]) & relevant_doc_ids)
        hits_at_10 += bool(set(ranked_ids[:10]) & relevant_doc_ids)
        for rank, doc_id in enumerate(ranked_ids, start=1):
            if doc_id in relevant_doc_ids:
                reciprocal_rank_sum += 1.0 / rank
                break

    count = len(gold)
    return {
        "Recall@5": hits_at_5 / count,
        "Recall@10": hits_at_10 / count,
        "MRR": reciprocal_rank_sum / count,
    }


def main() -> None:
    """Run and print the BM25 baseline metrics."""
    corpus, claims = _load_data()
    retriever = BM25Retriever(corpus)
    metrics = evaluate(retriever, claims)

    print("BM25 baseline")
    print(f"Recall@5:  {metrics['Recall@5']:.4f}")
    print(f"Recall@10: {metrics['Recall@10']:.4f}")
    print(f"MRR:       {metrics['MRR']:.4f}")


if __name__ == "__main__":
    main()
