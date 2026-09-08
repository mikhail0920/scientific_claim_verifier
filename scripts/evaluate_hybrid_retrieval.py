"""Evaluate hybrid RRF retrieval and inspect its BM25 recovery cases."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from claim_verifier.retrieval import BM25Retriever, HybridRetriever
from evaluate_retrieval import _gold_documents, _load_data, evaluate


def _hybrid_recovery_cases(
    bm25: BM25Retriever,
    hybrid: HybridRetriever,
    claims: Any,
    limit: int = 5,
) -> list[dict[str, Any]]:
    """Find claims where hybrid top-10 includes gold evidence and BM25 does not."""
    claim_texts: dict[str, str] = {}
    for row in claims:
        claim_texts.setdefault(str(row["id"]), str(row["claim"]))

    cases = []
    for claim_id, relevant_doc_ids in _gold_documents(claims).items():
        query = claim_texts[claim_id]
        bm25_ids = [
            str(result["doc_id"]) for result in bm25.search(query, top_k=10)
        ]
        hybrid_results = hybrid.search(query, top_k=10)
        hybrid_ranks = {
            str(result["doc_id"]): rank
            for rank, result in enumerate(hybrid_results, start=1)
        }
        recovered = relevant_doc_ids & hybrid_ranks.keys()
        if not recovered or relevant_doc_ids & set(bm25_ids):
            continue

        evidence_id = sorted(recovered)[0]
        document = next(
            result for result in hybrid_results if str(result["doc_id"]) == evidence_id
        )
        cases.append(
            {
                "claim": query,
                "doc_id": evidence_id,
                "title": document["title"],
                "hybrid_rank": hybrid_ranks[evidence_id],
            }
        )
        if len(cases) == limit:
            break
    return cases


def main() -> None:
    """Run the hybrid evaluation and print qualitative recovery examples."""
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    corpus, claims = _load_data()
    bm25 = BM25Retriever(corpus)
    hybrid = HybridRetriever(corpus, candidate_k=100, rrf_k=60)


    metrics = evaluate(hybrid, claims, top_k=10)

    print("Hybrid retrieval (RRF, candidate depth=100, rrf_k=60)")
    print(f"Recall@5:  {metrics['Recall@5']:.4f}")
    print(f"Recall@10: {metrics['Recall@10']:.4f}")
    print(f"MRR:       {metrics['MRR']:.4f}")

    cases = _hybrid_recovery_cases(bm25, hybrid, claims)
    print("\nHybrid top-10 evidence cases missed by BM25 top-10:")
    if not cases:
        print("None on this evaluation split.")
    for index, case in enumerate(cases, start=1):
        print(f"{index}. Claim: {case['claim']}")
        print(
            f"   Evidence doc {case['doc_id']} — {case['title']} "
            f"(hybrid rank {case['hybrid_rank']})"
        )


if __name__ == "__main__":
    main()
