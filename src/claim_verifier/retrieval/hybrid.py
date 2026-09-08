"""Hybrid BM25 and dense retrieval using Reciprocal Rank Fusion."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

from .bm25 import BM25Retriever
from .dense import DEFAULT_MODEL_NAME, DenseRetriever


class HybridRetriever:
    """Fuse BM25 and dense rankings without mixing their score scales.

    Each component contributes its first ``candidate_k`` results. Documents
    are scored with ``1 / (rrf_k + rank)`` for each list in which they occur,
    where ranks start at one.
    """

    def __init__(
        self,
        corpus: Iterable[Mapping[str, Any]],
        model_name: str = DEFAULT_MODEL_NAME,
        candidate_k: int = 100,
        rrf_k: int = 60,
        device: str | None = None,
    ) -> None:
        if candidate_k <= 0:
            raise ValueError("candidate_k must be positive")
        if rrf_k <= 0:
            raise ValueError("rrf_k must be positive")

        self.bm25 = BM25Retriever(corpus)
        self.dense = DenseRetriever(
            self.bm25.documents,
            model_name=model_name,
            device=device,
        )
        self.candidate_k = candidate_k
        self.rrf_k = rrf_k

    def search(self, query: str, top_k: int = 10) -> list[dict[str, Any]]:
        """Return documents ranked by their fused reciprocal rank score."""
        if top_k <= 0 or not query.strip():
            return []

        bm25_results = self.bm25.search(query, top_k=self.candidate_k)
        dense_results = self.dense.search(query, top_k=self.candidate_k)
        fused: dict[str, dict[str, Any]] = {}

        for retriever_name, results in (
            ("bm25", bm25_results),
            ("dense", dense_results),
        ):
            for rank, result in enumerate(results, start=1):
                doc_id = str(result["doc_id"])
                entry = fused.setdefault(
                    doc_id,
                    {
                        "doc_id": result["doc_id"],
                        "title": result.get("title", ""),
                        "abstract": result.get("abstract", ""),
                        "score": 0.0,
                        "bm25_rank": None,
                        "dense_rank": None,
                    },
                )
                entry["score"] += 1.0 / (self.rrf_k + rank)
                entry[f"{retriever_name}_rank"] = rank

        ranked = sorted(
            fused.values(),
            key=lambda result: (-result["score"], str(result["doc_id"])),
        )
        return ranked[:top_k]
