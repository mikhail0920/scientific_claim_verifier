import pytest

from claim_verifier.retrieval.bm25 import BM25Retriever


def test_bm25_ranks_matching_document_first() -> None:
    corpus = [
        {
            "doc_id": 1,
            "title": "Vitamin D and respiratory infections",
            "abstract": ["Vitamin D supplementation reduced infection risk."],
        },
        {
            "doc_id": 2,
            "title": "Marine ecosystems",
            "abstract": ["Coral reefs support diverse fish populations."],
        },
    ]

    results = BM25Retriever(corpus).search("vitamin D infection", top_k=2)

    assert [item["doc_id"] for item in results] == [1, 2]
    assert results[0]["score"] >= results[1]["score"]


def test_bm25_handles_empty_queries_and_limits() -> None:
    retriever = BM25Retriever(
        [{"doc_id": "paper", "title": "Title", "abstract": "Abstract"}]
    )

    assert retriever.search("", top_k=10) == []
    assert retriever.search("title", top_k=0) == []
    assert len(retriever.search("title", top_k=10)) == 1


def test_bm25_rejects_empty_corpus() -> None:
    with pytest.raises(ValueError, match="non-empty corpus"):
        BM25Retriever([])
