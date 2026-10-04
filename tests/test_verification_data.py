import pytest

from claim_verifier.verification.data import build_oracle_examples


def test_build_oracle_examples_combines_evidence_sentences() -> None:
    corpus = [
        {"doc_id": 10, "abstract": ["Background.", "First fact.", "Second fact."]}
    ]
    claims = [
        {
            "id": 7,
            "claim": "The treatment works.",
            "evidence_doc_id": 10,
            "evidence_label": "SUPPORT",
            "evidence_sentences": [2, 1],
        }
    ]

    assert build_oracle_examples(corpus, claims) == [
        {
            "claim_id": "7",
            "claim": "The treatment works.",
            "evidence": ["First fact.", "Second fact."],
            "label": "SUPPORT",
        }
    ]


def test_build_oracle_examples_rejects_unknown_labels() -> None:
    corpus = [{"doc_id": 10, "abstract": ["Evidence."]}]
    claims = [
        {
            "id": 7,
            "claim": "Claim",
            "evidence_doc_id": 10,
            "evidence_label": "UNKNOWN",
            "evidence_sentences": [0],
        }
    ]

    with pytest.raises(ValueError, match="Unexpected SciFact rationale label"):
        build_oracle_examples(corpus, claims)


def test_build_oracle_examples_rejects_invalid_sentence_index() -> None:
    corpus = [{"doc_id": 10, "abstract": ["Evidence."]}]
    claims = [
        {
            "id": 7,
            "claim": "Claim",
            "evidence_doc_id": 10,
            "evidence_label": "CONTRADICT",
            "evidence_sentences": [3],
        }
    ]

    with pytest.raises(ValueError, match="Invalid evidence sentence index"):
        build_oracle_examples(corpus, claims)
