"""Evaluate cross-encoder evidence sentence ranking in oracle documents."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from claim_verifier.evidence import CrossEncoderEvidenceSelector, EvidenceSelector
from evaluate_evidence_selection import (
    _gold_claim_document_pairs,
    _load_data,
    evaluate,
)


def _fixed_biencoder_errors(
    bi_encoder: EvidenceSelector,
    cross_encoder: CrossEncoderEvidenceSelector,
    corpus: Any,
    claims: Any,
    limit: int = 10,
) -> list[dict[str, Any]]:
    """Find pairs where cross-encoder top-3 retrieves more gold sentences."""
    documents = {str(row["doc_id"]): row for row in corpus}
    examples = []
    for (_, doc_id), pair in _gold_claim_document_pairs(claims).items():
        document = documents[doc_id]
        abstract = document["abstract"]
        gold_ids = pair["gold_sentence_ids"]
        bi_results = bi_encoder.select(pair["claim"], abstract, top_k=3)
        cross_results = cross_encoder.select(pair["claim"], abstract, top_k=3)
        bi_ids = {item["sentence_id"] for item in bi_results}
        cross_ids = {item["sentence_id"] for item in cross_results}
        bi_correct = bi_ids & gold_ids
        cross_correct = cross_ids & gold_ids
        if len(cross_correct) <= len(bi_correct):
            continue

        sentences = [abstract] if isinstance(abstract, str) else list(abstract)
        examples.append(
            {
                "claim": pair["claim"],
                "title": document["title"],
                "bi_correct": len(bi_correct),
                "cross_correct": len(cross_correct),
                "gold_sentences": [sentences[index] for index in sorted(cross_correct)],
                "bi_top3": [item["sentence"] for item in bi_results],
                "cross_top3": [item["sentence"] for item in cross_results],
            }
        )
        if len(examples) >= limit:
            break
    return examples


def main() -> None:
    """Run cross-encoder evaluation and print qualitative corrections."""
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    corpus, claims = _load_data()
    model_name = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    cross_encoder = CrossEncoderEvidenceSelector(model_name=model_name)
    metrics = evaluate(cross_encoder, corpus, claims)

    print(f"Cross-encoder evidence selection (oracle documents; {model_name})")
    for cutoff in (1, 3):
        print(f"Precision@{cutoff}: {metrics[f'Precision@{cutoff}']:.4f}")
        print(f"Recall@{cutoff}:    {metrics[f'Recall@{cutoff}']:.4f}")
        print(f"F1@{cutoff}:        {metrics[f'F1@{cutoff}']:.4f}")

    bi_encoder = EvidenceSelector(model_name="sentence-transformers/all-MiniLM-L6-v2")
    examples = _fixed_biencoder_errors(bi_encoder, cross_encoder, corpus, claims)
    print("\nExamples where cross-encoder top-3 contains more gold sentences than bi-encoder:")
    if not examples:
        print("No corrections found on this validation split.")
    for index, example in enumerate(examples, start=1):
        print(
            f"{index}. Claim: {example['claim']}\n"
            f"   Document: {example['title']}\n"
            f"   Gold sentences in top-3: bi-encoder "
            f"{example['bi_correct']}, cross-encoder {example['cross_correct']}"
        )
        for sentence in example["gold_sentences"]:
            print(f"   Recovered evidence: {sentence}")


if __name__ == "__main__":
    main()
