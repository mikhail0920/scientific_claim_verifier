"""Evaluate pretrained cosine sentence selection in the oracle document setting."""

from __future__ import annotations

import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from datasets import load_from_disk


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data" / "scifact"
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from claim_verifier.evidence import EvidenceSelector


def _load_data() -> tuple[Any, Any]:
    """Load the SciFact corpus and annotated validation claims."""
    try:
        corpus = load_from_disk(str(DATA_DIR / "corpus"))["train"]
        claims = load_from_disk(str(DATA_DIR / "claims"))["validation"]
    except FileNotFoundError as error:
        raise SystemExit(
            "SciFact is not downloaded. Run `python scripts/download_data.py` first."
        ) from error
    return corpus, claims


def _gold_claim_document_pairs(claims: Any) -> dict[tuple[str, str], dict[str, Any]]:
    """Group flattened SciFact rows into claim/document pairs and rationale IDs."""
    pairs: dict[tuple[str, str], dict[str, Any]] = {}
    for row in claims:
        doc_id = str(row["evidence_doc_id"])
        if not doc_id:
            continue

        key = (str(row["id"]), doc_id)
        pair = pairs.setdefault(
            key,
            {"claim": str(row["claim"]), "gold_sentence_ids": set()},
        )
        pair["gold_sentence_ids"].update(int(index) for index in row["evidence_sentences"])
    return pairs


def evaluate(
    selector: EvidenceSelector,
    corpus: Any,
    claims: Any,
    cutoffs: tuple[int, ...] = (1, 3),
) -> dict[str, float]:
    """Macro-average sentence P/R/F1 over oracle claim/document pairs."""
    documents = {str(row["doc_id"]): row for row in corpus}
    pairs = _gold_claim_document_pairs(claims)
    if not pairs:
        raise ValueError("No annotated evidence claim/document pairs found")

    scores_by_metric: dict[str, list[float]] = {
        f"{metric}@{cutoff}": []
        for cutoff in cutoffs
        for metric in ("Precision", "Recall", "F1")
    }

    for (_, doc_id), pair in pairs.items():
        document = documents.get(doc_id)
        if document is None:
            raise ValueError(f"Evidence document {doc_id} is missing from the corpus")
        abstract = document["abstract"]
        sentence_count = len(abstract) if not isinstance(abstract, str) else 1
        gold = {
            sentence_id
            for sentence_id in pair["gold_sentence_ids"]
            if 0 <= sentence_id < sentence_count
        }
        if not gold:
            continue

        max_cutoff = max(cutoffs)
        selected = selector.select(pair["claim"], abstract, top_k=max_cutoff)
        ranked_ids = [result["sentence_id"] for result in selected]
        for cutoff in cutoffs:
            predicted = set(ranked_ids[:cutoff])
            true_positives = len(predicted & gold)
            precision = true_positives / cutoff
            recall = true_positives / len(gold)
            f1 = (
                2 * precision * recall / (precision + recall)
                if precision + recall
                else 0.0
            )
            scores_by_metric[f"Precision@{cutoff}"].append(precision)
            scores_by_metric[f"Recall@{cutoff}"].append(recall)
            scores_by_metric[f"F1@{cutoff}"].append(f1)

    if not scores_by_metric["Precision@1"]:
        raise ValueError("No valid evidence sentence annotations found")
    return {
        metric: float(np.mean(values))
        for metric, values in scores_by_metric.items()
    }


def main() -> None:
    """Run oracle evidence sentence selection and print its metrics."""
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    corpus, claims = _load_data()
    selector = EvidenceSelector(model_name="sentence-transformers/all-MiniLM-L6-v2")
    metrics = evaluate(selector, corpus, claims)
    pair_count = len(_gold_claim_document_pairs(claims))

    print(f"Evidence sentence selection (oracle documents, {pair_count} claim-document pairs)")
    for cutoff in (1, 3):
        print(f"Precision@{cutoff}: {metrics[f'Precision@{cutoff}']:.4f}")
        print(f"Recall@{cutoff}:    {metrics[f'Recall@{cutoff}']:.4f}")
        print(f"F1@{cutoff}:        {metrics[f'F1@{cutoff}']:.4f}")


if __name__ == "__main__":
    main()
