"""Download the SciFact dataset from the Hugging Face Hub.

The downloaded DatasetDict objects are stored under ``data/scifact`` and are
ignored by Git. Run this script from any working directory:

    python scripts/download_data.py
"""

import shutil
import json
import tarfile
import urllib.request
from pathlib import Path

from datasets import Dataset, DatasetDict, load_dataset


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data" / "scifact"
RELEASE_URL = "https://scifact.s3.amazonaws.com/release/latest/data.tar.gz"


def _load_from_release() -> tuple[object, object]:
    """Load SciFact JSONL files when the Hub's legacy loader is unavailable."""
    raw_dir = DATA_DIR / "raw"
    archive_path = raw_dir / "data.tar.gz"
    extract_dir = raw_dir / "extracted"
    raw_dir.mkdir(parents=True, exist_ok=True)

    if not archive_path.exists():
        print(f"Downloading the SciFact release archive from {RELEASE_URL}...")
        urllib.request.urlretrieve(RELEASE_URL, archive_path)
    if not extract_dir.exists():
        extract_dir.mkdir()
        with tarfile.open(archive_path, "r:gz") as archive:
            archive.extractall(extract_dir)

    data_dir = extract_dir / "data"
    def read_claims(path: Path, split: str) -> Dataset:
        records = []
        with path.open(encoding="utf-8") as claims_file:
            for row in claims_file:
                claim = json.loads(row)
                evidences = claim.get("evidence", {})
                if split == "test" or not evidences:
                    records.append(
                        {
                            "id": claim["id"],
                            "claim": claim["claim"],
                            "evidence_doc_id": "",
                            "evidence_label": "",
                            "evidence_sentences": [],
                            "cited_doc_ids": claim.get("cited_doc_ids", []),
                        }
                    )
                    continue
                for doc_id, doc_evidence in evidences.items():
                    for evidence in doc_evidence:
                        records.append(
                            {
                                "id": claim["id"],
                                "claim": claim["claim"],
                                "evidence_doc_id": doc_id,
                                "evidence_label": evidence["label"],
                                "evidence_sentences": evidence["sentences"],
                                "cited_doc_ids": claim.get("cited_doc_ids", []),
                            }
                        )
        return Dataset.from_list(records)

    claims = DatasetDict(
        {
            "train": read_claims(data_dir / "claims_train.jsonl", "train"),
            "validation": read_claims(data_dir / "claims_dev.jsonl", "validation"),
            "test": read_claims(data_dir / "claims_test.jsonl", "test"),
        }
    )
    corpus = load_dataset("json", data_files={"train": str(data_dir / "corpus.jsonl")})
    return claims, corpus


def main() -> None:
    """Download and persist both SciFact configurations."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    try:
        claims = load_dataset("allenai/scifact", "claims", trust_remote_code=True)
        corpus = load_dataset("allenai/scifact", "corpus", trust_remote_code=True)
    except Exception as error:
        print(f"The Hub loader was unavailable ({error}); using the SciFact release archive.")
        claims, corpus = _load_from_release()

    for config, dataset in (("claims", claims), ("corpus", corpus)):
        output_dir = DATA_DIR / config
        if output_dir.exists():
            shutil.rmtree(output_dir)
        dataset.save_to_disk(str(output_dir))
        print(f"Saved SciFact/{config} to {output_dir}")


if __name__ == "__main__":
    main()
