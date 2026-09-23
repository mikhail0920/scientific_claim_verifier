"""FastAPI service for end-to-end scientific claim verification."""

from __future__ import annotations

import os
from contextlib import asynccontextmanager
from pathlib import Path
from threading import Lock
from typing import Literal

from datasets import load_from_disk
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field, field_validator

from claim_verifier.pipeline import ScientificClaimVerifier


PROJECT_ROOT = Path(__file__).resolve().parent
CORPUS_DIR = Path(
    os.environ.get("SCIFACT_CORPUS_DIR", PROJECT_ROOT / "data/scifact/corpus")
)
VERIFIER_DIR = Path(
    os.environ.get("VERIFIER_CHECKPOINT_DIR", PROJECT_ROOT / "checkpoints/verifier")
)


class VerifyRequest(BaseModel):
    """Request body for POST /verify."""

    claim: str = Field(min_length=1, max_length=1000)

    @field_validator("claim")
    @classmethod
    def strip_and_reject_blank(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("claim must not be blank")
        return value


class EvidenceResponse(BaseModel):
    sentence: str
    paper: str
    doc_id: str
    sentence_score: float
    retrieval_rank: int
    rrf_score: float


class VerifyResponse(BaseModel):
    verdict: Literal["SUPPORT", "CONTRADICT", "NEI"]
    confidence: float
    evidence: list[EvidenceResponse]


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load SciFact and all retrieval/verifier models once at service startup."""
    if not CORPUS_DIR.exists():
        raise RuntimeError(
            f"SciFact corpus is missing at {CORPUS_DIR}. "
            "Download it with `python scripts/download_data.py` before building the image."
        )
    if not VERIFIER_DIR.exists():
        raise RuntimeError(
            f"Verifier checkpoint is missing at {VERIFIER_DIR}. "
            "Create it with `python scripts/train_verifier.py` before building the image."
        )

    corpus = load_from_disk(str(CORPUS_DIR))["train"]
    app.state.pipeline = ScientificClaimVerifier(corpus, verifier_model=VERIFIER_DIR)
    app.state.verify_lock = Lock()
    yield
    app.state.pipeline = None


app = FastAPI(
    title="Scientific Claim Verifier API",
    description="Retrieve paper evidence and classify a scientific claim.",
    version="0.1.0",
    lifespan=lifespan,
)


@app.get("/health")
def health(request: Request) -> dict[str, str]:
    """Report whether the model pipeline finished loading."""
    status = "ready" if getattr(request.app.state, "pipeline", None) else "starting"
    return {"status": status}


@app.post("/verify", response_model=VerifyResponse)
def verify_claim(body: VerifyRequest, request: Request) -> VerifyResponse:
    """Return a verdict with selected evidence and source scores."""
    pipeline: ScientificClaimVerifier | None = getattr(request.app.state, "pipeline", None)
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Verification pipeline is not ready")

    try:
        with request.app.state.verify_lock:
            result = pipeline.verify(body.claim)
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error

    documents = {str(item["doc_id"]): item for item in result["documents"]}
    evidence = []
    for item in result["evidence"]:
        document = documents.get(str(item["doc_id"]), {})
        evidence.append(
            EvidenceResponse(
                sentence=item["sentence"],
                paper=item["title"],
                doc_id=str(item["doc_id"]),
                sentence_score=float(item["score"]),
                retrieval_rank=int(item["document_rank"]),
                rrf_score=float(document.get("score", 0.0)),
            )
        )
    return VerifyResponse(
        verdict=result["verdict"],
        confidence=float(result["confidence"]),
        evidence=evidence,
    )
