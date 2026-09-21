"""Interactive Streamlit demo for scientific claim verification."""

from __future__ import annotations

from html import escape
import sys
from pathlib import Path
from typing import Any

import streamlit as st
from datasets import load_from_disk


PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data" / "scifact"
CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints" / "verifier"
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from claim_verifier.pipeline import ScientificClaimVerifier


st.set_page_config(
    page_title="Scientific Claim Verifier",
    page_icon="🔬",
    layout="wide",
)

st.markdown(
    """
    <style>
    .block-container { max-width: 980px; padding-top: 2.5rem; }
    .eyebrow { color: #59716d; font-size: .78rem; font-weight: 700;
               letter-spacing: .12em; text-transform: uppercase; }
    .verdict { border-radius: 14px; padding: 1.1rem 1.35rem; margin: .5rem 0 1rem; }
    .verdict-label { font-size: 1.45rem; font-weight: 750; margin: 0; }
    .support { background: #e9f6ef; color: #14623b; }
    .contradict { background: #fff0ed; color: #983c2d; }
    .nei { background: #f0f2f5; color: #4d5968; }
    .evidence-card { border: 1px solid #e7e9ec; border-radius: 12px;
                     padding: 1rem 1.1rem; margin: .65rem 0; }
    .evidence-text { font-size: 1.02rem; line-height: 1.55; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown('<div class="eyebrow">SciFact · evidence-backed NLI</div>', unsafe_allow_html=True)
st.title("Scientific Claim Verifier")
st.write(
    "Enter a scientific claim to retrieve relevant papers, surface supporting "
    "or contradicting abstract sentences, and get a model verdict."
)


@st.cache_resource(show_spinner=False)
def load_pipeline() -> ScientificClaimVerifier:
    """Load corpus and models once per Streamlit server process."""
    corpus_path = DATA_DIR / "corpus"
    if not corpus_path.exists():
        raise FileNotFoundError("SciFact is missing. Run `python scripts/download_data.py`.")
    if not CHECKPOINT_DIR.exists():
        raise FileNotFoundError(
            "The fine-tuned verifier is missing. Run `python scripts/train_verifier.py`."
        )
    corpus = load_from_disk(str(corpus_path))["train"]
    return ScientificClaimVerifier(corpus, verifier_model=CHECKPOINT_DIR)


with st.form("claim_form"):
    claim = st.text_input(
        "Claim",
        placeholder="Vitamin D reduces respiratory infections",
        max_chars=1000,
    )
    submitted = st.form_submit_button("Verify claim", type="primary", use_container_width=True)

if submitted:
    st.session_state.pop("verification_result", None)
    if not claim.strip():
        st.warning("Enter a claim to verify.")
    else:
        try:
            with st.spinner(
                "Loading models and indexing SciFact on first run, then verifying the claim…"
            ):
                verifier = load_pipeline()
            with st.spinner("Retrieving papers and checking evidence sentences…"):
                st.session_state["verification_result"] = verifier.verify(claim.strip())
        except FileNotFoundError as error:
            st.error(str(error))
        except Exception as error:
            st.error(f"Verification failed: {error}")


result: dict[str, Any] | None = st.session_state.get("verification_result")
if result:
    st.divider()
    st.markdown(f"**Claim:** {result['claim']}")

    verdict = result["verdict"]
    style = {
        "SUPPORT": "support",
        "CONTRADICT": "contradict",
        "NEI": "nei",
    }.get(verdict, "nei")
    verdict_col, confidence_col = st.columns([2, 1])
    with verdict_col:
        st.markdown(
            f'<div class="verdict {style}"><div class="eyebrow">Verdict</div>'
            f'<p class="verdict-label">{verdict}</p></div>',
            unsafe_allow_html=True,
        )
    with confidence_col:
        st.metric("Confidence", f"{result['confidence']:.2f}")

    st.subheader("Evidence")
    st.caption(
        "Cross-encoder and RRF values are ranking scores; verifier confidence is "
        "the model's softmax probability."
    )
    evidence_items = result["evidence"]
    document_by_id = {str(document["doc_id"]): document for document in result["documents"]}
    if not evidence_items:
        st.info("No evidence sentences were found in the retrieved abstracts.")
    else:
        for index, evidence in enumerate(evidence_items, start=1):
            doc_id = str(evidence["doc_id"])
            document = document_by_id.get(doc_id, {})
            retrieval_score = document.get("score")
            rrf_text = f" · RRF {retrieval_score:.4f}" if retrieval_score is not None else ""
            st.markdown(
                f'<div class="evidence-card"><div class="evidence-text">'
                f'{index}. “{escape(evidence["sentence"])}”</div></div>',
                unsafe_allow_html=True,
            )
            st.markdown(f"**Paper:** {evidence['title']} · doc `{doc_id}`")
            st.caption(
                f"Cross-encoder score {evidence['score']:.4f} · "
                f"retrieval rank {evidence['document_rank']}{rrf_text}"
            )

    with st.expander(f"Retrieved papers ({len(result['documents'])})"):
        for rank, document in enumerate(result["documents"], start=1):
            st.markdown(
                f"**{rank}. {document['title']}**  \n"
                f"doc `{document['doc_id']}` · RRF {document['score']:.4f} · "
                f"BM25 rank {document['bm25_rank']} · dense rank {document['dense_rank']}"
            )
