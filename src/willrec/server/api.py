from __future__ import annotations
from fastapi import FastAPI, Query
from pydantic import BaseModel
from typing import List, Literal
from pathlib import Path
import csv, json, numpy as np

from willrec.io import read_jsonl
from willrec.sparse.bm25 import BM25Searcher
from willrec.dense.encoder import TextEncoder
from willrec.dense.index_faiss import search as faiss_search
from willrec.hybrid.fuse import fuse_rankings

APP = FastAPI(title="willrec API", version="0.1")

# ---- Load FiQA-mini assets once on startup ----
PROC_DIR = Path("data/proc/fiqa-mini")
INDEX_DIR = Path("data/index")
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
K_CANDIDATES_DEFAULT = 50

class SearchHit(BaseModel):
    doc_id: str
    score: float | None = None
    snippet: str

class SearchResponse(BaseModel):
    mode: Literal["bm25","dense","hybrid"]
    k: int
    alpha: float | None = None
    results: List[SearchHit]

def _load_docs():
    docs = [{"id": r["id"], "text": r["text"]} for r in read_jsonl(PROC_DIR / "corpus.jsonl")]
    return docs, [d["id"] for d in docs], [d["text"] for d in docs]

def _snippet(text: str, n: int = 180) -> str:
    t = " ".join(text.split())
    return t[:n] + ("…" if len(t) > n else "")

@APP.on_event("startup")
def _startup():
    global DOCS, DOC_IDS, DOC_TEXTS, BM25, ENC, INDEX, DOC_IDS_SAVED
    DOCS, DOC_IDS, DOC_TEXTS = _load_docs()
    BM25 = BM25Searcher(DOCS)
    ENC = TextEncoder(MODEL_NAME)

    import faiss  # type: ignore
    # prefer HNSW if available; else flat
    idx_path = INDEX_DIR / "fiqa-mini.hnsw.faiss"
    if not idx_path.exists():
        idx_path = INDEX_DIR / "fiqa-mini.flat.faiss"
    INDEX = faiss.read_index(str(idx_path))
    DOC_IDS_SAVED = np.load(INDEX_DIR / "fiqa-mini.doc_ids.npy", allow_pickle=True).tolist()
    assert DOC_IDS_SAVED == DOC_IDS, "doc_ids mismatch; rebuild index."

@APP.get("/health")
def health():
    return {"ok": True, "docs": len(DOC_IDS)}

@APP.get("/search", response_model=SearchResponse)
def search(
    q: str = Query(..., min_length=1),
    k: int = Query(10, ge=1, le=100),
    mode: Literal["bm25","dense","hybrid"] = "hybrid",
    alpha: float = 0.5,
    k_candidates: int = K_CANDIDATES_DEFAULT,
    norm: Literal["z","minmax","none"] = "z",
):
    qtext = q.strip()

    if mode == "bm25":
        toks = qtext.lower().split()
        scores = BM25.bm25.get_scores(toks)
        order = np.argsort(-scores)[:k]
        hits = [SearchHit(doc_id=DOC_IDS[i], score=float(scores[i]), snippet=_snippet(DOC_TEXTS[i])) for i in order]
        return SearchResponse(mode=mode, k=k, results=hits)

    # dense first (candidate generation via FAISS)
    q_vec = ENC.encode([qtext])[0:1]  # normalized
    D, I = faiss_search(INDEX, q_vec, max(k, k_candidates))
    cand_idx = I[0].tolist()
    dense_scores = D[0].tolist()

    if mode == "dense":
        hits = [SearchHit(doc_id=DOC_IDS[i], score=float(dense_scores[j]), snippet=_snippet(DOC_TEXTS[i]))
                for j, i in enumerate(cand_idx[:k])]
        return SearchResponse(mode=mode, k=k, results=hits)

    # hybrid: fuse BM25 (over candidates) + dense
    bm25_scores_all = BM25.bm25.get_scores(qtext.lower().split()).tolist()
    bm25_scores = [bm25_scores_all[i] for i in cand_idx]

    fused_ids = fuse_rankings(
        doc_ids=[DOC_IDS[i] for i in cand_idx],
        bm25_scores=bm25_scores,
        dense_scores=dense_scores,
        alpha=alpha,
        norm=norm,
        k=k,
    )
    # create a quick score map (not strictly comparable after fusion)
    # we’ll report dense score for visibility
    dense_map = {DOC_IDS[i]: float(dense_scores[j]) for j, i in enumerate(cand_idx)}
    hits = [SearchHit(doc_id=did, score=dense_map.get(did, None),
                      snippet=_snippet(DOC_TEXTS[DOC_IDS.index(did)])) for did in fused_ids]
    return SearchResponse(mode=mode, k=k, alpha=alpha, results=hits)
