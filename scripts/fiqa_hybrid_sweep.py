from __future__ import annotations
from pathlib import Path
import csv, json, time
import numpy as np

from willrec.io import read_jsonl
from willrec.sparse.bm25 import BM25Searcher
from willrec.dense.encoder import TextEncoder
from willrec.dense.index_faiss import search as faiss_search
from willrec.eval.metrics import evaluate_run
from willrec.hybrid.fuse import fuse_rankings
from willrec.utils import pct

PROC_DIR = Path("data/proc/fiqa-mini")
INDEX_DIR = Path("data/index")
RESULTS = Path("results")

def load_qrels(tsv_path: Path):
    qrels = {}
    with tsv_path.open("r", encoding="utf-8") as f:
        rdr = csv.reader(f, delimiter="\t")
        for row in rdr:
            if len(row) < 4:
                continue
            qid, _, did, rel = row[0], row[1], row[2], int(row[3])
            qrels.setdefault(qid, {})[did] = rel
    return qrels

def main(
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    index_name: str = "fiqa-mini.hnsw.faiss",
    k_candidates: int = 50,
    k_eval: int = 10,
    alphas_csv: str = "0.0,0.25,0.5,0.75,1.0",
    norm: str = "z",
    batch: int = 64,
):
    RESULTS.mkdir(parents=True, exist_ok=True)

    # Load data
    docs = [{"id": r["id"], "text": r["text"]} for r in read_jsonl(PROC_DIR / "corpus.jsonl")]
    doc_ids = [d["id"] for d in docs]
    queries = [{"id": r["id"], "text": r["text"]} for r in read_jsonl(PROC_DIR / "queries.jsonl")]
    qrels = load_qrels(PROC_DIR / "qrels.tsv")

    # BM25 (for per-query sparse scores)
    bm25 = BM25Searcher(docs)

    # FAISS index + doc ids from previous build
    import faiss  # type: ignore
    index = faiss.read_index(str(INDEX_DIR / index_name))
    doc_ids_saved = np.load(INDEX_DIR / "fiqa-mini.doc_ids.npy", allow_pickle=True).tolist()
    assert doc_ids_saved == doc_ids, "doc_ids mismatch; rebuild index."

    # Encode queries
    enc = TextEncoder(model_name)
    qtexts = [q["text"] for q in queries]

    t0 = time.perf_counter()
    q_vecs = enc.encode(qtexts, batch_size=batch, normalize=True)
    t_encode = (time.perf_counter() - t0) * 1000.0

    # Dense candidate generation via FAISS
    per_q_dense_idx = []
    per_q_dense_scores = []
    t_search_ms = []

    for i in range(0, len(queries), batch):
        chunk_vecs = q_vecs[i:i+batch]
        t1 = time.perf_counter()
        D, I = faiss_search(index, chunk_vecs, k_candidates)
        t_search_ms.append((time.perf_counter() - t1) * 1000.0)
        per_q_dense_idx.extend(I.tolist())
        per_q_dense_scores.extend(D.tolist())

    # For each query, prepare aligned score arrays (same doc order) for fusion
    alpha_vals = [float(x.strip()) for x in alphas_csv.split(",") if x.strip()]
    per_alpha_metrics = {}
    best = {"alpha": None, "nDCG@10": -1.0, "MRR@10": -1.0}

    # Precompute BM25 scores per query across ALL docs once (vector form)
    # WARNING: This is O(N_docs) per query; acceptable for 10k docs.
    bm25_all_scores = []
    for q in queries:
        bm25_all_scores.append(bm25.bm25.get_scores(q["text"].lower().split()).tolist())

    for alpha in alpha_vals:
        run = {}
        # Fuse using BM25 scores (all docs) and dense scores only for candidate subset
        for qi, q in enumerate(queries):
            # candidate doc indices from dense ANN
            cand_idx = per_q_dense_idx[qi]
            dense_scores = per_q_dense_scores[qi]
            # grab BM25 scores for those candidates
            bm25_scores = [bm25_all_scores[qi][di] for di in cand_idx]
            cand_ids = [doc_ids[di] for di in cand_idx]

            fused_ids = fuse_rankings(
                doc_ids=cand_ids,
                bm25_scores=bm25_scores,
                dense_scores=dense_scores,
                alpha=alpha,
                norm=norm,
                k=k_eval,
            )
            run[q["id"]] = fused_ids

        metrics = evaluate_run(run, qrels, k=k_eval)
        per_alpha_metrics[f"{alpha:.2f}"] = metrics

        if (metrics["nDCG@10"] > best["nDCG@10"]) or (
            np.isclose(metrics["nDCG@10"], best["nDCG@10"]) and metrics["MRR@10"] > best["MRR@10"]
        ):
            best = {"alpha": alpha, **metrics}

    report = {
        "k_candidates": k_candidates,
        "k_eval": k_eval,
        "norm": norm,
        "alphas": alpha_vals,
        "best": best,
        "per_alpha": per_alpha_metrics,
        "latency_ms": {
            "encode_total": t_encode,
            "search_p50": float(np.percentile(np.asarray(t_search_ms), 50)) if t_search_ms else 0.0,
            "search_p95": float(np.percentile(np.asarray(t_search_ms), 95)) if t_search_ms else 0.0,
        },
    }

    RESULTS.joinpath("fiqa_hybrid_sweep.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"[hybrid] best alpha={best['alpha']} metrics={ {'nDCG@10': best['nDCG@10'], 'MRR@10': best['MRR@10']} }")
    print(f"[hybrid] latency -> encode_total_ms={report['latency_ms']['encode_total']:.1f}, "
          f"search_p50_ms={report['latency_ms']['search_p50']:.1f}, search_p95_ms={report['latency_ms']['search_p95']:.1f}")

if __name__ == "__main__":
    main()
