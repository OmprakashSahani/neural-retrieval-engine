from __future__ import annotations
from pathlib import Path
import json
import typer
import numpy as np

from willrec.io import load_toy_docs, load_toy_queries, load_toy_qrels
from willrec.eval.metrics import evaluate_run
from willrec.sparse.bm25 import BM25Searcher
from willrec.dense.encoder import TextEncoder
from willrec.dense.simple_dense import DenseSearcher
from willrec.hybrid.fuse import fuse_rankings

app = typer.Typer(add_completion=False)

def bm25_with_scores(searcher: BM25Searcher, query_text: str, doc_texts: list[str]):
    # rank_bm25 exposes get_scores(tokenized_q). We’ll recompute scores directly.
    toks = query_text.lower().split()
    scores = searcher.bm25.get_scores(toks).tolist()
    return scores

def dense_with_scores(searcher: DenseSearcher, q_vec, doc_vecs):
    # cosine == dot product (vectors already L2-normalized)
    return (q_vec @ doc_vecs.T).tolist()

@app.command()
def run(
    data_dir: str = typer.Option("data/raw", help="Directory with toy files"),
    out_dir: str = typer.Option("results", help="Where to write outputs"),
    k: int = typer.Option(10, help="Top-k for evaluation"),
    norm: str = typer.Option("z", help="Score normalization: z|minmax|none"),
    alphas: str = typer.Option("0.0,0.25,0.5,0.75,1.0", help="Comma-separated alpha list"),
    model_name: str = typer.Option("sentence-transformers/all-MiniLM-L6-v2"),
):
    data_dir = Path(data_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    docs = load_toy_docs(data_dir / "toy.jsonl")
    queries = load_toy_queries(data_dir / "toy-queries.jsonl")
    qrels = load_toy_qrels(data_dir / "toy-qrels.jsonl")

    doc_ids = [d["id"] for d in docs]
    doc_texts = [d["text"] for d in docs]
    q_texts = [q["text"] for q in queries]

    # Sparse
    bm25 = BM25Searcher(docs)

    # Dense
    encoder = TextEncoder(model_name=model_name)
    doc_vecs = encoder.encode(doc_texts)
    q_vecs = encoder.encode(q_texts)
    dense = DenseSearcher(doc_ids, doc_vecs)

    # Precompute per-query raw scores for both systems
    per_query_scores = []
    for qi, q in enumerate(queries):
        bm25_scores = bm25_with_scores(bm25, q["text"], doc_texts)
        dense_scores = dense_with_scores(dense, q_vecs[qi], doc_vecs)
        per_query_scores.append((bm25_scores, dense_scores))

    # Grid search over alphas
    alpha_vals = [float(x.strip()) for x in alphas.split(",") if x.strip()]
    results = {}
    best = {"alpha": None, "nDCG@10": -1.0, "MRR@10": -1.0}

    for a in alpha_vals:
        run_dict = {}
        for qi, q in enumerate(queries):
            b_scores, d_scores = per_query_scores[qi]
            fused_ids = fuse_rankings(
                doc_ids=doc_ids,
                bm25_scores=b_scores,
                dense_scores=d_scores,
                alpha=a,
                norm=norm,
                k=k,
            )
            run_dict[q["id"]] = fused_ids

        metrics = evaluate_run(run_dict, qrels, k=min(k, 10))
        results[f"{a:.2f}"] = metrics

        # track best on nDCG@10 then MRR@10 as tiebreak
        if (metrics["nDCG@10"] > best["nDCG@10"]) or (
            np.isclose(metrics["nDCG@10"], best["nDCG@10"]) and metrics["MRR@10"] > best["MRR@10"]
        ):
            best = {"alpha": a, **metrics}

    # Save & print
    with (out_dir / "fusion_grid_results.json").open("w", encoding="utf-8") as f:
        json.dump({"per_alpha": results, "best": best, "norm": norm}, f, indent=2)

    typer.echo(f"Best fusion (norm={norm}): alpha={best['alpha']}, metrics={ {'nDCG@10': best['nDCG@10'], 'MRR@10': best['MRR@10']} }")

if __name__ == "__main__":
    app()
