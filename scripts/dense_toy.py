from __future__ import annotations
from pathlib import Path
import json
import typer

from willrec.io import load_toy_docs, load_toy_queries, load_toy_qrels
from willrec.eval.metrics import evaluate_run
from willrec.dense.encoder import TextEncoder
from willrec.dense.simple_dense import DenseSearcher

app = typer.Typer(add_completion=False)

@app.command()
def run(
    data_dir: str = typer.Option("data/raw", help="Directory with toy files"),
    k: int = typer.Option(10, help="Top-k"),
    out_dir: str = typer.Option("results", help="Where to write outputs"),
    model_name: str = typer.Option("sentence-transformers/all-MiniLM-L6-v2", help="SBERT model"),
):
    data_dir = Path(data_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    docs = load_toy_docs(data_dir / "toy.jsonl")
    queries = load_toy_queries(data_dir / "toy-queries.jsonl")
    qrels = load_toy_qrels(data_dir / "toy-qrels.jsonl")

    # Encode
    encoder = TextEncoder(model_name=model_name)
    doc_ids = [d["id"] for d in docs]
    doc_texts = [d["text"] for d in docs]
    q_texts = [q["text"] for q in queries]

    doc_vecs = encoder.encode(doc_texts)
    q_vecs = encoder.encode(q_texts)

    # Search
    searcher = DenseSearcher(doc_ids, doc_vecs)
    topk_lists = searcher.search(q_vecs, k=k)

    # Convert to run dict
    run_dict = {q["id"]: ids for q, ids in zip(queries, topk_lists)}
    metrics = evaluate_run(run_dict, qrels, k=min(k, 10))

    # Save artifacts
    with (out_dir / "dense_toy_run.json").open("w", encoding="utf-8") as f:
        json.dump(run_dict, f, ensure_ascii=False, indent=2)
    with (out_dir / "dense_toy_metrics.json").open("w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    typer.echo(f"Metrics: {metrics}")

if __name__ == "__main__":
    app()
