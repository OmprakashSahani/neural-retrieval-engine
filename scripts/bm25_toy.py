from __future__ import annotations
from pathlib import Path
import json
import typer
from willrec.io import load_toy_docs, load_toy_queries, load_toy_qrels
from willrec.sparse.bm25 import BM25Searcher
from willrec.eval.metrics import evaluate_run

app = typer.Typer(add_completion=False)

@app.command()
def run(
    data_dir: str = typer.Option("data/raw", help="Directory with toy files"),
    k: int = typer.Option(10, help="Top-k"),
    out_dir: str = typer.Option("results", help="Where to write outputs"),
):
    data_dir = Path(data_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    docs = load_toy_docs(data_dir / "toy.jsonl")
    queries = load_toy_queries(data_dir / "toy-queries.jsonl")
    qrels = load_toy_qrels(data_dir / "toy-qrels.jsonl")

    searcher = BM25Searcher(docs)
    run_dict = searcher.search(queries, k=k)

    metrics = evaluate_run(run_dict, qrels, k=min(k, 10))

    # save artifacts
    with (out_dir / "bm25_toy_run.json").open("w", encoding="utf-8") as f:
        json.dump(run_dict, f, ensure_ascii=False, indent=2)
    with (out_dir / "bm25_toy_metrics.json").open("w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    typer.echo(f"Metrics: {metrics}")

if __name__ == "__main__":
    app()
