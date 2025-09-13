from __future__ import annotations
from pathlib import Path
import json
import typer
import csv

from willrec.io import read_jsonl
from willrec.eval.metrics import evaluate_run
from willrec.dense.encoder import TextEncoder
from willrec.dense.simple_dense import DenseSearcher

app = typer.Typer(add_completion=False)

@app.command()
def run(proc_dir: str = "data/proc/fiqa-mini", k: int = 10, out_dir: str = "results", model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
    proc = Path(proc_dir)
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)

    docs = [{"id": r["id"], "text": r["text"]} for r in read_jsonl(proc / "corpus.jsonl")]
    queries = [{"id": r["id"], "text": r["text"]} for r in read_jsonl(proc / "queries.jsonl")]

    qrels = {}
    with (proc / "qrels.tsv").open("r", encoding="utf-8") as f:
        rdr = csv.reader(f, delimiter="\t")
        for row in rdr:
            if len(row) < 4: 
                continue
            qid, _, did, rel = row[0], row[1], row[2], int(row[3])
            qrels.setdefault(qid, {})[did] = rel

    enc = TextEncoder(model_name)
    doc_ids = [d["id"] for d in docs]
    doc_vecs = enc.encode([d["text"] for d in docs])
    q_vecs = enc.encode([q["text"] for q in queries])

    run_lists = DenseSearcher(doc_ids, doc_vecs).search(q_vecs, k=k)
    run_dict = {q["id"]: ids for q, ids in zip(queries, run_lists)}
    metrics = evaluate_run(run_dict, qrels, k=min(k, 10))

    (out / "fiqa_mini_dense_run.json").write_text(json.dumps(run_dict, indent=2), encoding="utf-8")
    (out / "fiqa_mini_dense_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    typer.echo(f"[fiqa-mini dense] {metrics}")

if __name__ == "__main__":
    app()
