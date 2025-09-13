from __future__ import annotations
from pathlib import Path
import json
import typer
from willrec.io import read_jsonl
from willrec.sparse.bm25 import BM25Searcher
from willrec.eval.metrics import evaluate_run

app = typer.Typer(add_completion=False)

@app.command()
def run(proc_dir: str = "data/proc/fiqa-mini", k: int = 10, out_dir: str = "results"):
    proc = Path(proc_dir)
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)

    docs = [{"id": r["id"], "text": r["text"]} for r in read_jsonl(proc / "corpus.jsonl")]
    queries = [{"id": r["id"], "text": r["text"]} for r in read_jsonl(proc / "queries.jsonl")]

    # read qrels.tsv (qid, Q0, did, rel)
    import csv
    qrels = {}
    with (proc / "qrels.tsv").open("r", encoding="utf-8") as f:
        rdr = csv.reader(f, delimiter="\t")
        for row in rdr:
            if len(row) < 4: 
                continue
            qid, _, did, rel = row[0], row[1], row[2], int(row[3])
            qrels.setdefault(qid, {})[did] = rel

    searcher = BM25Searcher(docs)
    run_dict = searcher.search(queries, k=k)
    metrics = evaluate_run(run_dict, qrels, k=min(k, 10))

    (out / "fiqa_mini_bm25_run.json").write_text(json.dumps(run_dict, indent=2), encoding="utf-8")
    (out / "fiqa_mini_bm25_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    typer.echo(f"[fiqa-mini bm25] {metrics}")

if __name__ == "__main__":
    app()
