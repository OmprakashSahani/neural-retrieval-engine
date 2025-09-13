from __future__ import annotations
import typer
from pathlib import Path
import json

from willrec.io import load_toy_docs, load_toy_queries, load_toy_qrels, read_jsonl
from willrec.sparse.bm25 import BM25Searcher
from willrec.dense.encoder import TextEncoder
from willrec.dense.simple_dense import DenseSearcher
from willrec.eval.metrics import evaluate_run
from willrec.hybrid.fuse import fuse_rankings

app = typer.Typer(add_completion=False, help="willrec: hybrid retrieval utilities")

# -----------------------
# Toy dataset commands
# -----------------------
toy = typer.Typer(help="Toy dataset runners")
app.add_typer(toy, name="toy")

@toy.command("bm25")
def toy_bm25(k: int = 10, data_dir: str = "data/raw", out_dir: str = "results"):
    data = Path(data_dir)
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    docs = load_toy_docs(data / "toy.jsonl")
    queries = load_toy_queries(data / "toy-queries.jsonl")
    qrels = load_toy_qrels(data / "toy-qrels.jsonl")
    run = BM25Searcher(docs).search(queries, k=k)
    metrics = evaluate_run(run, qrels, k=min(k, 10))
    (out / "toy_bm25_run.json").write_text(json.dumps(run, indent=2), encoding="utf-8")
    (out / "toy_bm25_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    typer.echo(f"[toy bm25] {metrics}")

@toy.command("dense")
def toy_dense(
    k: int = 10,
    data_dir: str = "data/raw",
    out_dir: str = "results",
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
):
    data = Path(data_dir)
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    docs = load_toy_docs(data / "toy.jsonl")
    queries = load_toy_queries(data / "toy-queries.jsonl")
    qrels = load_toy_qrels(data / "toy-qrels.jsonl")
    enc = TextEncoder(model_name)
    doc_ids = [d["id"] for d in docs]
    doc_vecs = enc.encode([d["text"] for d in docs])
    q_vecs = enc.encode([q["text"] for q in queries])
    run_lists = DenseSearcher(doc_ids, doc_vecs).search(q_vecs, k=k)
    run = {q["id"]: ids for q, ids in zip(queries, run_lists)}
    metrics = evaluate_run(run, qrels, k=min(k, 10))
    (out / "toy_dense_run.json").write_text(json.dumps(run, indent=2), encoding="utf-8")
    (out / "toy_dense_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    typer.echo(f"[toy dense] {metrics}")

@toy.command("fuse")
def toy_fuse(
    k: int = 10,
    norm: str = "z",
    alpha: float = 0.5,
    data_dir: str = "data/raw",
    out_dir: str = "results",
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
):
    data = Path(data_dir)
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    docs = load_toy_docs(data / "toy.jsonl")
    queries = load_toy_queries(data / "toy-queries.jsonl")
    qrels = load_toy_qrels(data / "toy-qrels.jsonl")
    # sparse
    bm25 = BM25Searcher(docs)
    # dense
    enc = TextEncoder(model_name)
    doc_ids = [d["id"] for d in docs]
    doc_texts = [d["text"] for d in docs]
    doc_vecs = enc.encode(doc_texts)
    q_vecs = enc.encode([q["text"] for q in queries])

    def bm25_scores(qtext: str):
        return bm25.bm25.get_scores(qtext.lower().split()).tolist()

    run = {}
    for qi, q in enumerate(queries):
        b = bm25_scores(q["text"])
        d = (q_vecs[qi] @ doc_vecs.T).tolist()
        fused = fuse_rankings(doc_ids, b, d, alpha=alpha, norm=norm, k=k)
        run[q["id"]] = fused

    metrics = evaluate_run(run, qrels, k=min(k, 10))
    (out / "toy_fuse_run.json").write_text(json.dumps(run, indent=2), encoding="utf-8")
    (out / "toy_fuse_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    typer.echo(f"[toy fuse a={alpha} norm={norm}] {metrics}")

# -----------------------
# BEIR (local) ingestion
# -----------------------
ingest = typer.Typer(help="Ingest BEIR-style datasets from a local folder")
app.add_typer(ingest, name="ingest")

@ingest.command("beir-local")
def beir_local(
    src_dir: str = typer.Argument(..., help="BEIR dataset root containing corpus.jsonl, queries.jsonl, qrels folder"),
    split: str = typer.Option("test", help="BEIR qrels split: train|dev|test"),
    limit_docs: int = typer.Option(10000, help="Cap number of documents"),
    out_name: str = typer.Option("fiqa-mini", help="Name of processed subset under data/proc/"),
):
    """
    Expect layout like:
      <src_dir>/corpus.jsonl
      <src_dir>/queries.jsonl
      <src_dir>/qrels/<split>.tsv
    Writes data/proc/<out_name>/{corpus.jsonl, queries.jsonl, qrels.tsv} limited to the first N docs,
    and filters qrels/queries to those IDs.
    """
    import csv

    src = Path(src_dir)
    corpus_path = src / "corpus.jsonl"
    queries_path = src / "queries.jsonl"
    qrels_path = src / "qrels" / f"{split}.tsv"

    assert corpus_path.exists(), f"Missing {corpus_path}"
    assert queries_path.exists(), f"Missing {queries_path}"
    assert qrels_path.exists(), f"Missing {qrels_path}"

    out_dir = Path("data/proc") / out_name
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1) take first N docs
    kept_docs = []
    kept_doc_ids = set()
    for i, row in enumerate(read_jsonl(corpus_path)):
        if i >= limit_docs:
            break
        # BEIR corpus rows: {"_id": "docid", "title": "...", "text": "..."}
        did = row.get("_id") or row.get("id")
        text = " ".join([row.get("title", ""), row.get("text", "")]).strip()
        kept_docs.append({"id": did, "text": text})
        kept_doc_ids.add(did)

    # 2) load queries
    kept_queries = []
    query_texts = {}
    for row in read_jsonl(queries_path):
        qid = row.get("_id") or row.get("id")
        qtext = row.get("text") or row.get("query") or ""
        kept_queries.append({"id": qid, "text": qtext})
        query_texts[qid] = qtext

    # 3) filter qrels to (qid, did) where did in kept_doc_ids and qtext non-empty
    kept_qrels = []
    with qrels_path.open("r", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t")
        # header may or may not be present; tolerate both
        for row in reader:
            if len(row) < 3:
                continue
            qid, _, did, rel = row[0], row[1], row[2], row[3] if len(row) > 3 else "1"
            if did in kept_doc_ids and query_texts.get(qid, ""):
                kept_qrels.append({"qid": qid, "did": did, "rel": int(rel)})

    # 4) Persist
    (out_dir / "corpus.jsonl").write_text(
        "\n".join(json.dumps(d, ensure_ascii=False) for d in kept_docs),
        encoding="utf-8",
    )
    (out_dir / "queries.jsonl").write_text(
        "\n".join(json.dumps(q, ensure_ascii=False) for q in kept_queries),
        encoding="utf-8",
    )
    # tsv
    with (out_dir / "qrels.tsv").open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, delimiter="\t")
        # qid, _, did, rel — keep BEIR style
        for r in kept_qrels:
            w.writerow([r["qid"], "Q0", r["did"], r["rel"]])

    typer.echo(f"[ingest] Wrote subset to {out_dir} with {len(kept_docs)} docs and {len(kept_qrels)} qrels.")

if __name__ == "__main__":
    app()
