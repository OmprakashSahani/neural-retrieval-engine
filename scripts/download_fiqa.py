from __future__ import annotations
from pathlib import Path
import json, csv
import ir_datasets

OUT = Path("data/raw/fiqa")

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def write_jsonl(path: Path, rows):
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

def write_qrels_tsv(path: Path, rows):
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, delimiter="\t")
        # BEIR-style: qid  Q0  did  rel  (no header)
        for qid, did, rel in rows:
            w.writerow([qid, "Q0", did, rel])

def main():
    # Docs live at "beir/fiqa" (corpus)
    docs_ds = ir_datasets.load("beir/fiqa")
    # Queries + qrels live under a split, e.g. "beir/fiqa/test"
    split_ds = ir_datasets.load("beir/fiqa/test")

    # 1) Write corpus.jsonl with {_id,title,text}
    docs_rows = []
    for d in docs_ds.docs_iter():
        # ir_datasets docs generally have: d.doc_id, d.text, maybe d.title
        title = getattr(d, "title", "")
        text = getattr(d, "text", "") or ""
        docs_rows.append({"_id": d.doc_id, "title": title or "", "text": text})
    write_jsonl(OUT / "corpus.jsonl", docs_rows)

    # 2) Write queries.jsonl with {_id,text}
    queries_rows = []
    for q in split_ds.queries_iter():
        # usually q.query_id / q.text
        qid = getattr(q, "query_id", None) or getattr(q, "id", None)
        qtext = getattr(q, "text", "")
        queries_rows.append({"_id": qid, "text": qtext})
    write_jsonl(OUT / "queries.jsonl", queries_rows)

    # 3) Write qrels/test.tsv with qid\tQ0\tdid\trel
    qrels_rows = []
    for r in split_ds.qrels_iter():
        # r.query_id, r.doc_id, r.relevance
        qrels_rows.append((r.query_id, r.doc_id, int(getattr(r, "relevance", 1))))
    write_qrels_tsv(OUT / "qrels" / "test.tsv", qrels_rows)

    # Sanity
    req = [OUT / "corpus.jsonl", OUT / "queries.jsonl", OUT / "qrels" / "test.tsv"]
    missing = [str(p) for p in req if not p.exists()]
    if missing:
        raise SystemExit(f"[error] Missing: {missing}")
    print(f"[ok] FiQA ready at {OUT}")

if __name__ == "__main__":
    main()
