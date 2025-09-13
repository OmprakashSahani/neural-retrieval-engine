from __future__ import annotations
from pathlib import Path
import csv, json
import numpy as np

from willrec.io import read_jsonl
from willrec.dense.encoder import TextEncoder
from willrec.dense.index_faiss import search as faiss_search
from willrec.eval.metrics import evaluate_run
from willrec.utils import timer, pct

INDEX_DIR = Path("data/index")
PROC_DIR = Path("data/proc/fiqa-mini")
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

def main(model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
         index_name: str = "fiqa-mini.hnsw.faiss",
         k: int = 10, batch: int = 64):
    RESULTS.mkdir(parents=True, exist_ok=True)

    # Load docs ids and index
    doc_ids = np.load(INDEX_DIR / "fiqa-mini.doc_ids.npy", allow_pickle=True).tolist()

    import faiss  # type: ignore
    index = faiss.read_index(str(INDEX_DIR / index_name))

    # Load queries and qrels
    queries = [{"id": r["id"], "text": r["text"]} for r in read_jsonl(PROC_DIR / "queries.jsonl")]
    qrels = load_qrels(PROC_DIR / "qrels.tsv")

    enc = TextEncoder(model_name)

    timings_total = []
    timings_encode = []
    timings_search = []

    all_ids = []
    # Batch encode queries and search
    for i in range(0, len(queries), batch):
        chunk = queries[i:i+batch]
        t = {}

        with timer(t, "total"):
            with timer(t, "encode"):
                q_vecs = enc.encode([q["text"] for q in chunk], batch_size=batch, normalize=True)
            with timer(t, "search"):
                D, I = faiss_search(index, q_vecs, k)
            # Map to ids
            for row in I:
                all_ids.append([doc_ids[j] for j in row.tolist()])

        timings_total.append(t["total"])
        timings_encode.append(t["encode"])
        timings_search.append(t["search"])

    # Build run dict
    run = {q["id"]: ids for q, ids in zip(queries, all_ids)}
    metrics = evaluate_run(run, qrels, k=min(k, 10))

    # Summaries
    report = {
        "k": k,
        "queries": len(queries),
        "metrics": metrics,
        "latency_ms": {
            "encode": {"p50": pct(timings_encode, 50)*1000, "p95": pct(timings_encode, 95)*1000},
            "search": {"p50": pct(timings_search, 50)*1000, "p95": pct(timings_search, 95)*1000},
            "total":  {"p50": pct(timings_total, 50)*1000,  "p95": pct(timings_total, 95)*1000},
        },
    }

    RESULTS.joinpath("fiqa_faiss_run.json").write_text(json.dumps(run, indent=2), encoding="utf-8")
    RESULTS.joinpath("fiqa_faiss_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"[faiss] metrics={metrics}")
    print(f"[faiss] latency ms -> encode: P50={report['latency_ms']['encode']['p50']:.1f}, "
          f"P95={report['latency_ms']['encode']['p95']:.1f}; "
          f"search: P50={report['latency_ms']['search']['p50']:.1f}, "
          f"P95={report['latency_ms']['search']['p95']:.1f}; "
          f"total: P50={report['latency_ms']['total']['p50']:.1f}, "
          f"P95={report['latency_ms']['total']['p95']:.1f}")
    
if __name__ == "__main__":
    main()
