from pathlib import Path
from willrec.io import load_toy_docs, load_toy_queries, load_toy_qrels
from willrec.dense.encoder import TextEncoder
from willrec.dense.simple_dense import DenseSearcher
from willrec.eval.metrics import evaluate_run

def test_dense_toy_pipeline():
    data = Path("data/raw")
    docs = load_toy_docs(data / "toy.jsonl")
    queries = load_toy_queries(data / "toy-queries.jsonl")
    qrels = load_toy_qrels(data / "toy-qrels.jsonl")

    enc = TextEncoder()
    doc_ids = [d["id"] for d in docs]
    doc_vecs = enc.encode([d["text"] for d in docs])
    q_vecs = enc.encode([q["text"] for q in queries])

    searcher = DenseSearcher(doc_ids, doc_vecs)
    topk = searcher.search(q_vecs, k=3)
    run_dict = {q["id"]: ids for q, ids in zip(queries, topk)}
    metrics = evaluate_run(run_dict, qrels, k=3)
    assert 0.0 <= metrics["nDCG@10"] <= 1.0
    assert 0.0 <= metrics["MRR@10"] <= 1.0
