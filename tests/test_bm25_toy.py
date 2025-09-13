from willrec.io import load_toy_docs, load_toy_queries, load_toy_qrels
from willrec.sparse.bm25 import BM25Searcher
from willrec.eval.metrics import evaluate_run
from pathlib import Path

def test_bm25_toy_pipeline():
    data = Path("data/raw")
    docs = load_toy_docs(data / "toy.jsonl")
    queries = load_toy_queries(data / "toy-queries.jsonl")
    qrels = load_toy_qrels(data / "toy-qrels.jsonl")
    searcher = BM25Searcher(docs)
    run = searcher.search(queries, k=3)
    metrics = evaluate_run(run, qrels, k=3)
    assert 0.0 <= metrics["nDCG@10"] <= 1.0
    assert 0.0 <= metrics["MRR@10"] <= 1.0
