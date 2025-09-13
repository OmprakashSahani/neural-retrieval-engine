from typing import List, Dict
import numpy as np

def dcg_at_k(relevances: List[int], k: int) -> float:
    """Compute Discounted Cumulative Gain (DCG) at rank k."""
    relevances = np.array(relevances)[:k]
    if relevances.size == 0:
        return 0.0
    discounts = np.log2(np.arange(2, relevances.size + 2))
    return float(np.sum((2**relevances - 1) / discounts))

def ndcg_at_k(relevances: List[int], k: int) -> float:
    """Normalized DCG at rank k."""
    dcg = dcg_at_k(relevances, k)
    ideal = dcg_at_k(sorted(relevances, reverse=True), k)
    return dcg / ideal if ideal > 0 else 0.0

def mrr_at_k(relevances: List[int], k: int) -> float:
    """Mean Reciprocal Rank at k."""
    for i, rel in enumerate(relevances[:k], start=1):
        if rel > 0:
            return 1.0 / i
    return 0.0

def evaluate_run(run: Dict[str, List[str]], qrels: Dict[str, Dict[str, int]], k: int = 10) -> Dict[str, float]:
    """
    run: {query_id: [doc_id1, doc_id2, ...]}
    qrels: {query_id: {doc_id: relevance}}
    """
    ndcgs, mrrs = [], []
    for qid, docs in run.items():
        rels = [qrels.get(qid, {}).get(doc, 0) for doc in docs]
        ndcgs.append(ndcg_at_k(rels, k))
        mrrs.append(mrr_at_k(rels, k))
    return {
        "nDCG@10": float(np.mean(ndcgs)),
        "MRR@10": float(np.mean(mrrs)),
    }
