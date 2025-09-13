from __future__ import annotations
from typing import List, Dict
import numpy as np


def cosine_topk(
    query_vecs: np.ndarray,
    doc_vecs: np.ndarray,
    doc_ids: List[str],
    k: int = 10,
) -> List[List[str]]:
    """
    Returns top-k doc_ids per query by cosine similarity.
    Assumes vectors are already L2-normalized; cosine == dot product.
    """
    # (Q x D) score matrix via dot product
    scores = query_vecs @ doc_vecs.T  # shape: [num_queries, num_docs]
    k = min(k, doc_vecs.shape[0])
    # argsort descending per row
    idxs = np.argpartition(-scores, kth=k - 1, axis=1)[:, :k]
    # refine exact order for those k elements
    ordered = np.take_along_axis(
        idxs,
        np.argsort(-np.take_along_axis(scores, idxs, axis=1), axis=1),
        axis=1,
    )
    # map to IDs
    results: List[List[str]] = []
    for row in ordered:
        results.append([doc_ids[i] for i in row.tolist()])
    return results


class DenseSearcher:
    def __init__(self, doc_ids: List[str], doc_vecs: np.ndarray):
        self.doc_ids = doc_ids
        self.doc_vecs = doc_vecs

    def search(self, query_vecs: np.ndarray, k: int = 10) -> List[List[str]]:
        return cosine_topk(query_vecs, self.doc_vecs, self.doc_ids, k=k)
