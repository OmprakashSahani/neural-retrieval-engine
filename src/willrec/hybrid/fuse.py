from __future__ import annotations
from typing import Dict, List, Tuple
import numpy as np

def _zscore(x: np.ndarray) -> np.ndarray:
    m = x.mean()
    s = x.std()
    return (x - m) / (s + 1e-12)

def normalize_per_query(scores: List[float], method: str = "z") -> np.ndarray:
    arr = np.asarray(scores, dtype=float)
    if method == "z":
        return _zscore(arr)
    elif method == "minmax":
        lo, hi = arr.min(), arr.max()
        return (arr - lo) / (hi - lo + 1e-12)
    else:
        return arr  # no-op

def fuse_rankings(
    doc_ids: List[str],
    bm25_scores: List[float],
    dense_scores: List[float],
    alpha: float,
    norm: str = "z",
    k: int = 10,
) -> List[str]:
    """
    Fuse per-query scores via: alpha * bm25 + (1 - alpha) * dense.
    """
    b = normalize_per_query(bm25_scores, norm)
    d = normalize_per_query(dense_scores, norm)
    fused = alpha * b + (1.0 - alpha) * d
    order = np.argsort(-fused)[:k]
    return [doc_ids[i] for i in order.tolist()]
