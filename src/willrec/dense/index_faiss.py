from __future__ import annotations
from typing import Tuple
import numpy as np

try:
    import faiss  # type: ignore
except Exception as e:
    raise RuntimeError("faiss-cpu is required. Install with: pip install -e .[ml]") from e


def build_hnsw_index(vecs: np.ndarray, m: int = 32, ef_search: int = 64) -> "faiss.Index":
    """
    Build an HNSW index for inner-product search. Assumes vecs are L2-normalized so cosine == dot.
    """
    d = vecs.shape[1]
    index = faiss.IndexHNSWFlat(d, m, faiss.METRIC_INNER_PRODUCT)  # cosine via IP on normalized
    index.hnsw.efSearch = ef_search
    index.add(vecs.astype(np.float32))
    return index

def build_flat_index(vecs: np.ndarray) -> "faiss.Index":
    d = vecs.shape[1]
    index = faiss.IndexFlatIP(d)
    index.add(vecs.astype(np.float32))
    return index

def search(index: "faiss.Index", query_vecs: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns (scores, idx) with shapes [Q, k].
    """
    D, I = index.search(query_vecs.astype(np.float32), k)
    return D, I
