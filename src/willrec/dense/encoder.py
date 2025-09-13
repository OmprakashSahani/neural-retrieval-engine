from __future__ import annotations
from typing import List
import numpy as np

try:
    from sentence_transformers import SentenceTransformer
except Exception as e:
    raise RuntimeError(
        "sentence-transformers is required. Install with: pip install -e .[ml]"
    ) from e


class TextEncoder:
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        # Small, fast model; good for CPU and toy/small subsets
        self.model = SentenceTransformer(model_name)

    def encode(self, texts: List[str], batch_size: int = 32, normalize: bool = True) -> np.ndarray:
        emb = self.model.encode(
            texts,
            batch_size=batch_size,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        if normalize:
            # L2-normalize for cosine similarity via dot product
            norms = np.linalg.norm(emb, axis=1, keepdims=True) + 1e-12
            emb = emb / norms
        return emb
