from __future__ import annotations
import time
from contextlib import contextmanager
from typing import Dict, List
import numpy as np

@contextmanager
def timer(record: Dict[str, float], key: str):
    t0 = time.perf_counter()
    try:
        yield
    finally:
        record[key] = record.get(key, 0.0) + (time.perf_counter() - t0)

def pct(values: List[float], p: float) -> float:
    if not values:
        return 0.0
    return float(np.percentile(np.asarray(values, dtype=float), p))
