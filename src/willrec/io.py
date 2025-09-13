from __future__ import annotations
from typing import List, Dict, Tuple
import json
from pathlib import Path

def read_jsonl(path: str | Path):
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)

def load_toy_docs(path: str | Path) -> List[Dict[str, str]]:
    return list(read_jsonl(path))

def load_toy_queries(path: str | Path) -> List[Dict[str, str]]:
    return list(read_jsonl(path))

def load_toy_qrels(path: str | Path) -> Dict[str, Dict[str, int]]:
    """Return {qid: {did: rel}}."""
    qrels: Dict[str, Dict[str, int]] = {}
    for row in read_jsonl(path):
        qid = row["qid"]
        did = row["did"]
        rel = int(row.get("rel", 0))
        qrels.setdefault(qid, {})[did] = rel
    return qrels
