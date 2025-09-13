from __future__ import annotations
from typing import List, Dict
from rank_bm25 import BM25Okapi

def _tokenize(text: str) -> List[str]:
    # ultra-simple whitespace/lowecase tokenizer; good enough for toy and sanity checks
    return text.lower().split()

class BM25Searcher:
    def __init__(self, docs: List[Dict[str, str]]):
        """
        docs: [{"id": "...", "text": "..."}]
        """
        self.doc_ids = [d["id"] for d in docs]
        corpus = [_tokenize(d["text"]) for d in docs]
        self.bm25 = BM25Okapi(corpus)

    def search(self, queries: List[Dict[str, str]], k: int = 10) -> Dict[str, List[str]]:
        """
        queries: [{"id": "...", "text": "..."}]
        returns: {qid: [doc_id1, doc_id2, ...]}
        """
        run: Dict[str, List[str]] = {}
        for q in queries:
            qid = q["id"]
            toks = _tokenize(q["text"])
            scores = self.bm25.get_scores(toks)
            # argsort descending
            ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
            run[qid] = [self.doc_ids[i] for i in ranked]
        return run
