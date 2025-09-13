from __future__ import annotations
from pathlib import Path
import json
import numpy as np
from willrec.io import read_jsonl
from willrec.dense.encoder import TextEncoder
from willrec.dense.index_faiss import build_hnsw_index, build_flat_index

OUT_DIR = Path("data/index")
PROC_DIR = Path("data/proc/fiqa-mini")

def main(model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
         use_hnsw: bool = True, m: int = 32, ef_search: int = 64):
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    docs = [{"id": r["id"], "text": r["text"]} for r in read_jsonl(PROC_DIR / "corpus.jsonl")]
    doc_ids = [d["id"] for d in docs]
    texts = [d["text"] for d in docs]

    enc = TextEncoder(model_name)
    vecs = enc.encode(texts, batch_size=64, normalize=True)  # [N, D]

    # Build index
    if use_hnsw:
        index = build_hnsw_index(vecs, m=m, ef_search=ef_search)
        index_path = OUT_DIR / "fiqa-mini.hnsw.faiss"
    else:
        index = build_flat_index(vecs)
        index_path = OUT_DIR / "fiqa-mini.flat.faiss"

    # Save index + ids
    import faiss  # type: ignore
    faiss.write_index(index, str(index_path))
    np.save(OUT_DIR / "fiqa-mini.doc_ids.npy", np.array(doc_ids, dtype=object))

    # Small manifest
    (OUT_DIR / "fiqa-mini.manifest.json").write_text(
        json.dumps({"model": model_name, "use_hnsw": use_hnsw, "m": m, "ef_search": ef_search,
                    "index_path": str(index_path)}, indent=2),
        encoding="utf-8",
    )
    print(f"[build] Saved index to {index_path} with {len(doc_ids)} docs.")

if __name__ == "__main__":
    main()
