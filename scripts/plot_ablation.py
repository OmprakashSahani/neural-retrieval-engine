from __future__ import annotations
import json
import matplotlib.pyplot as plt

data = json.load(open("results/fiqa_hybrid_sweep.json", "r", encoding="utf-8"))
alphas = [float(a) for a in data["alphas"]]
ndcg = [data["per_alpha"][f"{a:.2f}"]["nDCG@10"] for a in alphas]
mrr = [data["per_alpha"][f"{a:.2f}"]["MRR@10"] for a in alphas]

plt.figure(figsize=(6,4))
plt.plot(alphas, ndcg, marker="o", label="nDCG@10")
plt.plot(alphas, mrr, marker="s", label="MRR@10")
plt.axvline(data["best"]["alpha"], color="gray", linestyle="--", label=f"best α={data['best']['alpha']}")
plt.xlabel("α (BM25 weight)")
plt.ylabel("Score")
plt.title("FiQA-mini Hybrid Fusion Ablation")
plt.legend()
plt.tight_layout()
plt.savefig("results/fiqa_hybrid_ablation.png", dpi=150)
plt.show()
