from __future__ import annotations
import streamlit as st
import requests

st.set_page_config(page_title="willrec demo", layout="centered")
st.title("Neural Retrieval Engine (Hybrid) — FiQA-mini")

api = st.text_input("API base URL", value="http://127.0.0.1:8000")
q = st.text_input("Query", value="python programming")
mode = st.selectbox("Mode", ["hybrid", "dense", "bm25"], index=0)
k = st.slider("k", 1, 20, 10)
alpha = st.slider("alpha (hybrid only)", 0.0, 1.0, 0.5, 0.05)
norm = st.selectbox("Normalization", ["z", "minmax", "none"], index=0)

if st.button("Search"):
    params = {"q": q, "k": k, "mode": mode, "alpha": alpha, "norm": norm}
    try:
        r = requests.get(f"{api}/search", params=params, timeout=30)
        r.raise_for_status()
        data = r.json()
        st.write(f"**Mode:** {data.get('mode')}  |  **k:** {data.get('k')}  |  **alpha:** {data.get('alpha')}")
        for i, hit in enumerate(data["results"], 1):
            st.markdown(f"**{i}. {hit['doc_id']}**  \n`score={hit.get('score')}`  \n{hit['snippet']}")
            st.markdown("---")
    except Exception as e:
        st.error(f"Request failed: {e}")
