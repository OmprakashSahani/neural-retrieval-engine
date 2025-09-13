.PHONY: setup fmt lint test

setup:
	python -m pip install -U pip wheel
	pip install -e .[dev]

fmt:
	ruff check --fix .
	black .

lint:
	ruff check .
	black --check .
	mypy src

test:
	pytest -q

# --- CLI wrappers ---
toy-bm25:
	python scripts/cli.py toy bm25

toy-dense:
	python scripts/cli.py toy dense

toy-fuse:
	python scripts/cli.py toy fuse

ingest-fiqa-local:
	python scripts/cli.py ingest beir-local data/raw/fiqa --limit-docs=10000 --out-name=fiqa-mini

faiss-build:
	python scripts/fiqa_faiss_build.py

faiss-search:
	python scripts/fiqa_faiss_search.py

serve-api:
	python scripts/serve_api.py

serve-demo:
	streamlit run src/willrec/server/demo_app.py