FROM python:3.12-slim

WORKDIR /srv

COPY requirements.txt .
RUN pip install --no-cache-dir --extra-index-url https://download.pytorch.org/whl/cpu -r requirements.txt

# Embedding-Modell beim Build cachen, damit der Start schnell ist. Kein
# Cross-Encoder-Pre-Cache mehr: der LLM-Reranker-Pfad (Standard) laedt keinen
# Cross-Encoder, der Rollback-Pfad laedt ein anderes privates HF-Repo
# (VoidFloat/chrono24-faq-reranker) -- das MMARCO-Modell war totes Gewicht.
RUN python -c "from sentence_transformers import SentenceTransformer; \
    SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')"

COPY app/ app/
COPY pipeline/ pipeline/
COPY static/ static/
COPY data/corpus.json data/corpus.json
COPY data/variants.json data/variants.json

# Index wird beim Build lokal aus dem Corpus erzeugt (Embedding laeuft ohne
# API-Call), nicht mehr committet -- siehe
# docs/superpowers/specs/2026-08-23-corpus-storage-rethink-design.md Schritt 2.
RUN python -m pipeline.index

EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--forwarded-allow-ips=*"]
