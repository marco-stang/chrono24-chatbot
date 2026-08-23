FROM python:3.12-slim

WORKDIR /srv

COPY requirements.txt .
RUN pip install --no-cache-dir --extra-index-url https://download.pytorch.org/whl/cpu -r requirements.txt

# Embedding- und Reranker-Modell beim Build cachen, damit der Start schnell ist
RUN python -c "from sentence_transformers import SentenceTransformer; \
    SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')"
RUN python -c "from sentence_transformers import CrossEncoder; \
    CrossEncoder('cross-encoder/mmarco-mMiniLMv2-L12-H384-v1')"

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
