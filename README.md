# Chrono24-FAQ-Chatbot

Ein RAG-Chatbot, der Fragen zu Kauf, Verkauf, Käuferschutz und Versand auf
Basis der öffentlichen Chrono24-Hilfeseiten beantwortet — mit Quellenangaben
und sichtbaren Retrieval-Treffern statt einer Blackbox-Antwort.

**Demo-Link:** folgt (Deploy steht noch aus).

## Disclaimer

Inoffizielles Portfolio-Projekt. Nicht mit Chrono24 verbunden, nicht von
Chrono24 autorisiert oder unterstützt. Antworten ohne Gewähr; Quelle ist
ausschließlich der öffentliche Hilfebereich von chrono24.de.

## Architektur

Zwei getrennte Läufe: eine lokale Offline-Pipeline baut den Index, der
Online-Service liest ihn nur noch — er scrapt zur Laufzeit nie.

```
Offline-Pipeline (lokal, einmalig bzw. bei Bedarf)
───────────────────────────────────────────────────
scraper (Playwright) → data/raw/*.html
parser               → data/corpus.json   (versioniert im Repo)
indexer              → data/index/        (Chroma + BM25, versioniert)

Online-Service (Docker)
───────────────────────────────────────────────────
Browser-Chat-UI (statisches HTML/JS, von FastAPI ausgeliefert)
   ↓ POST /api/chat (SSE-Stream zurück)
FastAPI
   ├─ Retrieval: BM25 + Vektor → RRF-Fusion → Top 5
   ├─ Claude Haiku: Antwort nur aus Kontext, mit Quellen
   └─ Guards: IP-Rate-Limit, Tages-Token-Budget
```

Der Server braucht zur Laufzeit kein Scraping und keinen Zugriff auf
Chrono24 — nur den fertigen Index aus dem Repo und den Anthropic-API-Key.
`data/corpus.json` ist ein zwischengespeicherter Snapshot öffentlicher
Chrono24-Hilfeseiten zu Demo-Zwecken, kein Live-Datenzugriff.

## Warum Hybrid-RAG

Jedes FAQ-Paar bleibt als eigenes Dokument erhalten statt in generische
Chunks zerschnitten zu werden — das erhält die Q&A-Struktur, und die
Vektorsuche embedded gezielt die Frage, nicht die Antwort, weil Nutzer:innen
selbst Fragen stellen (Frage-auf-Frage-Matching). BM25 fängt exakte
Begriffe (Produktnamen, Fachbegriffe) ab, die Embeddings manchmal
verwässern; ein RRF-Fusion aus beiden Rankings kombiniert die Stärken.

Gemessen mit einem handgeschriebenen 33-Fragen-Set (26 FAQ- und 7
Seiten-Chunk-Ziele, 3 davon englisch), iterativ verbessert — jede Stufe
wurde einzeln gemessen, auch die, die erstmal nichts bringt:

| Retrieval-Stufe | Hit-Rate@5 |
|---|---|
| Baseline: BM25 + Vektor + RRF | 76 % (25/33) |
| + Near-Duplicate-Merge der Seiten-Chunks (−5 Docs) | 73 % (24/33) |
| + Cross-Encoder-Reranker (mmarco-mMiniLMv2) über die RRF-Top-10 | **88 % (29/33)** |
| + Query-Übersetzung nicht-deutscher Fragen vor BM25 | 88 % (29/33) |

Der Dedupe-Schritt kostet solo einen BM25-Randfall, verhindert aber, dass
fast-identische Chunks die Top-5 verstopfen — zusammen mit dem Reranker
ist er neutral bis positiv. Die Query-Übersetzung ändert auf diesem Set
keinen Zähler (zwei der drei englischen Fragen trafen schon über die
multilingualen Embeddings), macht den Live-Pfad für englische Fragen aber
robust, weil BM25 sonst am deutschen Korpus vorbeiläuft. Die 4
verbleibenden Misses sind diagnostizierte harte Fälle (mehrdeutige
Zuordnung, z. B. „Certified" mit vielen nahen Kandidaten) — keine
geschönten Fragen, keine kaputte Konfidenzschwelle.

## Scraping-Ethik

`robots.txt` von chrono24.de wurde vor dem Scrape-Lauf geprüft: für
`User-agent: *` ist `/info/` nicht disallowed — kein Eintrag der
Disallow-Liste erfasst `/info/*`. `Crawl-delay: 0.1` verlangt mindestens
0,1 s zwischen Requests; der Scraper hält sich an 1 Request/Sekunde, also
deutlich höflicher als gefordert. Der Lauf war ein einmaliger lokaler
Vorgang (21 Seiten, mit echtem Chrome statt headless Chromium, weil
headless auf eine Cloudflare-Challenge lief) — der deployte Server scrapt
nie, er liest ausschließlich den versionierten Index.

## Lokal starten

```bash
python -m venv .venv
.venv/Scripts/pip install -r requirements-dev.txt
# .env anlegen (siehe .env.example) und ANTHROPIC_API_KEY eintragen
.venv/Scripts/uvicorn app.main:app --port 8000
```

Danach `http://localhost:8000` im Browser öffnen.

Chroma verändert Index-Dateien unter `data/index/` schon beim bloßen Öffnen
(z. B. durch lokales Starten oder Testläufe) — solche unstaged Änderungen
mit `git restore data/index/` verwerfen, bevor committet wird.

## Pipeline neu bauen

Nur nötig, wenn sich die Chrono24-Hilfeseiten geändert haben oder der Index
neu erzeugt werden soll:

```bash
python -m pipeline.scrape && python -m pipeline.parse && python -m pipeline.index
```

Bei einem chromadb-Upgrade (Version in `requirements.txt`) muss der Index
danach neu gebaut werden — das committete Format ist an die gepinnte Version
gekoppelt.

## Tests

```bash
pytest tests/
```

Retrieval-Qualität messen:

```bash
python -m eval.run_eval
```

## Guards

Öffentliche Demo, deshalb zwei Schutzmechanismen gegen Missbrauch/Kosten:

- **Rate-Limit:** 10 Anfragen/Minute und 50 Anfragen/Tag pro IP.
- **Tagesbudget:** 200.000 Tokens/Tag global, danach liefert `/api/chat`
  `429` bis zum nächsten Tag.

Das Rate-Limit greift pro Client-IP (`get_remote_address`). Hinter einem
Reverse Proxy wie auf Render sieht uvicorn ohne Weiteres nur die
Proxy-IP — damit teilen sich dann alle Besucher:innen einen einzigen
Rate-Limit-Eimer. Deshalb läuft uvicorn dort mit
`--forwarded-allow-ips="*"` (siehe Dockerfile-`CMD`), damit es die
`X-Forwarded-For`-Adresse des Proxys übernimmt. Lokal ohne Proxy ist das
wirkungslos, weil Clients den Server dort direkt erreichen. Trade-off: der
`X-Forwarded-For`-Header ist grundsätzlich spoofbar, das Rate-Limit ist also
kein hartes Sicherheitsnetz — das globale Tagesbudget bleibt der eigentliche
Kostendeckel, unabhängig von der IP.

## Deployment (Render)

Läuft als Docker-Runtime (siehe `Dockerfile`) auf Render.

- **RAM-Bedarf:** Embedding-Modell plus Cross-Encoder-Reranker brauchen
  zusammen deutlich mehr als die 512 MB des Render-Free-Tiers — für die
  Demo ist ein Tier mit ≥ 1 GB nötig oder alternativ Hugging Face Spaces
  (Docker).

- **Env-Var:** `ANTHROPIC_API_KEY` muss gesetzt sein — ohne ihn startet der
  Service gar nicht erst (fail-fast beim Boot statt kaputter Antworten zur
  Laufzeit).
- **Health-Check-Pfad:** `/api/health`.
- **Kaltstart:** auf dem Render-Free-Tier ca. 30 s, weil der Container nach
  Inaktivität einschläft.
- **Proxy:** `--forwarded-allow-ips="*"` ist im Dockerfile-`CMD` gesetzt,
  siehe Begründung im Guards-Abschnitt oben.
- **Tagesbudget ist nicht persistent:** der Token-Zähler liegt in SQLite auf
  dem ephemeren Dateisystem des Containers und resettet bei jedem Neustart
  oder Redeploy — das Tagesbudget ist also kein verlässlicher Kostendeckel
  über einen Neustart hinweg, sondern nur innerhalb einer laufenden
  Instanz.
