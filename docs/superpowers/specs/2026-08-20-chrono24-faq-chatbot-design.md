# Design: Chrono24-FAQ-Chatbot

**Datum:** 2026-08-20
**Status:** Entwurf validiert (Brainstorming abgeschlossen)

## Zweck

Ein RAG-Chatbot, der Fragen zu den öffentlichen Hilfeseiten von Chrono24
(FAQ unter `https://www.chrono24.de/info/faqs.htm` plus verlinkte
`/info/*`-Unterseiten) auf Deutsch beantwortet.

Primärer Zweck: **Portfolio-Projekt und Bewerbungsunterlage** (Ziel:
Bewerbung bei Chrono24). Die Demo muss öffentlich erreichbar, technisch
vorzeigbar und ehrlich sein. Produktionsbetrieb ist nicht das Ziel.

**Erfolgskriterien:**

- Öffentlicher Demo-Link, den ein Recruiter ohne Anleitung benutzen kann.
- Antworten stammen nachvollziehbar aus den Chrono24-Hilfeseiten, mit
  klickbaren Quellenangaben.
- Messbare Retrieval-Qualität (Hit-Rate@5 auf eigenem Eval-Set, Ergebnis
  im README).
- Laufende Kosten nahe null; Missbrauch der öffentlichen Demo begrenzt.

## Gewählter Ansatz

**Struktur-bewusstes Hybrid-RAG** (Ansatz B aus dem Brainstorming):

- FAQ-Einträge bleiben intakte Frage-Antwort-Paare; das Embedding wird
  auf der *Frage* berechnet, weil Nutzerfragen Fragen ähneln, nicht
  Fließtext.
- Info-Unterseiten werden an Überschriften in Chunks geteilt.
- Retrieval kombiniert BM25 (Keyword) und Vektorsuche per Reciprocal
  Rank Fusion.
- Claude Haiku generiert die Antwort ausschließlich aus dem gelieferten
  Kontext, mit nummerierten Quellen.

Verworfen: klassisches Chunk-RAG (verliert die Q&A-Struktur, schlechteres
Matching) und Agentic RAG mit Reranker/Multi-Step (Overkill für ~100
Dokumente; einzige Ausnahme siehe Query-Rewriting unten).

## Architektur

Zwei getrennte Läufe:

**Offline-Pipeline** (lokal, einmalig bzw. bei Bedarf):

```
scraper (Playwright) → data/raw/*.html
parser               → data/corpus.json   (versioniert im Repo)
indexer              → data/index/        (Chroma + BM25, versioniert)
```

**Online-Service** (Render Free-Tier, Docker):

```
Browser-Chat-UI (statisches HTML/JS, von FastAPI ausgeliefert)
   ↓ POST /api/chat (SSE-Stream zurück)
FastAPI
   ├─ Retrieval: BM25 + Vektor → RRF-Fusion → Top 5
   ├─ Claude Haiku: Antwort nur aus Kontext, mit Quellen
   └─ Guards: IP-Rate-Limit, Tages-Token-Budget
```

Der Server braucht zur Laufzeit kein Scraping und keinen Zugriff auf
Chrono24 — nur den fertigen Index aus dem Repo und den Anthropic-API-Key.

**Projektstruktur:**

```
chrono24-chatbot/
├─ pipeline/        # Scraper, Parser, Indexer (nur lokal ausgeführt)
├─ app/             # FastAPI: main, retrieval, llm, guards
├─ static/          # Chat-UI (Vanilla JS, kein Build-Schritt)
├─ data/            # corpus.json + Index (committed)
├─ docs/superpowers/specs/
├─ tests/
└─ Dockerfile
```

## Datenpipeline

### Scraper (`pipeline/scrape.py`)

- Playwright headless (umgeht den 403-Bot-Block, den einfache
  HTTP-Clients bekommen).
- Startpunkt FAQ-Seite; sammelt alle Links auf `/info/*`-Unterseiten,
  lädt jede Seite genau einmal, speichert Roh-HTML nach `data/raw/`.
- Höfliches Tempo: 1 Request/Sekunde; ehrlicher User-Agent.
- Vor dem ersten Lauf `robots.txt` prüfen und das Ergebnis im README
  dokumentieren.
- Läuft ausschließlich lokal, nie im Deployment.

### Parser (`pipeline/parse.py`)

- FAQ-Seite → Dokumenttyp `faq`: `question`, `answer`, `url`, `category`.
- Info-Unterseiten → Dokumenttyp `page_chunk`: `title`, `heading`,
  `text`, `url`. Split an `h2`/`h3`; Chunks über ~800 Tokens werden am
  Absatz weiter geteilt.
- Output: `data/corpus.json` — menschenlesbar, diffbar, versioniert.
- Validierung schlägt laut fehl (Mindestanzahl Einträge, Pflichtfelder
  vorhanden), damit HTML-Änderungen bei Chrono24 nicht still ein leeres
  oder kaputtes Korpus erzeugen.

### Indexer (`pipeline/index.py`)

- Embeddings: `paraphrase-multilingual-MiniLM-L12-v2`
  (sentence-transformers; lokal, gratis, deutsch + englisch, passt in
  den RAM des Render-Free-Tiers). Fallback, falls RAM doch nicht
  reicht: Embedding-API (Voyage/OpenAI) statt lokalem Modell.
- FAQ-Dokumente: Embedding auf `question`. Chunks: auf `heading + text`.
- Chroma persistent nach `data/index/`; BM25-Index (rank-bm25) als
  Pickle daneben. Beides committed, damit der Render-Build nichts
  berechnen muss.

## Retrieval und Antwort-Generierung

### Retrieval (`app/retrieval.py`)

1. Nutzerfrage → BM25-Top-10 und Vektor-Top-10.
2. Reciprocal Rank Fusion mischt beide Listen → Top 5 an das LLM.
3. Score-Schwelle: Liegt der beste Treffer unter dem Minimum, erfolgt
   kein LLM-Call; der Bot antwortet direkt „Dazu finde ich nichts in den
   Chrono24-Hilfeseiten.“ Das spart Kosten und verhindert Halluzination
   bei themenfremden Fragen.

### Generierung (`app/llm.py`)

- Modell: `claude-haiku-4-5`, Antwort gestreamt via SSE.
- System-Prompt: nur aus dem gelieferten Kontext antworten, auf Deutsch,
  kurz; fehlende Information ehrlich benennen; Quellen als `[1]`, `[2]`
  zitieren.
- Kontext = die 5 Treffer mit Nummern und URLs; das Frontend rendert die
  Quellen als klickbare Chrono24-Links unter der Antwort.
- Konversationsverlauf: die letzten ~6 Turns werden mitgeschickt.
  Retrieval läuft pro Turn auf der neuen Frage; kurze Folgefragen
  („und beim Verkauf?“) werden vorher durch einen billigen
  Haiku-Mini-Call in eine eigenständige Frage umgeschrieben
  (Query-Rewriting — die einzige „agentic“ Komponente).
- Transparenz-Feature: aufklappbares „Retrieval-Details“-Panel im UI
  zeigt die gefundenen Dokumente samt Scores.

## API, Frontend, Absicherung

### API (FastAPI)

- `POST /api/chat` — Body `{messages: [...]}`; Antwort als SSE-Stream
  (Token-Events, am Ende ein Quellen-Event und die Retrieval-Details).
- `GET /api/health` — für den Render-Healthcheck.
- Statisches Frontend wird unter `/` direkt von FastAPI ausgeliefert:
  ein Container, ein Deploy.
- Input-Validierung mit Pydantic: maximale Fragelänge 500 Zeichen,
  maximal 20 Messages Verlauf.

### Guards (`app/guards.py`)

- IP-Rate-Limit: 10 Anfragen/Minute, 50/Tag (slowapi).
- Globales Tagesbudget: Token-Zähler in einer SQLite-Datei; Deckel
  200k Tokens/Tag. Bei Überschreitung HTTP 429 mit der Meldung
  „Demo-Budget für heute erschöpft“.
- Anthropic-API-Key nur als Environment-Variable auf Render, nie im
  Repo; `.env.example` dokumentiert die nötigen Variablen.

### Frontend (`static/`, Vanilla JS)

- Kein Build-Schritt (konsistent mit marco-os).
- Chat-Verlauf, Streaming-Anzeige, Quellen-Links, aufklappbares
  Retrieval-Details-Panel, 3–4 anklickbare Beispielfragen als Einstieg.
- Deutschsprachig, responsiv; dunkles Theme optional passend zu
  marco-os, wo das Projekt später als Planet mit iframe/Link erscheint.
- Disclaimer im Footer: „Inoffizielles Portfolio-Projekt, nicht mit
  Chrono24 verbunden. Antworten ohne Gewähr. Quelle: öffentliche
  Hilfeseiten.“

### Deployment

- Dockerfile auf python-slim-Basis; Embedding-Modell wird beim Build
  gecacht.
- Render Free-Tier, Auto-Deploy vom GitHub-`main`-Branch. Kaltstart
  nach Schlafmodus (~30 s) ist akzeptiert.

## Testing und Fehlerbehandlung

### Tests (pytest, Ziel: 80 %+ auf `app/` und dem Parser)

- **Unit:** Parser (HTML-Fixtures → erwartete Dokumente), RRF-Fusion,
  Score-Schwelle, Budget-Zähler, Query-Rewriting-Prompt-Bau.
- **Integration:** `/api/chat` mit gemocktem Anthropic-Client (CI
  verbraucht keine API-Tokens), Rate-Limit-Verhalten, SSE-Format.
- **Retrieval-Eval:** ~30 handgeschriebene Testfragen mit erwartetem
  Quell-Dokument; misst Hit-Rate@5. Läuft als Skript; die
  Ergebnis-Tabelle kommt ins README und belegt zugleich die
  Entscheidung für Hybrid-RAG gegenüber reinem Chunk-RAG.
- **CI:** GitHub Actions — ruff (Lint), pytest, Docker-Build.

### Fehlerbehandlung

- Anthropic-API nicht erreichbar oder Timeout → SSE-Fehler-Event; UI
  zeigt „Antwort gerade nicht möglich, versuch's gleich nochmal“ mit
  Retry-Button.
- Fehlender Index beim Start → Service startet nicht (fail fast mit
  klarer Log-Zeile), statt leere Antworten zu liefern.
- Parser-Validierung verhindert stilles Kaputt-Scrapen (siehe oben).

## Offene Punkte / bewusste Annahmen

- `robots.txt`-Prüfung steht aus (erster Schritt der Pipeline-Arbeit);
  für eine Bewerbungs-Demo gegenüber Chrono24 wird respektvolles,
  einmaliges Scraping öffentlicher Hilfeseiten als vertretbar
  eingeschätzt und transparent dokumentiert.
- Sollte das MiniLM-Embedding-Modell den Render-RAM sprengen, wird auf
  eine Embedding-API umgestellt (Architektur lässt beides zu).
- Hosting-Alternative, falls Render enttäuscht: Hugging Face Spaces
  (Docker) — die Anwendung bleibt Docker-portabel.
