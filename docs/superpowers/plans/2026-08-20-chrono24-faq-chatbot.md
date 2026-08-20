# Chrono24-FAQ-Chatbot Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** RAG-Chatbot, der Fragen zu den öffentlichen Chrono24-Hilfeseiten auf Deutsch beantwortet — mit Hybrid-Retrieval (BM25 + Vektor, RRF), Claude Haiku, SSE-Streaming-Web-UI und Guards für eine öffentliche Demo.

**Architecture:** Offline-Pipeline (Playwright-Scraper → Parser → Indexer, Ergebnisse versioniert im Repo) plus Online-Service (FastAPI mit statischem Vanilla-JS-Frontend, Retrieval + Claude-Streaming, Rate-Limit + Token-Budget). Docker-Deploy auf Render.

**Tech Stack:** Python 3.12, FastAPI, uvicorn, anthropic (AsyncAnthropic), chromadb, sentence-transformers (`paraphrase-multilingual-MiniLM-L12-v2`), rank-bm25, slowapi, BeautifulSoup, Playwright (nur Pipeline), pytest, ruff.

**Spec:** `docs/superpowers/specs/2026-08-20-chrono24-faq-chatbot-design.md`

## Global Constraints

- Python ≥ 3.11 (entwickelt auf 3.12).
- LLM-Modell: exakt `claude-haiku-4-5` (bewusste Kostenentscheidung; NICHT auf Opus/Sonnet hochstufen).
- Embedding-Modell: exakt `paraphrase-multilingual-MiniLM-L12-v2`.
- Alle UI-Texte und Bot-Antworten deutsch.
- Kein Secret im Repo; `ANTHROPIC_API_KEY` nur als Env-Var; `.env` in `.gitignore`.
- `data/corpus.json` und `data/index/` werden committed; `data/raw/` NICHT (in `.gitignore`).
- Scraper läuft nur lokal, nie im Deployment; 1 Request/Sekunde; Seeds: FAQ-Seite und `/info/index.htm` (Hilfe-Übersicht); Crawl genau eine Ebene tief (Links der Seed-Seiten, keine Rekursion).
- Frontend: Vanilla JS, kein Build-Schritt, keine Frameworks.
- Commits: Conventional Commits (`feat:`, `fix:`, `test:`, `chore:`, `docs:`, `ci:`).
- Lint: ruff mit Default-Regeln; Tests: pytest; Ziel 80 %+ Coverage auf `app/` und `pipeline/parse.py`.
- Fehler nie still schlucken; Validierung an Systemgrenzen (Pydantic, Korpus-Validierung).

---

### Task 1: Projektgerüst + Scraper

**Files:**
- Create: `pyproject.toml`, `.gitignore`, `.env.example`, `requirements.txt`, `requirements-dev.txt`
- Create: `pipeline/__init__.py`, `pipeline/scrape.py`
- Test: `tests/__init__.py`, `tests/test_scrape.py`

**Interfaces:**
- Consumes: nichts (erster Task).
- Produces:
  - `pipeline.scrape.BASE_URL: str` = `"https://www.chrono24.de"`
  - `pipeline.scrape.START_URL: str` = `"https://www.chrono24.de/info/faqs.htm"` (FAQ-Seite, braucht der Parser)
  - `pipeline.scrape.SEED_URLS: list[str]` = `[START_URL, "https://www.chrono24.de/info/index.htm"]`
  - `pipeline.scrape.collect_info_links(html: str, base_url: str) -> list[str]` — sortierte, deduplizierte absolute URLs auf `/info/*.htm`
  - `pipeline.scrape.url_to_filename(url: str) -> str` — z. B. `"info__faqs.htm.html"`
  - `pipeline.scrape.filename_to_url(name: str) -> str` — Umkehrfunktion
  - Manuell erzeugtes `data/raw/*.html` (lokal, nicht committed)

- [ ] **Step 1: Projektgerüst anlegen**

`pyproject.toml`:

```toml
[project]
name = "chrono24-chatbot"
version = "0.1.0"
requires-python = ">=3.11"

[tool.ruff]
line-length = 100

[tool.pytest.ini_options]
testpaths = ["tests"]
asyncio_mode = "auto"
```

`requirements.txt` (Laufzeit, kommt später auch in Docker):

```
fastapi
uvicorn[standard]
anthropic
chromadb
sentence-transformers
rank-bm25
slowapi
pydantic-settings
```

`requirements-dev.txt`:

```
-r requirements.txt
beautifulsoup4
playwright
pytest
pytest-asyncio
httpx
ruff
```

`.gitignore`:

```
__pycache__/
.venv/
.env
data/raw/
data/budget.sqlite3
.pytest_cache/
.ruff_cache/
*.egg-info/
```

`.env.example`:

```
ANTHROPIC_API_KEY=sk-ant-dein-key-hier
```

Leere Dateien: `pipeline/__init__.py`, `tests/__init__.py`.

Dann: `python -m venv .venv && .venv/Scripts/pip install -r requirements-dev.txt`

- [ ] **Step 2: Failing Test für Link-Sammlung und Dateinamen schreiben**

`tests/test_scrape.py`:

```python
from pipeline.scrape import collect_info_links, filename_to_url, url_to_filename

SAMPLE_HTML = """
<html><body>
<a href="/info/buyer-protection.htm">Käuferschutz</a>
<a href="https://www.chrono24.de/info/trusted-checkout.htm?x=1#top">Trusted Checkout</a>
<a href="/info/buyer-protection.htm">Duplikat</a>
<a href="/watches/rolex.htm">keine Info-Seite</a>
<a href="https://example.com/info/fremd.htm">fremde Domain</a>
<a href="/info/faqs.htm">FAQ selbst</a>
</body></html>
"""


def test_collect_info_links_filters_and_dedupes():
    links = collect_info_links(SAMPLE_HTML, "https://www.chrono24.de")
    assert links == [
        "https://www.chrono24.de/info/buyer-protection.htm",
        "https://www.chrono24.de/info/faqs.htm",
        "https://www.chrono24.de/info/trusted-checkout.htm",
    ]


def test_url_filename_roundtrip():
    url = "https://www.chrono24.de/info/buyer-protection.htm"
    name = url_to_filename(url)
    assert name == "info__buyer-protection.htm.html"
    assert filename_to_url(name) == url
```

- [ ] **Step 3: Test laufen lassen — muss fehlschlagen**

Run: `.venv/Scripts/python -m pytest tests/test_scrape.py -v`
Expected: FAIL mit `ModuleNotFoundError` bzw. `ImportError`.

- [ ] **Step 4: Scraper implementieren**

`pipeline/scrape.py`:

```python
"""Einmaliger lokaler Scraper für die Chrono24-Hilfeseiten. Läuft nie im Deployment."""
from __future__ import annotations

import asyncio
from pathlib import Path
from urllib.parse import urljoin, urlparse

from bs4 import BeautifulSoup

BASE_URL = "https://www.chrono24.de"
START_URL = f"{BASE_URL}/info/faqs.htm"
SEED_URLS = [START_URL, f"{BASE_URL}/info/index.htm"]
RAW_DIR = Path("data/raw")
REQUEST_DELAY_S = 1.0


def collect_info_links(html: str, base_url: str) -> list[str]:
    soup = BeautifulSoup(html, "html.parser")
    base_host = urlparse(base_url).netloc
    links: set[str] = set()
    for a in soup.find_all("a", href=True):
        href = urljoin(base_url, a["href"]).split("#")[0].split("?")[0]
        parsed = urlparse(href)
        if parsed.netloc == base_host and parsed.path.startswith("/info/") and parsed.path.endswith(".htm"):
            links.add(href)
    return sorted(links)


def url_to_filename(url: str) -> str:
    return urlparse(url).path.strip("/").replace("/", "__") + ".html"


def filename_to_url(name: str) -> str:
    return f"{BASE_URL}/" + name.removesuffix(".html").replace("__", "/")


async def scrape() -> None:
    """Lädt beide Seed-Seiten, sammelt deren /info/-Links und lädt jede Seite genau
    einmal. Bewusst nur eine Ebene tief — keine Rekursion."""
    from playwright.async_api import async_playwright

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    async with async_playwright() as pw:
        browser = await pw.chromium.launch(headless=True)
        page = await browser.new_page()
        saved: set[str] = set()
        queue: list[str] = list(SEED_URLS)
        queued: set[str] = set(SEED_URLS)
        while queue:
            url = queue.pop(0)
            if saved:
                await asyncio.sleep(REQUEST_DELAY_S)
            await page.goto(url, wait_until="networkidle")
            html = await page.content()
            (RAW_DIR / url_to_filename(url)).write_text(html, encoding="utf-8")
            saved.add(url)
            print(f"gespeichert: {url}")
            if url in SEED_URLS:
                for link in collect_info_links(html, BASE_URL):
                    if link not in queued:
                        queued.add(link)
                        queue.append(link)
        await browser.close()
        print(f"{len(saved)} Seiten gespeichert")


if __name__ == "__main__":
    asyncio.run(scrape())
```

- [ ] **Step 5: Tests laufen lassen — müssen bestehen**

Run: `.venv/Scripts/python -m pytest tests/test_scrape.py -v`
Expected: 2 PASS.

- [ ] **Step 6: robots.txt prüfen (manuell)**

`https://www.chrono24.de/robots.txt` im Browser öffnen (oder per Playwright-Einzeiler laden) und prüfen: Ist `/info/` für `User-agent: *` per `Disallow` gesperrt? Befund notieren — kommt in Task 11 ins README. **Wenn `/info/` disallowed ist: STOPP, Rücksprache mit Marco bevor gescrapt wird.**

- [ ] **Step 7: Scrape-Lauf ausführen (manuell, einmalig)**

Run: `.venv/Scripts/playwright install chromium` dann `.venv/Scripts/python -m pipeline.scrape`
Expected: `data/raw/` enthält `info__faqs.htm.html`, `info__index.htm.html` plus eine Datei je verlinkter Info-Seite (Erwartung: 10–80 Dateien). Stichprobe: eine Datei öffnen, prüfen dass echter Seiteninhalt (nicht Bot-Block-Seite) drinsteht — Bot-Block erkennt man an fehlendem Hilfetext.

- [ ] **Step 8: Commit**

```bash
git add pyproject.toml .gitignore .env.example requirements.txt requirements-dev.txt pipeline/ tests/
git commit -m "feat: Projektgerüst und Playwright-Scraper für Chrono24-Hilfeseiten"
```

---

### Task 2: FAQ-Parser

**Files:**
- Create: `pipeline/parse.py`
- Create: `tests/fixtures/faq_sample.html` (Ausschnitt aus echtem `data/raw/info__faqs.htm.html`)
- Test: `tests/test_parse_faq.py`

**Interfaces:**
- Consumes: `data/raw/info__faqs.htm.html` aus Task 1 (nur zum Fixture-Bau).
- Produces:
  - `pipeline.parse.parse_faq_page(html: str, url: str) -> list[dict]` — Dicts mit Keys `id` (`"faq-0001"`…), `type` (`"faq"`), `question`, `answer`, `category`, `url`
  - `pipeline.parse.CorpusValidationError(Exception)`

- [ ] **Step 1: Fixture aus echtem HTML bauen**

Aus `data/raw/info__faqs.htm.html` einen repräsentativen Ausschnitt (eine Kategorie-Überschrift + 2–3 Fragen mit Antworten, mitsamt umgebender Struktur) nach `tests/fixtures/faq_sample.html` kopieren. **Vorher die echte Struktur ansehen** — der Code unten nimmt an: Kategorien als `<h2>`, Fragen als `<h3>`, Antwort = Folge-Elemente bis zur nächsten Überschrift. Weicht die echte Struktur ab (z. B. Akkordeon-`<details>`, `<dt>/<dd>`), Fixture trotzdem 1:1 aus dem echten HTML übernehmen und die Selektoren in Step 4 an die echte Struktur anpassen — der Test bleibt gleich geformt (erwartete Fragen/Antworten aus dem Fixture ablesen).

- [ ] **Step 2: Failing Test schreiben**

`tests/test_parse_faq.py` (erwartete Strings an tatsächlichen Fixture-Inhalt anpassen):

```python
from pathlib import Path

from pipeline.parse import parse_faq_page

FIXTURE = Path("tests/fixtures/faq_sample.html").read_text(encoding="utf-8")
URL = "https://www.chrono24.de/info/faqs.htm"


def test_parse_faq_extracts_qa_pairs():
    docs = parse_faq_page(FIXTURE, URL)
    assert len(docs) >= 2
    first = docs[0]
    assert first["type"] == "faq"
    assert first["id"] == "faq-0001"
    assert first["url"] == URL
    assert first["question"].strip()
    assert first["answer"].strip()
    assert first["category"].strip()


def test_parse_faq_ids_are_unique_and_sequential():
    docs = parse_faq_page(FIXTURE, URL)
    ids = [d["id"] for d in docs]
    assert ids == [f"faq-{i:04d}" for i in range(1, len(ids) + 1)]
```

- [ ] **Step 3: Test laufen lassen — muss fehlschlagen**

Run: `.venv/Scripts/python -m pytest tests/test_parse_faq.py -v`
Expected: FAIL mit `ModuleNotFoundError: No module named 'pipeline.parse'`.

- [ ] **Step 4: Parser implementieren**

`pipeline/parse.py` (Selektoren ggf. an echte Fixture-Struktur anpassen, siehe Step 1):

```python
"""Parser: Roh-HTML der Hilfeseiten → strukturierte Korpus-Dokumente."""
from __future__ import annotations

from bs4 import BeautifulSoup


class CorpusValidationError(Exception):
    """Korpus unvollständig oder strukturell kaputt — laut scheitern statt leise."""


def _text_until_next_heading(tag) -> str:
    parts = []
    for sib in tag.find_next_siblings():
        if sib.name in ("h1", "h2", "h3"):
            break
        text = sib.get_text(" ", strip=True)
        if text:
            parts.append(text)
    return "\n".join(parts)


def parse_faq_page(html: str, url: str) -> list[dict]:
    soup = BeautifulSoup(html, "html.parser")
    docs: list[dict] = []
    category = ""
    for tag in soup.find_all(["h2", "h3"]):
        if tag.name == "h2":
            category = tag.get_text(strip=True)
            continue
        question = tag.get_text(strip=True)
        answer = _text_until_next_heading(tag)
        if not question or not answer:
            continue
        docs.append(
            {
                "id": f"faq-{len(docs) + 1:04d}",
                "type": "faq",
                "question": question,
                "answer": answer,
                "category": category,
                "url": url,
            }
        )
    return docs
```

- [ ] **Step 5: Test laufen lassen — muss bestehen**

Run: `.venv/Scripts/python -m pytest tests/test_parse_faq.py -v`
Expected: 2 PASS.

- [ ] **Step 6: Gegen echtes Voll-HTML prüfen**

Run: `.venv/Scripts/python -c "from pathlib import Path; from pipeline.parse import parse_faq_page; docs = parse_faq_page(Path('data/raw/info__faqs.htm.html').read_text(encoding='utf-8'), 'https://www.chrono24.de/info/faqs.htm'); print(len(docs), 'FAQ-Dokumente'); [print('-', d['question'][:70]) for d in docs[:5]]"`
Expected: plausible Anzahl (≥ 20) und echte Fragen als Ausgabe. Wenn 0 oder Müll: Selektoren anhand des echten HTML korrigieren, Steps 2–5 wiederholen.

- [ ] **Step 7: Commit**

```bash
git add pipeline/parse.py tests/fixtures/ tests/test_parse_faq.py
git commit -m "feat: FAQ-Parser mit Fixture aus echtem Chrono24-HTML"
```

---

### Task 3: Info-Seiten-Parser + Korpus-Bau + Validierung

**Files:**
- Modify: `pipeline/parse.py`
- Create: `tests/fixtures/info_sample.html` (Ausschnitt aus echter Info-Unterseite in `data/raw/`)
- Test: `tests/test_parse_info.py`, `tests/test_corpus.py`
- Produce: `data/corpus.json` (committed)

**Interfaces:**
- Consumes: `parse_faq_page`, `CorpusValidationError` (Task 2); `filename_to_url`, `START_URL`, `url_to_filename` (Task 1).
- Produces:
  - `pipeline.parse.parse_info_page(html: str, url: str) -> list[dict]` — Dicts mit `id` (`"<slug>-0001"`), `type` (`"page_chunk"`), `title`, `heading`, `text`, `url`
  - `pipeline.parse.build_corpus(raw_dir: Path) -> dict` — `{"scraped_at": "YYYY-MM-DD", "documents": [...]}`
  - `pipeline.parse.validate_corpus(corpus: dict) -> None` — wirft `CorpusValidationError`
  - `pipeline.parse.MIN_DOCS: int = 30`, `pipeline.parse.MAX_CHUNK_WORDS: int = 600`
  - CLI: `python -m pipeline.parse` schreibt validiertes `data/corpus.json`
  - Datei `data/corpus.json`

- [ ] **Step 1: Fixture bauen**

Aus einer echten Info-Seite in `data/raw/` (z. B. Käuferschutz) einen Ausschnitt mit `<title>`, mindestens zwei `h2`/`h3`-Abschnitten und einem langen Abschnitt (> 600 Wörter, notfalls Absatz duplizieren) nach `tests/fixtures/info_sample.html` kopieren.

- [ ] **Step 2: Failing Tests schreiben**

`tests/test_parse_info.py`:

```python
from pathlib import Path

from pipeline.parse import MAX_CHUNK_WORDS, parse_info_page

FIXTURE = Path("tests/fixtures/info_sample.html").read_text(encoding="utf-8")
URL = "https://www.chrono24.de/info/buyer-protection.htm"


def test_parse_info_page_builds_chunks():
    docs = parse_info_page(FIXTURE, URL)
    assert len(docs) >= 2
    for doc in docs:
        assert doc["type"] == "page_chunk"
        assert doc["title"].strip()
        assert doc["heading"].strip()
        assert doc["text"].strip()
        assert doc["url"] == URL


def test_long_sections_are_split():
    docs = parse_info_page(FIXTURE, URL)
    for doc in docs:
        assert len(doc["text"].split()) <= MAX_CHUNK_WORDS
```

`tests/test_corpus.py`:

```python
import pytest

from pipeline.parse import CorpusValidationError, validate_corpus


def make_corpus(docs):
    return {"scraped_at": "2026-08-20", "documents": docs}


def make_faq(i):
    return {"id": f"faq-{i:04d}", "type": "faq", "question": f"Frage {i}?",
            "answer": f"Antwort {i}.", "category": "Kaufen",
            "url": "https://www.chrono24.de/info/faqs.htm"}


def test_validate_accepts_valid_corpus():
    validate_corpus(make_corpus([make_faq(i) for i in range(1, 31)]))


def test_validate_rejects_too_few_docs():
    with pytest.raises(CorpusValidationError, match="Dokumente"):
        validate_corpus(make_corpus([make_faq(1)]))


def test_validate_rejects_missing_field():
    docs = [make_faq(i) for i in range(1, 31)]
    docs[0]["answer"] = ""
    with pytest.raises(CorpusValidationError, match="answer"):
        validate_corpus(make_corpus(docs))


def test_validate_rejects_duplicate_ids():
    docs = [make_faq(i) for i in range(1, 31)]
    docs[1]["id"] = docs[0]["id"]
    with pytest.raises(CorpusValidationError, match="doppelte"):
        validate_corpus(make_corpus(docs))


def test_validate_rejects_unknown_type():
    docs = [make_faq(i) for i in range(1, 31)]
    docs[0]["type"] = "blogpost"
    with pytest.raises(CorpusValidationError, match="Typ"):
        validate_corpus(make_corpus(docs))
```

- [ ] **Step 3: Tests laufen lassen — müssen fehlschlagen**

Run: `.venv/Scripts/python -m pytest tests/test_parse_info.py tests/test_corpus.py -v`
Expected: FAIL mit `ImportError` (Funktionen existieren noch nicht).

- [ ] **Step 4: Implementieren**

In `pipeline/parse.py` ergänzen:

```python
import json
from datetime import date
from pathlib import Path
from urllib.parse import urlparse

MIN_DOCS = 30
MAX_CHUNK_WORDS = 600

REQUIRED_FIELDS = {
    "faq": ("id", "question", "answer", "category", "url"),
    "page_chunk": ("id", "title", "heading", "text", "url"),
}


def _slug(url: str) -> str:
    return urlparse(url).path.strip("/").replace("/", "-").removesuffix(".htm")


def _split_long_text(text: str, max_words: int = MAX_CHUNK_WORDS) -> list[str]:
    if len(text.split()) <= max_words:
        return [text]
    parts: list[str] = []
    current: list[str] = []
    count = 0
    for paragraph in text.split("\n"):
        words = len(paragraph.split())
        if current and count + words > max_words:
            parts.append("\n".join(current))
            current, count = [], 0
        current.append(paragraph)
        count += words
    if current:
        parts.append("\n".join(current))
    return parts


def parse_info_page(html: str, url: str) -> list[dict]:
    soup = BeautifulSoup(html, "html.parser")
    title = soup.title.get_text(strip=True) if soup.title else url
    docs: list[dict] = []
    for tag in soup.find_all(["h2", "h3"]):
        heading = tag.get_text(strip=True)
        text = _text_until_next_heading(tag)
        if not heading or not text:
            continue
        for part in _split_long_text(text):
            docs.append(
                {
                    "id": f"{_slug(url)}-{len(docs) + 1:04d}",
                    "type": "page_chunk",
                    "title": title,
                    "heading": heading,
                    "text": part,
                    "url": url,
                }
            )
    return docs


def build_corpus(raw_dir: Path) -> dict:
    from pipeline.scrape import START_URL, filename_to_url, url_to_filename

    faq_name = url_to_filename(START_URL)
    docs = parse_faq_page((raw_dir / faq_name).read_text(encoding="utf-8"), START_URL)
    for path in sorted(raw_dir.glob("*.html")):
        if path.name == faq_name:
            continue
        docs = docs + parse_info_page(path.read_text(encoding="utf-8"), filename_to_url(path.name))
    return {"scraped_at": date.today().isoformat(), "documents": docs}


def validate_corpus(corpus: dict) -> None:
    docs = corpus.get("documents", [])
    if len(docs) < MIN_DOCS:
        raise CorpusValidationError(f"nur {len(docs)} Dokumente, erwartet mindestens {MIN_DOCS}")
    seen: set[str] = set()
    for doc in docs:
        fields = REQUIRED_FIELDS.get(doc.get("type", ""))
        if fields is None:
            raise CorpusValidationError(f"unbekannter Typ: {doc.get('type')!r} (id={doc.get('id')!r})")
        for field in fields:
            if not doc.get(field):
                raise CorpusValidationError(f"Dokument {doc.get('id')!r}: Feld {field!r} fehlt oder leer")
        if doc["id"] in seen:
            raise CorpusValidationError(f"doppelte id {doc['id']!r}")
        seen.add(doc["id"])


if __name__ == "__main__":
    corpus = build_corpus(Path("data/raw"))
    validate_corpus(corpus)
    out = Path("data/corpus.json")
    out.write_text(json.dumps(corpus, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"{len(corpus['documents'])} Dokumente nach {out} geschrieben")
```

- [ ] **Step 5: Tests laufen lassen — müssen bestehen**

Run: `.venv/Scripts/python -m pytest tests/ -v`
Expected: alle PASS.

- [ ] **Step 6: Korpus bauen und committen**

Run: `.venv/Scripts/python -m pipeline.parse`
Expected: Meldung mit Dokumentanzahl ≥ 30. `data/corpus.json` stichprobenartig lesen: echte Inhalte, saubere Umlaute (`ensure_ascii=False`).

```bash
git add pipeline/parse.py tests/ data/corpus.json
git commit -m "feat: Info-Seiten-Parser, Korpus-Bau mit Validierung, corpus.json"
```

---

### Task 4: Indexer (Chroma + BM25)

**Files:**
- Create: `app/__init__.py`, `app/config.py`, `app/textproc.py`, `pipeline/index.py`
- Test: `tests/test_textproc.py`, `tests/test_index.py`
- Produce: `data/index/` (committed)

**Interfaces:**
- Consumes: `data/corpus.json` (Task 3).
- Produces:
  - `app.config.Settings(BaseSettings)` mit Feldern `anthropic_api_key: str = ""`, `model: str = "claude-haiku-4-5"`, `embed_model: str = "paraphrase-multilingual-MiniLM-L12-v2"`, `index_dir: Path = Path("data/index")`, `corpus_path: Path = Path("data/corpus.json")`, `daily_token_budget: int = 200_000`, `budget_db: Path = Path("data/budget.sqlite3")`; Modul-Singleton `app.config.settings`
  - `app.textproc.tokenize(text: str) -> list[str]` — lowercase, deutsche Umlaute erhalten
  - `pipeline.index.doc_embed_text(doc: dict) -> str` und `pipeline.index.doc_search_text(doc: dict) -> str`
  - `pipeline.index.build_index(corpus_path: Path, index_dir: Path, encoder=None) -> None` — `encoder: Callable[[list[str]], list[list[float]]]`, Default: SentenceTransformer
  - Auf Platte: `data/index/chroma/` (Collection `"docs"`, cosine) und `data/index/bm25.pkl` (Pickle-Dict `{"doc_ids": list[str], "bm25": BM25Okapi}`)
  - CLI: `python -m pipeline.index`

- [ ] **Step 1: Failing Tests schreiben**

`tests/test_textproc.py`:

```python
from app.textproc import tokenize


def test_tokenize_lowercases_and_keeps_umlauts():
    assert tokenize("Käuferschutz greift SOFORT!") == ["käuferschutz", "greift", "sofort"]


def test_tokenize_splits_on_punctuation_and_digits_stay():
    assert tokenize("Artikel 14 Tage Rückgabe.") == ["artikel", "14", "tage", "rückgabe"]
```

`tests/test_index.py`:

```python
import json
import pickle

from pipeline.index import build_index, doc_embed_text, doc_search_text

FAQ = {"id": "faq-0001", "type": "faq", "question": "Wie funktioniert der Käuferschutz?",
       "answer": "Er sichert Zahlungen ab.", "category": "Kaufen",
       "url": "https://www.chrono24.de/info/faqs.htm"}
CHUNK = {"id": "info-buyer-protection-0001", "type": "page_chunk", "title": "Käuferschutz",
         "heading": "Ablauf", "text": "Der Ablauf ist einfach.",
         "url": "https://www.chrono24.de/info/buyer-protection.htm"}


def fake_encoder(texts):
    return [[1.0, 0.0] if "Käuferschutz?" in t else [0.0, 1.0] for t in texts]


def test_embed_text_uses_question_for_faq():
    assert doc_embed_text(FAQ) == "Wie funktioniert der Käuferschutz?"
    assert doc_embed_text(CHUNK) == "Ablauf\nDer Ablauf ist einfach."


def test_search_text_includes_answer():
    assert "sichert Zahlungen" in doc_search_text(FAQ)


def test_build_index_writes_chroma_and_bm25(tmp_path):
    corpus_path = tmp_path / "corpus.json"
    corpus_path.write_text(json.dumps({"scraped_at": "2026-08-20", "documents": [FAQ, CHUNK]}),
                           encoding="utf-8")
    index_dir = tmp_path / "index"
    build_index(corpus_path, index_dir, encoder=fake_encoder)

    import chromadb
    coll = chromadb.PersistentClient(path=str(index_dir / "chroma")).get_collection("docs")
    assert coll.count() == 2
    res = coll.query(query_embeddings=[[1.0, 0.0]], n_results=1)
    assert res["ids"][0] == ["faq-0001"]

    with open(index_dir / "bm25.pkl", "rb") as f:
        data = pickle.load(f)
    assert data["doc_ids"] == ["faq-0001", "info-buyer-protection-0001"]
```

- [ ] **Step 2: Tests laufen lassen — müssen fehlschlagen**

Run: `.venv/Scripts/python -m pytest tests/test_textproc.py tests/test_index.py -v`
Expected: FAIL mit `ModuleNotFoundError`.

- [ ] **Step 3: Implementieren**

`app/__init__.py`: leer.

`app/config.py`:

```python
from pathlib import Path

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    anthropic_api_key: str = ""
    model: str = "claude-haiku-4-5"
    embed_model: str = "paraphrase-multilingual-MiniLM-L12-v2"
    index_dir: Path = Path("data/index")
    corpus_path: Path = Path("data/corpus.json")
    daily_token_budget: int = 200_000
    budget_db: Path = Path("data/budget.sqlite3")

    model_config = {"env_file": ".env"}


settings = Settings()
```

`app/textproc.py`:

```python
import re

TOKEN_RE = re.compile(r"[a-zäöüß0-9]+")


def tokenize(text: str) -> list[str]:
    return TOKEN_RE.findall(text.lower())
```

`pipeline/index.py`:

```python
"""Baut Vektor- (Chroma) und Keyword-Index (BM25) aus data/corpus.json."""
from __future__ import annotations

import json
import pickle
from pathlib import Path

import chromadb
from rank_bm25 import BM25Okapi

from app.config import settings
from app.textproc import tokenize


def doc_embed_text(doc: dict) -> str:
    if doc["type"] == "faq":
        return doc["question"]
    return f"{doc['heading']}\n{doc['text']}"


def doc_search_text(doc: dict) -> str:
    if doc["type"] == "faq":
        return f"{doc['question']}\n{doc['answer']}"
    return f"{doc['heading']}\n{doc['text']}"


def _default_encoder(texts: list[str]) -> list[list[float]]:
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(settings.embed_model)
    return model.encode(texts, normalize_embeddings=True).tolist()


def build_index(corpus_path: Path, index_dir: Path, encoder=None) -> None:
    encoder = encoder or _default_encoder
    docs = json.loads(corpus_path.read_text(encoding="utf-8"))["documents"]
    index_dir.mkdir(parents=True, exist_ok=True)

    client = chromadb.PersistentClient(path=str(index_dir / "chroma"))
    try:
        client.delete_collection("docs")
    except Exception:
        pass
    coll = client.create_collection("docs", metadata={"hnsw:space": "cosine"})
    coll.add(ids=[d["id"] for d in docs], embeddings=encoder([doc_embed_text(d) for d in docs]))

    bm25 = BM25Okapi([tokenize(doc_search_text(d)) for d in docs])
    with open(index_dir / "bm25.pkl", "wb") as f:
        pickle.dump({"doc_ids": [d["id"] for d in docs], "bm25": bm25}, f)


if __name__ == "__main__":
    build_index(settings.corpus_path, settings.index_dir)
    print(f"Index nach {settings.index_dir} geschrieben")
```

- [ ] **Step 4: Tests laufen lassen — müssen bestehen**

Run: `.venv/Scripts/python -m pytest tests/test_textproc.py tests/test_index.py -v`
Expected: alle PASS.

- [ ] **Step 5: Echten Index bauen und committen**

Run: `.venv/Scripts/python -m pipeline.index` (erster Lauf lädt das Embedding-Modell herunter, dauert einige Minuten).
Expected: `data/index/chroma/` und `data/index/bm25.pkl` existieren.

```bash
git add app/ pipeline/index.py tests/test_textproc.py tests/test_index.py data/index/
git commit -m "feat: Indexer mit Chroma-Vektoren und BM25, versionierter Index"
```

---

### Task 5: Hybrid-Retrieval mit RRF

**Files:**
- Create: `app/retrieval.py`
- Test: `tests/test_retrieval.py`

**Interfaces:**
- Consumes: `data/index/` (Task 4), `app.textproc.tokenize`, `app.config.settings`, Korpus-Dokumentformate (Task 3).
- Produces:
  - `app.retrieval.RetrievedDoc` — Dataclass: `id: str`, `type: str`, `title: str`, `url: str`, `text: str`, `score: float`
  - `app.retrieval.rrf_fuse(rankings: list[list[str]], k: int = 60) -> dict[str, float]`
  - `app.retrieval.Retriever(index_dir: Path, corpus_path: Path, encoder=None)` mit `retrieve(query: str, top_k: int = 5) -> list[RetrievedDoc]` — leere Liste bei niedriger Konfidenz; `encoder: Callable[[str], list[float]]`
  - Konstanten: `TOP_K_CANDIDATES = 10`, `RRF_K = 60`, `SIM_THRESHOLD = 0.35`, `BM25_THRESHOLD = 4.0`

- [ ] **Step 1: Failing Tests schreiben**

`tests/test_retrieval.py` — Wichtig am Fake-Encoder: der Default-Vektor für unbekannte Texte ist `[-1.0, 0.0, 0.0]`, damit Off-Topic-Fragen zu allen Dokument-Vektoren Cosine-Similarity ≤ 0 haben und das Konfidenz-Gate greift:

```python
import json

import pytest

from app.retrieval import RetrievedDoc, Retriever, rrf_fuse
from pipeline.index import build_index

DOCS = [
    {"id": "faq-0001", "type": "faq", "question": "Wie funktioniert der Käuferschutz?",
     "answer": "Der Käuferschutz sichert deine Zahlung ab.", "category": "Kaufen",
     "url": "https://www.chrono24.de/info/faqs.htm"},
    {"id": "faq-0002", "type": "faq", "question": "Wie verkaufe ich eine Uhr?",
     "answer": "Über ein Verkäuferkonto.", "category": "Verkaufen",
     "url": "https://www.chrono24.de/info/faqs.htm"},
    {"id": "info-shipping-0001", "type": "page_chunk", "title": "Versand",
     "heading": "Versicherter Versand", "text": "Uhren werden versichert verschickt.",
     "url": "https://www.chrono24.de/info/shipping.htm"},
]

DOC_VECS = {"Käuferschutz": [1.0, 0.0, 0.0], "verkaufe": [0.0, 1.0, 0.0],
            "Versand": [0.0, 0.0, 1.0]}


def encode_one(text):
    for key, vec in DOC_VECS.items():
        if key in text:
            return vec
    return [-1.0, 0.0, 0.0]


@pytest.fixture()
def retriever(tmp_path):
    corpus_path = tmp_path / "corpus.json"
    corpus_path.write_text(json.dumps({"scraped_at": "2026-08-20", "documents": DOCS}),
                           encoding="utf-8")
    index_dir = tmp_path / "index"
    build_index(corpus_path, index_dir, encoder=lambda texts: [encode_one(t) for t in texts])
    return Retriever(index_dir, corpus_path, encoder=encode_one)


def test_rrf_fuse_rewards_docs_in_both_rankings():
    scores = rrf_fuse([["a", "b"], ["b", "c"]], k=60)
    assert scores["b"] > scores["a"]
    assert scores["b"] > scores["c"]
    assert scores["b"] == pytest.approx(1 / 62 + 1 / 61)


def test_retrieve_finds_matching_faq(retriever):
    docs = retriever.retrieve("Wie funktioniert der Käuferschutz?")
    assert docs
    assert isinstance(docs[0], RetrievedDoc)
    assert docs[0].id == "faq-0001"
    assert docs[0].title == "Wie funktioniert der Käuferschutz?"
    assert "sichert deine Zahlung" in docs[0].text


def test_retrieve_returns_empty_for_offtopic(retriever):
    docs = retriever.retrieve("Gedicht über Katzen bitte")
    assert docs == []
```

Hinweis: „Gedicht über Katzen bitte" enthält keines der BM25-Tokens der drei Dokumente und keinen `DOC_VECS`-Schlüssel — beide Signale bleiben unter den Schwellen.

- [ ] **Step 2: Tests laufen lassen — müssen fehlschlagen**

Run: `.venv/Scripts/python -m pytest tests/test_retrieval.py -v`
Expected: FAIL mit `ModuleNotFoundError: No module named 'app.retrieval'`.

- [ ] **Step 3: Implementieren**

`app/retrieval.py`:

```python
"""Hybrid-Retrieval: BM25 + Vektorsuche, fusioniert per Reciprocal Rank Fusion."""
from __future__ import annotations

import json
import pickle
from dataclasses import dataclass
from pathlib import Path

import chromadb

from app.config import settings
from app.textproc import tokenize

TOP_K_CANDIDATES = 10
RRF_K = 60
# Konfidenz-Gate: liegen BEIDE Signale unter ihrer Schwelle, gilt die Frage als
# themenfremd und es gibt keinen LLM-Call. Nach Task 10 (Eval) nachjustieren.
SIM_THRESHOLD = 0.35
BM25_THRESHOLD = 4.0


@dataclass
class RetrievedDoc:
    id: str
    type: str
    title: str
    url: str
    text: str
    score: float


def rrf_fuse(rankings: list[list[str]], k: int = RRF_K) -> dict[str, float]:
    scores: dict[str, float] = {}
    for ranking in rankings:
        for rank, doc_id in enumerate(ranking):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank + 1)
    return scores


def _default_encoder():
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(settings.embed_model)
    return lambda text: model.encode([text], normalize_embeddings=True)[0].tolist()


class Retriever:
    def __init__(self, index_dir: Path, corpus_path: Path, encoder=None):
        self.encoder = encoder or _default_encoder()
        client = chromadb.PersistentClient(path=str(index_dir / "chroma"))
        self.collection = client.get_collection("docs")
        with open(index_dir / "bm25.pkl", "rb") as f:
            data = pickle.load(f)
        self.bm25 = data["bm25"]
        self.doc_ids: list[str] = data["doc_ids"]
        corpus = json.loads(corpus_path.read_text(encoding="utf-8"))
        self.docs = {d["id"]: d for d in corpus["documents"]}

    def retrieve(self, query: str, top_k: int = 5) -> list[RetrievedDoc]:
        n = min(TOP_K_CANDIDATES, len(self.doc_ids))
        res = self.collection.query(query_embeddings=[list(self.encoder(query))], n_results=n)
        vector_ranking = res["ids"][0]
        best_sim = 1.0 - res["distances"][0][0] if res["distances"][0] else 0.0

        bm25_scores = self.bm25.get_scores(tokenize(query))
        order = sorted(range(len(bm25_scores)), key=lambda i: bm25_scores[i], reverse=True)
        bm25_ranking = [self.doc_ids[i] for i in order[:n] if bm25_scores[i] > 0]
        best_bm25 = bm25_scores[order[0]] if len(order) else 0.0

        if best_sim < SIM_THRESHOLD and best_bm25 < BM25_THRESHOLD:
            return []

        fused = rrf_fuse([vector_ranking, bm25_ranking])
        top = sorted(fused.items(), key=lambda kv: kv[1], reverse=True)[:top_k]
        return [self._to_doc(doc_id, score) for doc_id, score in top]

    def _to_doc(self, doc_id: str, score: float) -> RetrievedDoc:
        doc = self.docs[doc_id]
        if doc["type"] == "faq":
            title, text = doc["question"], doc["answer"]
        else:
            title, text = f"{doc['title']} — {doc['heading']}", doc["text"]
        return RetrievedDoc(id=doc_id, type=doc["type"], title=title, url=doc["url"],
                            text=text, score=round(score, 4))
```

- [ ] **Step 4: Tests laufen lassen — müssen bestehen**

Run: `.venv/Scripts/python -m pytest tests/test_retrieval.py -v`
Expected: 3 PASS.

- [ ] **Step 5: Manuelle Stichprobe gegen echten Index**

Run: `.venv/Scripts/python -c "from app.config import settings; from app.retrieval import Retriever; r = Retriever(settings.index_dir, settings.corpus_path); [print(d.score, d.id, d.title[:60]) for d in r.retrieve('Wie funktioniert der Käuferschutz?')]"`
Expected: plausible Treffer, Käuferschutz-Dokumente vorn.

- [ ] **Step 6: Commit**

```bash
git add app/retrieval.py tests/test_retrieval.py
git commit -m "feat: Hybrid-Retrieval mit RRF-Fusion und Konfidenz-Gate"
```

---

### Task 6: LLM-Modul (Kontext, Query-Rewriting, Streaming)

**Files:**
- Create: `app/llm.py`
- Test: `tests/test_llm.py`

**Interfaces:**
- Consumes: `app.retrieval.RetrievedDoc` (Task 5), `app.config.settings`.
- Produces:
  - `app.llm.SYSTEM_PROMPT: str`, `app.llm.REWRITE_SYSTEM: str`
  - `app.llm.build_context(docs: list[RetrievedDoc]) -> str` — nummerierte Blöcke `[1]…`
  - `app.llm.build_rewrite_prompt(history: list[dict], question: str) -> str`
  - `app.llm.get_client() -> AsyncAnthropic` — wirft `RuntimeError` ohne API-Key
  - `app.llm.rewrite_query(history: list[dict], question: str, client) -> str` (async)
  - `app.llm.stream_answer(question: str, docs: list[RetrievedDoc], history: list[dict], client) -> AsyncIterator[dict]` (async) — yielded Events: `{"type": "token", "text": str}` und final `{"type": "usage", "input_tokens": int, "output_tokens": int}`
  - `history`-Format überall: `[{"role": "user"|"assistant", "content": str}, ...]`

- [ ] **Step 1: Failing Tests schreiben**

`tests/test_llm.py`:

```python
from app.llm import build_context, build_rewrite_prompt, rewrite_query
from app.retrieval import RetrievedDoc

DOCS = [
    RetrievedDoc(id="faq-0001", type="faq", title="Wie funktioniert der Käuferschutz?",
                 url="https://www.chrono24.de/info/faqs.htm",
                 text="Der Käuferschutz sichert deine Zahlung ab.", score=0.05),
    RetrievedDoc(id="info-shipping-0001", type="page_chunk", title="Versand — Versichert",
                 url="https://www.chrono24.de/info/shipping.htm",
                 text="Uhren werden versichert verschickt.", score=0.03),
]


def test_build_context_numbers_docs_with_urls():
    context = build_context(DOCS)
    assert "[1] Wie funktioniert der Käuferschutz?" in context
    assert "[2] Versand — Versichert" in context
    assert "https://www.chrono24.de/info/shipping.htm" in context
    assert "sichert deine Zahlung" in context


def test_build_rewrite_prompt_contains_history_and_question():
    history = [{"role": "user", "content": "Wie kaufe ich eine Uhr?"},
               {"role": "assistant", "content": "Über die Plattform."}]
    prompt = build_rewrite_prompt(history, "und beim Verkauf?")
    assert "Wie kaufe ich eine Uhr?" in prompt
    assert "und beim Verkauf?" in prompt


async def test_rewrite_query_without_history_returns_question():
    result = await rewrite_query([], "Wie funktioniert der Käuferschutz?", client=None)
    assert result == "Wie funktioniert der Käuferschutz?"


class FakeTextBlock:
    type = "text"

    def __init__(self, text):
        self.text = text


class FakeResponse:
    def __init__(self, text):
        self.content = [FakeTextBlock(text)]


class FakeMessages:
    async def create(self, **kwargs):
        return FakeResponse("Wie verkaufe ich eine Uhr auf Chrono24?")


class FakeClient:
    messages = FakeMessages()


async def test_rewrite_query_with_history_calls_llm():
    history = [{"role": "user", "content": "Wie kaufe ich eine Uhr?"}]
    result = await rewrite_query(history, "und beim Verkauf?", client=FakeClient())
    assert result == "Wie verkaufe ich eine Uhr auf Chrono24?"
```

- [ ] **Step 2: Tests laufen lassen — müssen fehlschlagen**

Run: `.venv/Scripts/python -m pytest tests/test_llm.py -v`
Expected: FAIL mit `ModuleNotFoundError: No module named 'app.llm'`.

- [ ] **Step 3: Implementieren**

`app/llm.py`:

```python
"""Anbindung an Claude: Kontextbau, Query-Rewriting, gestreamte Antwort."""
from __future__ import annotations

from collections.abc import AsyncIterator

from anthropic import AsyncAnthropic

from app.config import settings
from app.retrieval import RetrievedDoc

SYSTEM_PROMPT = (
    "Du bist ein Assistent für Fragen zu den Hilfeseiten von Chrono24. "
    "Beantworte die Frage AUSSCHLIESSLICH mit Informationen aus dem gelieferten Kontext. "
    "Antworte auf Deutsch, kurz und präzise. "
    "Belege Aussagen mit den Quellennummern in eckigen Klammern, z. B. [1] oder [2]. "
    "Steht die Antwort nicht im Kontext, sage ehrlich: "
    "'Dazu finde ich nichts in den Chrono24-Hilfeseiten.' Erfinde nichts."
)

REWRITE_SYSTEM = (
    "Du erhältst einen Chatverlauf und eine Folgefrage. "
    "Formuliere die Folgefrage als eigenständige, vollständige Frage auf Deutsch um, "
    "sodass sie ohne den Verlauf verständlich ist. "
    "Antworte NUR mit der umformulierten Frage, ohne Erklärung."
)

MAX_ANSWER_TOKENS = 1024
MAX_REWRITE_TOKENS = 200

_client: AsyncAnthropic | None = None


def get_client() -> AsyncAnthropic:
    global _client
    if _client is None:
        if not settings.anthropic_api_key:
            raise RuntimeError("ANTHROPIC_API_KEY ist nicht gesetzt")
        _client = AsyncAnthropic(api_key=settings.anthropic_api_key)
    return _client


def build_context(docs: list[RetrievedDoc]) -> str:
    parts = [f"[{i}] {doc.title}\nURL: {doc.url}\n{doc.text}" for i, doc in enumerate(docs, 1)]
    return "\n\n".join(parts)


def build_rewrite_prompt(history: list[dict], question: str) -> str:
    lines = [f"{m['role']}: {m['content']}" for m in history]
    return "Chatverlauf:\n" + "\n".join(lines) + f"\n\nFolgefrage: {question}"


async def rewrite_query(history: list[dict], question: str, client) -> str:
    if not history:
        return question
    response = await client.messages.create(
        model=settings.model,
        max_tokens=MAX_REWRITE_TOKENS,
        system=REWRITE_SYSTEM,
        messages=[{"role": "user", "content": build_rewrite_prompt(history, question)}],
    )
    text = next((b.text for b in response.content if b.type == "text"), "").strip()
    return text or question


async def stream_answer(
    question: str, docs: list[RetrievedDoc], history: list[dict], client
) -> AsyncIterator[dict]:
    context = build_context(docs)
    messages = history + [
        {"role": "user", "content": f"Kontext:\n{context}\n\nFrage: {question}"}
    ]
    async with client.messages.stream(
        model=settings.model,
        max_tokens=MAX_ANSWER_TOKENS,
        system=SYSTEM_PROMPT,
        messages=messages,
    ) as stream:
        async for text in stream.text_stream:
            yield {"type": "token", "text": text}
        final = await stream.get_final_message()
    yield {
        "type": "usage",
        "input_tokens": final.usage.input_tokens,
        "output_tokens": final.usage.output_tokens,
    }
```

- [ ] **Step 4: Tests laufen lassen — müssen bestehen**

Run: `.venv/Scripts/python -m pytest tests/test_llm.py -v`
Expected: 5 PASS.

- [ ] **Step 5: Commit**

```bash
git add app/llm.py tests/test_llm.py
git commit -m "feat: LLM-Modul mit Kontextbau, Query-Rewriting und Streaming"
```

---

### Task 7: Guards (Token-Budget)

**Files:**
- Create: `app/guards.py`
- Test: `tests/test_guards.py`

**Interfaces:**
- Consumes: nichts Projektinternes.
- Produces:
  - `app.guards.TokenBudget(db_path: Path, daily_limit: int)` mit `spend(tokens: int) -> None`, `used_today() -> int`, `remaining() -> int`
  - (Das IP-Rate-Limit kommt in Task 8 direkt per slowapi an den Endpoint.)

- [ ] **Step 1: Failing Tests schreiben**

`tests/test_guards.py`:

```python
from app.guards import TokenBudget


def test_budget_starts_full(tmp_path):
    budget = TokenBudget(tmp_path / "b.sqlite3", daily_limit=1000)
    assert budget.remaining() == 1000
    assert budget.used_today() == 0


def test_spend_reduces_remaining(tmp_path):
    budget = TokenBudget(tmp_path / "b.sqlite3", daily_limit=1000)
    budget.spend(300)
    budget.spend(200)
    assert budget.used_today() == 500
    assert budget.remaining() == 500


def test_budget_persists_across_instances(tmp_path):
    path = tmp_path / "b.sqlite3"
    TokenBudget(path, daily_limit=1000).spend(400)
    assert TokenBudget(path, daily_limit=1000).remaining() == 600


def test_remaining_can_go_negative(tmp_path):
    budget = TokenBudget(tmp_path / "b.sqlite3", daily_limit=100)
    budget.spend(250)
    assert budget.remaining() == -150
```

- [ ] **Step 2: Tests laufen lassen — müssen fehlschlagen**

Run: `.venv/Scripts/python -m pytest tests/test_guards.py -v`
Expected: FAIL mit `ModuleNotFoundError`.

- [ ] **Step 3: Implementieren**

`app/guards.py`:

```python
"""Tages-Token-Budget in SQLite — Deckel für die öffentliche Demo."""
from __future__ import annotations

import sqlite3
from datetime import date
from pathlib import Path


class TokenBudget:
    def __init__(self, db_path: Path, daily_limit: int):
        self.daily_limit = daily_limit
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self.conn.execute(
            "CREATE TABLE IF NOT EXISTS budget (day TEXT PRIMARY KEY, tokens INTEGER NOT NULL)"
        )
        self.conn.commit()

    def spend(self, tokens: int) -> None:
        self.conn.execute(
            "INSERT INTO budget (day, tokens) VALUES (?, ?) "
            "ON CONFLICT(day) DO UPDATE SET tokens = tokens + excluded.tokens",
            (date.today().isoformat(), tokens),
        )
        self.conn.commit()

    def used_today(self) -> int:
        row = self.conn.execute(
            "SELECT tokens FROM budget WHERE day = ?", (date.today().isoformat(),)
        ).fetchone()
        return row[0] if row else 0

    def remaining(self) -> int:
        return self.daily_limit - self.used_today()
```

- [ ] **Step 4: Tests laufen lassen — müssen bestehen**

Run: `.venv/Scripts/python -m pytest tests/test_guards.py -v`
Expected: 4 PASS.

- [ ] **Step 5: Commit**

```bash
git add app/guards.py tests/test_guards.py
git commit -m "feat: Tages-Token-Budget mit SQLite-Persistenz"
```

---

### Task 8: FastAPI-Service mit SSE

**Files:**
- Create: `app/main.py`
- Test: `tests/test_api.py`

**Interfaces:**
- Consumes: `Retriever`, `RetrievedDoc` (Task 5); `get_client`, `rewrite_query`, `stream_answer` (Task 6); `TokenBudget` (Task 7); `settings` (Task 4).
- Produces:
  - `app.main.create_app(retriever=None, budget=None, answer_fn=None, rewrite_fn=None, llm_client=None) -> FastAPI` und Modul-Attribut `app.main.app` (nur außerhalb von pytest gebaut)
  - `POST /api/chat` — Body `{"messages": [{"role": "user"|"assistant", "content": str}]}`; Antwort `text/event-stream` mit Zeilen `data: {json}\n\n`; Event-Typen: `retrieval` (`{"type","docs":[{"id","title","score"}]}`), `token` (`{"type","text"}`), `sources` (`{"type","items":[{"n","title","url"}]}`), `error` (`{"type","message"}`), `done` (`{"type"}`)
  - `GET /api/health` — `{"status": "ok"}`
  - `/` liefert `static/index.html` (sobald Task 9 die Dateien anlegt)
  - Rate-Limit: `10/minute;50/day` je IP via slowapi; Budget erschöpft → HTTP 429 mit `detail` „Demo-Budget für heute erschöpft"
  - Konstanten: `MAX_QUESTION_CHARS = 500`, `MAX_HISTORY_MESSAGES = 20`, `HISTORY_TURNS_FOR_LLM = 6`

- [ ] **Step 1: Failing Tests schreiben**

`tests/test_api.py`:

```python
import json

import pytest
from fastapi.testclient import TestClient

from app.guards import TokenBudget
from app.main import create_app
from app.retrieval import RetrievedDoc

DOC = RetrievedDoc(id="faq-0001", type="faq", title="Wie funktioniert der Käuferschutz?",
                   url="https://www.chrono24.de/info/faqs.htm",
                   text="Der Käuferschutz sichert deine Zahlung ab.", score=0.05)


class FakeRetriever:
    def __init__(self, docs):
        self.docs = docs

    def retrieve(self, query, top_k=5):
        return self.docs


async def fake_answer_fn(question, docs, history, client):
    yield {"type": "token", "text": "Der Käuferschutz "}
    yield {"type": "token", "text": "sichert deine Zahlung ab. [1]"}
    yield {"type": "usage", "input_tokens": 100, "output_tokens": 20}


async def fake_rewrite_fn(history, question, client):
    return question


def make_client(tmp_path, docs, budget=None):
    app = create_app(retriever=FakeRetriever(docs),
                     budget=budget or TokenBudget(tmp_path / "b.sqlite3", daily_limit=1000),
                     answer_fn=fake_answer_fn, rewrite_fn=fake_rewrite_fn,
                     llm_client=object())
    return TestClient(app)


def parse_sse(text):
    return [json.loads(line.removeprefix("data: "))
            for line in text.splitlines() if line.startswith("data: ")]


def test_health(tmp_path):
    response = make_client(tmp_path, [DOC]).get("/api/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_chat_streams_tokens_sources_done(tmp_path):
    response = make_client(tmp_path, [DOC]).post("/api/chat", json={"messages": [
        {"role": "user", "content": "Wie funktioniert der Käuferschutz?"}]})
    assert response.status_code == 200
    events = parse_sse(response.text)
    types = [e["type"] for e in events]
    assert types == ["retrieval", "token", "token", "sources", "done"]
    sources = next(e for e in events if e["type"] == "sources")
    assert sources["items"][0]["url"] == "https://www.chrono24.de/info/faqs.htm"


def test_chat_offtopic_returns_notfound_message(tmp_path):
    response = make_client(tmp_path, []).post("/api/chat", json={"messages": [
        {"role": "user", "content": "Gedicht über Katzen"}]})
    events = parse_sse(response.text)
    assert any("finde ich nichts" in e.get("text", "") for e in events)
    assert events[-1]["type"] == "done"


def test_chat_spends_budget(tmp_path):
    budget = TokenBudget(tmp_path / "b3.sqlite3", daily_limit=1000)
    make_client(tmp_path, [DOC], budget=budget).post("/api/chat", json={"messages": [
        {"role": "user", "content": "Käuferschutz?"}]})
    assert budget.used_today() == 120


def test_chat_rejects_when_budget_empty(tmp_path):
    budget = TokenBudget(tmp_path / "b4.sqlite3", daily_limit=10)
    budget.spend(10)
    response = make_client(tmp_path, [DOC], budget=budget).post("/api/chat", json={
        "messages": [{"role": "user", "content": "Käuferschutz?"}]})
    assert response.status_code == 429
    assert "Demo-Budget" in response.json()["detail"]


def test_chat_validates_input(tmp_path):
    client = make_client(tmp_path, [DOC])
    assert client.post("/api/chat", json={"messages": []}).status_code == 422
    assert client.post("/api/chat", json={"messages": [
        {"role": "user", "content": "x" * 501}]}).status_code == 422
    too_many = [{"role": "user", "content": "hi"}] * 21
    assert client.post("/api/chat", json={"messages": too_many}).status_code == 422
    assert client.post("/api/chat", json={"messages": [
        {"role": "assistant", "content": "hi"}]}).status_code == 422
```

- [ ] **Step 2: Tests laufen lassen — müssen fehlschlagen**

Run: `.venv/Scripts/python -m pytest tests/test_api.py -v`
Expected: FAIL mit `ModuleNotFoundError: No module named 'app.main'`.

- [ ] **Step 3: Implementieren**

`app/main.py`:

```python
"""FastAPI-Service: Chat-Endpoint mit SSE, Healthcheck, statisches Frontend."""
from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, field_validator
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

from app import llm
from app.config import settings
from app.guards import TokenBudget
from app.retrieval import Retriever

logger = logging.getLogger("chrono24-chatbot")

MAX_QUESTION_CHARS = 500
MAX_HISTORY_MESSAGES = 20
HISTORY_TURNS_FOR_LLM = 6
NOT_FOUND_ANSWER = "Dazu finde ich nichts in den Chrono24-Hilfeseiten."


class ChatMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(min_length=1, max_length=MAX_QUESTION_CHARS)


class ChatRequest(BaseModel):
    messages: list[ChatMessage] = Field(min_length=1, max_length=MAX_HISTORY_MESSAGES)

    @field_validator("messages")
    @classmethod
    def last_message_must_be_user(cls, v: list[ChatMessage]) -> list[ChatMessage]:
        if v[-1].role != "user":
            raise ValueError("letzte Nachricht muss vom Nutzer sein")
        return v


def sse(payload: dict) -> str:
    return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"


def create_app(retriever=None, budget=None, answer_fn=None, rewrite_fn=None,
               llm_client=None) -> FastAPI:
    app = FastAPI(title="Chrono24-FAQ-Chatbot")
    limiter = Limiter(key_func=get_remote_address)
    app.state.limiter = limiter
    app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

    # Fail fast: ohne Index wirft Retriever() beim Start (statt leerer Antworten).
    app.state.retriever = retriever or Retriever(settings.index_dir, settings.corpus_path)
    app.state.budget = budget or TokenBudget(settings.budget_db, settings.daily_token_budget)
    app.state.answer_fn = answer_fn or llm.stream_answer
    app.state.rewrite_fn = rewrite_fn or llm.rewrite_query
    app.state.llm_client = llm_client

    @app.get("/api/health")
    async def health() -> dict:
        return {"status": "ok"}

    @app.post("/api/chat")
    @limiter.limit("10/minute;50/day")
    async def chat(request: Request, body: ChatRequest) -> StreamingResponse:
        if app.state.budget.remaining() <= 0:
            raise HTTPException(status_code=429, detail="Demo-Budget für heute erschöpft")

        history = [m.model_dump() for m in body.messages[:-1]][-HISTORY_TURNS_FOR_LLM * 2:]
        question = body.messages[-1].content

        async def event_stream():
            try:
                client = app.state.llm_client or llm.get_client()
                standalone = await app.state.rewrite_fn(history, question, client)
                docs = app.state.retriever.retrieve(standalone)
                yield sse({"type": "retrieval",
                           "docs": [{"id": d.id, "title": d.title, "score": d.score}
                                    for d in docs]})
                if not docs:
                    yield sse({"type": "token", "text": NOT_FOUND_ANSWER})
                    yield sse({"type": "done"})
                    return
                async for event in app.state.answer_fn(standalone, docs, history, client):
                    if event["type"] == "usage":
                        app.state.budget.spend(event["input_tokens"] + event["output_tokens"])
                    else:
                        yield sse(event)
                yield sse({"type": "sources",
                           "items": [{"n": i, "title": d.title, "url": d.url}
                                     for i, d in enumerate(docs, 1)]})
                yield sse({"type": "done"})
            except Exception:
                logger.exception("Chat-Anfrage fehlgeschlagen")
                yield sse({"type": "error",
                           "message": "Antwort gerade nicht möglich, versuch es gleich nochmal."})

        return StreamingResponse(event_stream(), media_type="text/event-stream")

    static_dir = Path("static")
    if static_dir.is_dir():
        app.mount("/", StaticFiles(directory=static_dir, html=True), name="static")

    return app


# Beim Import durch uvicorn (Deployment) App sofort bauen; Tests nutzen create_app()
# direkt — der pytest-Guard verhindert, dass beim Test-Import der echte Index lädt.
app = create_app() if "pytest" not in sys.modules else None
```

- [ ] **Step 4: Tests laufen lassen — müssen bestehen**

Run: `.venv/Scripts/python -m pytest tests/test_api.py -v`
Expected: 6 PASS.

- [ ] **Step 5: Ganze Suite + Lint**

Run: `.venv/Scripts/python -m pytest tests/ -v && .venv/Scripts/ruff check .`
Expected: alles PASS, keine Lint-Fehler.

- [ ] **Step 6: Commit**

```bash
git add app/main.py tests/test_api.py
git commit -m "feat: FastAPI-Chat-Endpoint mit SSE, Rate-Limit und Budget-Guard"
```

---

### Task 9: Frontend (Vanilla JS)

**Files:**
- Create: `static/index.html`, `static/style.css`, `static/app.js`

**Interfaces:**
- Consumes: `POST /api/chat` SSE-Protokoll aus Task 8 (Event-Typen `retrieval`, `token`, `sources`, `error`, `done`).
- Produces: Chat-UI unter `/`.

- [ ] **Step 1: index.html schreiben**

```html
<!DOCTYPE html>
<html lang="de">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Chrono24-FAQ-Chatbot — Portfolio-Demo</title>
  <link rel="stylesheet" href="style.css">
</head>
<body>
  <main class="chat">
    <header>
      <h1>Chrono24-FAQ-Chatbot</h1>
      <p class="subtitle">Stell eine Frage zu Kauf, Verkauf, Käuferschutz oder Versand.</p>
    </header>
    <div id="messages" class="messages"></div>
    <div id="examples" class="examples">
      <button class="example">Wie funktioniert der Käuferschutz?</button>
      <button class="example">Was kostet der Verkauf einer Uhr?</button>
      <button class="example">Wie läuft der Versand ab?</button>
      <button class="example">Was ist der Trusted Checkout?</button>
    </div>
    <form id="form" class="input-row">
      <input id="input" type="text" maxlength="500" autocomplete="off"
             placeholder="Deine Frage …" required>
      <button type="submit" id="send">Senden</button>
    </form>
    <footer>
      Inoffizielles Portfolio-Projekt, nicht mit Chrono24 verbunden. Antworten ohne
      Gewähr. Quelle: öffentliche Hilfeseiten von chrono24.de.
    </footer>
  </main>
  <script src="app.js"></script>
</body>
</html>
```

- [ ] **Step 2: app.js schreiben**

```javascript
const messagesEl = document.getElementById("messages");
const form = document.getElementById("form");
const input = document.getElementById("input");
const sendBtn = document.getElementById("send");
const examplesEl = document.getElementById("examples");

const history = [];

function addMessage(role, text = "") {
  const div = document.createElement("div");
  div.className = `msg ${role}`;
  div.textContent = text;
  messagesEl.appendChild(div);
  messagesEl.scrollTop = messagesEl.scrollHeight;
  return div;
}

function addSources(items) {
  const div = document.createElement("div");
  div.className = "sources";
  const links = items.map((s) => {
    const a = document.createElement("a");
    a.href = s.url;
    a.target = "_blank";
    a.rel = "noopener";
    a.textContent = `[${s.n}] ${s.title}`;
    return a;
  });
  div.append("Quellen: ", ...links);
  messagesEl.appendChild(div);
}

function addRetrievalDetails(docs) {
  if (!docs.length) return;
  const details = document.createElement("details");
  details.className = "retrieval";
  const summary = document.createElement("summary");
  summary.textContent = `Retrieval-Details (${docs.length} Treffer)`;
  const list = document.createElement("ul");
  for (const d of docs) {
    const li = document.createElement("li");
    const code = document.createElement("code");
    code.textContent = d.score.toFixed(3);
    li.append(code, ` ${d.title}`);
    list.appendChild(li);
  }
  details.append(summary, list);
  messagesEl.appendChild(details);
}

async function ask(question) {
  input.value = "";
  sendBtn.disabled = true;
  examplesEl.style.display = "none";
  addMessage("user", question);
  history.push({ role: "user", content: question });
  const botEl = addMessage("bot", "…");
  let answer = "";

  try {
    const response = await fetch("/api/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ messages: history.slice(-20) }),
    });
    if (response.status === 429) {
      const body = await response.json().catch(() => ({}));
      botEl.textContent = body.detail || "Zu viele Anfragen — bitte später erneut versuchen.";
      return;
    }
    if (!response.ok) throw new Error(`HTTP ${response.status}`);

    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    let buffer = "";
    for (;;) {
      const { done, value } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });
      const chunks = buffer.split("\n\n");
      buffer = chunks.pop();
      for (const chunk of chunks) {
        if (!chunk.startsWith("data: ")) continue;
        const event = JSON.parse(chunk.slice(6));
        if (event.type === "token") {
          answer += event.text;
          botEl.textContent = answer;
        } else if (event.type === "retrieval") {
          addRetrievalDetails(event.docs);
        } else if (event.type === "sources") {
          addSources(event.items);
        } else if (event.type === "error") {
          botEl.textContent = event.message;
        }
        messagesEl.scrollTop = messagesEl.scrollHeight;
      }
    }
    if (answer) history.push({ role: "assistant", content: answer });
  } catch {
    botEl.textContent = "Verbindungsfehler — bitte gleich nochmal versuchen.";
  } finally {
    sendBtn.disabled = false;
    input.focus();
  }
}

form.addEventListener("submit", (e) => {
  e.preventDefault();
  const question = input.value.trim();
  if (question) ask(question);
});

examplesEl.addEventListener("click", (e) => {
  if (e.target.classList.contains("example")) ask(e.target.textContent);
});
```

(Quellen/Retrieval-Details werden bewusst per `createElement`/`textContent` gebaut, nicht per `innerHTML` — kein XSS über Dokumenttitel.)

- [ ] **Step 3: style.css schreiben**

```css
:root {
  --bg: #0d1117;
  --panel: #161b22;
  --text: #e6edf3;
  --muted: #8b949e;
  --accent: #2f81f7;
  --user: #1f6feb33;
}

* { box-sizing: border-box; margin: 0; }

body {
  background: var(--bg);
  color: var(--text);
  font-family: system-ui, -apple-system, "Segoe UI", sans-serif;
  display: flex;
  justify-content: center;
  min-height: 100vh;
}

.chat {
  width: min(720px, 100%);
  display: flex;
  flex-direction: column;
  padding: 1rem;
  gap: 0.75rem;
}

header h1 { font-size: 1.3rem; }
.subtitle, footer { color: var(--muted); font-size: 0.85rem; }

.messages {
  flex: 1;
  overflow-y: auto;
  display: flex;
  flex-direction: column;
  gap: 0.5rem;
  min-height: 40vh;
}

.msg { padding: 0.6rem 0.9rem; border-radius: 10px; white-space: pre-wrap; max-width: 90%; }
.msg.user { background: var(--user); align-self: flex-end; }
.msg.bot { background: var(--panel); align-self: flex-start; }

.sources { font-size: 0.85rem; color: var(--muted); }
.sources a { color: var(--accent); margin-right: 0.5rem; }

.retrieval { font-size: 0.8rem; color: var(--muted); background: var(--panel);
  border-radius: 8px; padding: 0.4rem 0.7rem; }
.retrieval code { color: var(--accent); }
.retrieval ul { margin: 0.3rem 0 0 1.2rem; }

.examples { display: flex; flex-wrap: wrap; gap: 0.5rem; }
.example { background: var(--panel); color: var(--text); border: 1px solid var(--muted);
  border-radius: 999px; padding: 0.4rem 0.9rem; cursor: pointer; font-size: 0.85rem; }
.example:hover { border-color: var(--accent); }

.input-row { display: flex; gap: 0.5rem; }
#input { flex: 1; background: var(--panel); color: var(--text); border: 1px solid var(--muted);
  border-radius: 8px; padding: 0.6rem 0.9rem; font-size: 1rem; }
#send { background: var(--accent); color: #fff; border: 0; border-radius: 8px;
  padding: 0.6rem 1.2rem; cursor: pointer; }
#send:disabled { opacity: 0.5; }
```

- [ ] **Step 4: Manuell testen (mit echtem API-Key)**

`.env` mit echtem `ANTHROPIC_API_KEY` anlegen (nicht committen!).
Run: `.venv/Scripts/uvicorn app.main:app --port 8000`
Im Browser `http://localhost:8000` öffnen. Prüfen: Beispielfrage klicken → Antwort streamt, Quellen-Links erscheinen, Retrieval-Details aufklappbar, Off-Topic-Frage („Backrezept für Pizza") liefert „finde ich nichts"-Antwort, Folgefrage („und beim Verkauf?") wird sinnvoll beantwortet.

- [ ] **Step 5: Commit**

```bash
git add static/
git commit -m "feat: Chat-Frontend mit SSE-Streaming, Quellen und Retrieval-Details"
```

---

### Task 10: Retrieval-Eval

**Files:**
- Create: `eval/__init__.py`, `eval/questions.json`, `eval/run_eval.py`
- Test: `tests/test_eval.py`

**Interfaces:**
- Consumes: `Retriever`, `RetrievedDoc` (Task 5), echter Index (Task 4), `data/corpus.json`.
- Produces:
  - `eval/questions.json` — Liste `[{"question": str, "expected_doc_id": str}, ...]`, ~30 Einträge
  - `eval.run_eval.hit_rate_at_k(retriever, questions: list[dict], k: int = 5) -> tuple[float, list[dict]]` — (Rate, Fehlliste mit zusätzlichem Key `"got": list[str]`)
  - CLI: `python -m eval.run_eval` druckt Hit-Rate und Misses

- [ ] **Step 1: Failing Test schreiben**

`eval/__init__.py`: leer. `tests/test_eval.py`:

```python
from app.retrieval import RetrievedDoc
from eval.run_eval import hit_rate_at_k


class StubRetriever:
    def __init__(self, mapping):
        self.mapping = mapping

    def retrieve(self, query, top_k=5):
        return [RetrievedDoc(id=i, type="faq", title="t", url="u", text="x", score=0.1)
                for i in self.mapping.get(query, [])]


def test_hit_rate_counts_hits_in_top_k():
    retriever = StubRetriever({
        "F1": ["faq-0001", "faq-0002"],
        "F2": ["faq-0009"],
    })
    questions = [{"question": "F1", "expected_doc_id": "faq-0002"},
                 {"question": "F2", "expected_doc_id": "faq-0003"}]
    rate, misses = hit_rate_at_k(retriever, questions, k=5)
    assert rate == 0.5
    assert misses[0]["question"] == "F2"
    assert misses[0]["got"] == ["faq-0009"]
```

- [ ] **Step 2: Test laufen lassen — muss fehlschlagen**

Run: `.venv/Scripts/python -m pytest tests/test_eval.py -v`
Expected: FAIL mit `ModuleNotFoundError`.

- [ ] **Step 3: Implementieren**

`eval/run_eval.py`:

```python
"""Misst Hit-Rate@k des Retrievals gegen handgeschriebene Testfragen."""
from __future__ import annotations

import json
from pathlib import Path

QUESTIONS_PATH = Path("eval/questions.json")


def hit_rate_at_k(retriever, questions: list[dict], k: int = 5) -> tuple[float, list[dict]]:
    misses = []
    hits = 0
    for item in questions:
        ids = [d.id for d in retriever.retrieve(item["question"], top_k=k)]
        if item["expected_doc_id"] in ids:
            hits += 1
        else:
            misses.append({**item, "got": ids})
    return hits / len(questions), misses


if __name__ == "__main__":
    from app.config import settings
    from app.retrieval import Retriever

    questions = json.loads(QUESTIONS_PATH.read_text(encoding="utf-8"))
    retriever = Retriever(settings.index_dir, settings.corpus_path)
    rate, misses = hit_rate_at_k(retriever, questions)
    print(f"Hit-Rate@5: {rate:.0%} ({len(questions) - len(misses)}/{len(questions)})")
    for miss in misses:
        print(f"  MISS: {miss['question']!r} erwartet {miss['expected_doc_id']}, bekam {miss['got']}")
```

- [ ] **Step 4: Test laufen lassen — muss bestehen**

Run: `.venv/Scripts/python -m pytest tests/test_eval.py -v`
Expected: PASS.

- [ ] **Step 5: Eval-Fragen schreiben**

`eval/questions.json`: ~30 Einträge von Hand. Vorgehen: `data/corpus.json` durchgehen, zu ~30 verschiedenen Dokumenten je eine natürlich formulierte Nutzerfrage schreiben, die NICHT wortgleich mit der FAQ-Frage ist (Paraphrasen, umgangssprachlich, 2–3 englisch). Format:

```json
[
  {"question": "Ist mein Geld sicher, wenn ich dort eine Uhr kaufe?", "expected_doc_id": "faq-0003"},
  {"question": "How much does selling a watch cost?", "expected_doc_id": "faq-0012"}
]
```

(Die echten `expected_doc_id`-Werte aus `data/corpus.json` ablesen.)

- [ ] **Step 6: Eval laufen lassen, Schwellen justieren**

Run: `.venv/Scripts/python -m eval.run_eval`
Expected: Hit-Rate ≥ 80 %. Wenn niedriger: Misses ansehen; typische Ursachen: Konfidenz-Gate zu aggressiv (`SIM_THRESHOLD`/`BM25_THRESHOLD` in `app/retrieval.py` senken), falsche `expected_doc_id`, oder Frage passt auf mehrere Dokumente (dann `expected_doc_id` auf das tatsächlich beste ändern). Endgültige Rate notieren — kommt in Task 11 ins README.

- [ ] **Step 7: Commit**

```bash
git add eval/ tests/test_eval.py
git commit -m "feat: Retrieval-Eval mit Hit-Rate@5 auf handgeschriebenem Fragenset"
```

---

### Task 11: Docker, CI, README, Deploy

**Files:**
- Create: `Dockerfile`, `.dockerignore`, `.github/workflows/ci.yml`, `README.md`

**Interfaces:**
- Consumes: alles Vorherige.
- Produces: baubarer Container, grüne CI, dokumentiertes Repo, Live-Demo.

- [ ] **Step 1: Dockerfile schreiben**

```dockerfile
FROM python:3.12-slim

WORKDIR /srv

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Embedding-Modell beim Build cachen, damit der Start schnell ist
RUN python -c "from sentence_transformers import SentenceTransformer; \
    SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')"

COPY app/ app/
COPY static/ static/
COPY data/corpus.json data/corpus.json
COPY data/index/ data/index/

EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

`.dockerignore`:

```
.venv/
.git/
data/raw/
tests/
eval/
docs/
__pycache__/
```

- [ ] **Step 2: Container lokal bauen und Rauchtest**

Run: `docker build -t chrono24-chatbot .` dann `docker run --rm -p 8000:8000 -e ANTHROPIC_API_KEY=<key> chrono24-chatbot`
Expected: Build ok; `http://localhost:8000/api/health` liefert `{"status":"ok"}`; eine Chat-Frage im Browser funktioniert.

- [ ] **Step 3: CI-Workflow schreiben**

`.github/workflows/ci.yml`:

```yaml
name: CI
on:
  push:
    branches: [main]
  pull_request:

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.12"
          cache: pip
      - run: pip install -r requirements-dev.txt
      - run: ruff check .
      - run: pytest tests/ -v

  docker:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - run: docker build -t chrono24-chatbot .
```

- [ ] **Step 4: README schreiben**

`README.md` mit diesen Abschnitten (Inhalte aus Spec und den echten Ergebnissen dieses Repos, keine erfundenen Zahlen):
- **Was ist das:** 2 Sätze + Demo-Link (nach Step 7 nachtragen).
- **Disclaimer:** inoffizielles Portfolio-Projekt, nicht mit Chrono24 verbunden.
- **Architektur:** das Zwei-Läufe-Diagramm aus der Spec (Offline-Pipeline / Online-Service) als Codeblock.
- **Warum Hybrid-RAG:** 3 Sätze (Q&A-Struktur erhalten, Frage-auf-Frage-Embedding, BM25 + RRF) + echte Hit-Rate@5 aus Task 10.
- **Scraping-Ethik:** robots.txt-Befund aus Task 1, 1 Request/Sekunde, einmaliger lokaler Lauf, der Server scrapt nie.
- **Lokal starten:** venv, `pip install -r requirements-dev.txt`, `.env` anlegen, `uvicorn app.main:app`.
- **Pipeline neu bauen:** `python -m pipeline.scrape && python -m pipeline.parse && python -m pipeline.index`.
- **Tests:** `pytest tests/`, Eval: `python -m eval.run_eval`.
- **Guards:** Rate-Limit 10/min und 50/Tag pro IP, Tagesbudget 200k Tokens.

- [ ] **Step 5: Ganze Suite + Lint final**

Run: `.venv/Scripts/python -m pytest tests/ -v && .venv/Scripts/ruff check .`
Expected: alles grün.

- [ ] **Step 6: Commit**

```bash
git add Dockerfile .dockerignore .github/ README.md
git commit -m "ci: Docker-Build, GitHub-Actions-CI und README"
```

- [ ] **Step 7: Deploy auf Render (manuell, mit Marco)**

GitHub-Repo anlegen und pushen; auf render.com: „New Web Service" → Repo verbinden → Runtime Docker → Env-Var `ANTHROPIC_API_KEY` setzen → Health Check Path `/api/health`. Nach Deploy: Demo-Link testen, in README eintragen, Screenshot einfügen, committen. Falls der Free-Tier-RAM (512 MB) beim Start platzt (OOM im Log): Spec-Fallback aktivieren (Embedding-API statt lokalem Modell) — das ist ein neuer, eigener Task und braucht Rücksprache mit Marco.
