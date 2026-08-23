# Query-Varianten + CI Eval Gate Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** LLM-generierte Umformulierungen jeder FAQ-Frage fließen als zusätzliche Embedding-Einträge in den Vektorindex ein (Question-to-Question-Matching robuster gegen abweichende Nutzerformulierungen), und zwei CI-Jobs verhindern künftig stille Qualitätsregression im Retrieval (Hit-Rate, jeder Push/PR) und in der Antworttreue (Faithfulness, nur bei Push auf main).

**Architecture:** Neues Offline-Pipeline-Modul `pipeline/variants.py` generiert pro FAQ 3–5 Umformulierungen per Haiku und cached sie in `data/variants.json` (einmaliger API-Kosten-Lauf, wie der Scraper nie zur Laufzeit). `pipeline/index.py` embedded jede Variante zusätzlich zum Original und verknüpft sie per Chroma-Metadatum `canonical_id` mit dem Original-FAQ-Dokument; BM25 bleibt unverändert (Varianten adressieren gezielt die Schwäche des Embedding-Pfads, nicht die des Lexikal-Pfads, der über die Synonym-Expansion bereits abgesichert ist). `app/retrieval.py` löst Varianten-Treffer auf ihre kanonische ID auf und entfernt Duplikate vor der RRF-Fusion. `eval/run_eval.py` bekommt einen `--gate`-Modus (Exit-Code bei Hit-Rate-Regression), `eval/judge.py` einen `--gate`-Modus (Exit-Code bei Faithfulness-Regression); beide werden in `.github/workflows/ci.yml` als eigene Jobs verdrahtet.

**Tech Stack:** Python 3.12, chromadb, rank_bm25, sentence-transformers, Anthropic AsyncClient (Haiku via `settings.model`), pytest (`asyncio_mode = "auto"`), GitHub Actions.

**Spec:** Kein separates Spec-Dokument. Anforderungen entstanden im Chat vom 2026-08-23 (User-Vorschlag: Frage-Embedding, Query-Varianten, Contextual Chunking, Metadaten — Codebase-Check ergab: Frage-Embedding und Heading-Kontext sind bereits umgesetzt, siehe `pipeline/index.py:16-19` und `pipeline/parse.py`; verbleibender Hebel ist Query-Varianten, kombiniert mit dem vom User zuvor gewünschten CI Eval Gate). User-Entscheidung per Rückfrage: Plan zuerst, TDD-Umsetzung danach; Faithfulness-Gate nur bei Push auf main (API-Kosten sparen).

## Global Constraints

- Testlauf immer mit `.venv\Scripts\python.exe -m pytest …` (globales Python hat keine Abhängigkeiten).
- Deutsch für alle Kommentare, Prompts und README-Texte.
- Keine stillen Fallbacks: Fehler pro Einzelfrage/-FAQ dürfen den Gesamtlauf nicht stoppen (Muster aus `eval/judge.py::_run_all` — `except Exception: logger.exception(...)`), aber ein tatsächlicher Gate-Verstoß muss den Prozess mit Exit-Code ≠ 0 beenden.
- `data/index/`-Änderungen (Chroma mutiert beim bloßen Öffnen) vor jedem `git status`-Check prüfen; nur committen, wenn sie aus einem bewussten Reindex-Lauf stammen (Task 5), sonst `git restore data/index/`.
- **Task 5 verursacht echte API-Kosten** (~100+ Haiku-Calls für die Varianten-Generierung). Vor Ausführung von Task 5 explizit beim User nachfragen, nicht automatisch loslaufen lassen.
- Commits: Conventional Commits, Body deutsch, Co-Authored-By-Zeile wie in den letzten Commits.
- Ruff-Zeilenlänge 100 (`pyproject.toml`).

---

### Task 1: Hit-Rate-Gate in `eval/run_eval.py` + CI-Job

**Files:**
- Modify: `eval/run_eval.py`
- Modify: `.github/workflows/ci.yml`
- Test: `tests/test_run_eval.py` (neu)

**Interfaces:**
- Produces: `check_gate(tuning_rate: float, holdout_rate: float) -> list[str]` (leere Liste = Gate bestanden, sonst eine Verstoßmeldung pro verletzter Schwelle).
- Konsumiert nur vorhandenes `hit_rate_at_k` (unverändert).

- [ ] **Step 1: Failing Test schreiben** — `tests/test_run_eval.py` neu anlegen:

```python
from eval.run_eval import HOLDOUT_MIN_HIT_RATE, TUNING_MIN_HIT_RATE, check_gate


def test_check_gate_passes_when_both_rates_meet_minimum():
    assert check_gate(TUNING_MIN_HIT_RATE, HOLDOUT_MIN_HIT_RATE) == []


def test_check_gate_fails_on_tuning_regression():
    failures = check_gate(TUNING_MIN_HIT_RATE - 0.01, HOLDOUT_MIN_HIT_RATE)
    assert len(failures) == 1
    assert "Tuning" in failures[0]


def test_check_gate_fails_on_holdout_regression():
    failures = check_gate(TUNING_MIN_HIT_RATE, HOLDOUT_MIN_HIT_RATE - 0.01)
    assert len(failures) == 1
    assert "Holdout" in failures[0]


def test_check_gate_reports_both_failures_independently():
    failures = check_gate(0.0, 0.0)
    assert len(failures) == 2
```

- [ ] **Step 2: Fail sehen**

Run: `.venv\Scripts\python.exe -m pytest tests/test_run_eval.py -v`
Expected: ImportError `check_gate` (Modul existiert, Funktion noch nicht).

- [ ] **Step 3: Implementieren** — in `eval/run_eval.py` nach der `QUESTIONS_PATH`-Zeile (Zeile 7) einfügen:

```python
HOLDOUT_QUESTIONS_PATH = Path("eval/questions_holdout.json")

# Aktuell gemessen: Tuning 91 % (30/33), Holdout 87 % (13/15) — Schwellen bewusst
# mit Puffer für Einzelfrage-Rauschen (33/15 Fragen sind ein kleines Sample) und
# nach jeder bewussten Verbesserung hier nachziehen, nicht nur nach oben schieben.
TUNING_MIN_HIT_RATE = 0.85
HOLDOUT_MIN_HIT_RATE = 0.80
```

Am Ende der Datei, vor `if __name__ == "__main__":`, einfügen:

```python
def check_gate(tuning_rate: float, holdout_rate: float) -> list[str]:
    """Prüft beide Hit-Raten gegen ihre Mindestschwelle. Leer = Gate bestanden."""
    failures = []
    if tuning_rate < TUNING_MIN_HIT_RATE:
        failures.append(
            f"Tuning-Hit-Rate {tuning_rate:.0%} unter Minimum {TUNING_MIN_HIT_RATE:.0%}"
        )
    if holdout_rate < HOLDOUT_MIN_HIT_RATE:
        failures.append(
            f"Holdout-Hit-Rate {holdout_rate:.0%} unter Minimum {HOLDOUT_MIN_HIT_RATE:.0%}"
        )
    return failures
```

`if __name__ == "__main__":`-Block ersetzen durch:

```python
if __name__ == "__main__":
    import sys

    from app.config import settings
    from app.retrieval import Retriever

    retriever = Retriever(settings.index_dir, settings.corpus_path)

    if "--gate" in sys.argv:
        tuning = json.loads(QUESTIONS_PATH.read_text(encoding="utf-8"))
        holdout = json.loads(HOLDOUT_QUESTIONS_PATH.read_text(encoding="utf-8"))
        tuning_rate, _ = hit_rate_at_k(retriever, tuning)
        holdout_rate, _ = hit_rate_at_k(retriever, holdout)
        print(f"Tuning-Hit-Rate@5: {tuning_rate:.0%}")
        print(f"Holdout-Hit-Rate@5: {holdout_rate:.0%}")
        failures = check_gate(tuning_rate, holdout_rate)
        for failure in failures:
            print(f"GATE FAIL: {failure}")
        sys.exit(1 if failures else 0)

    questions = json.loads(_questions_path(sys.argv).read_text(encoding="utf-8"))
    if "--with-rewrite" in sys.argv:
        questions = _rewrite_questions(questions)
    rate, misses = hit_rate_at_k(retriever, questions)
    print(f"Hit-Rate@5: {rate:.0%} ({len(questions) - len(misses)}/{len(questions)})")
    for miss in misses:
        print(f"  MISS: {miss['question']!r} erwartet {miss['expected_doc_id']}, bekam {miss['got']}")
```

- [ ] **Step 4: Pass sehen**

Run: `.venv\Scripts\python.exe -m pytest tests/test_run_eval.py -v`
Expected: 4 PASSED.

- [ ] **Step 5: Gate lokal gegen den echten Index prüfen**

Run: `.venv\Scripts\python.exe -m eval.run_eval --gate`
Expected: Exit-Code 0, Ausgabe zeigt Tuning- und Holdout-Hit-Rate über den Schwellen (aktueller Stand 91 %/87 %).

- [ ] **Step 6: CI-Job ergänzen** — in `.github/workflows/ci.yml` nach dem `docker`-Job anhängen:

```yaml
  eval-gate:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.12"
          cache: pip
      - run: pip install -r requirements-dev.txt
      - run: python -m eval.run_eval --gate
```

- [ ] **Step 7: Commit**

```bash
git add eval/run_eval.py tests/test_run_eval.py .github/workflows/ci.yml
git commit -m "$(cat <<'EOF'
feat: Hit-Rate-Gate fuer Retrieval-Regression in CI

Neuer --gate-Modus in eval/run_eval.py prueft Tuning- und Holdout-Hit-Rate
gegen Mindestschwellen (Exit-Code 1 bei Unterschreitung), neuer CI-Job
eval-gate laesst ihn bei jedem Push/PR laufen -- kein API-Call noetig, rein
lokales Retrieval gegen den committeten Index.

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
EOF
)"
```

---

### Task 2: Query-Varianten-Generator (`pipeline/variants.py`)

**Files:**
- Create: `pipeline/variants.py`
- Test: `tests/test_variants.py`

**Interfaces:**
- Produces: `parse_variants(text: str) -> list[str]`, `generate_variants(question: str, client, model: str) -> list[str]` (async), `build_variants(faq_docs: list[dict], client, model: str) -> dict[str, list[str]]` (async, `faq_docs`-Elemente brauchen mindestens `id` und `question`) — Task 3 konsumiert die geschriebene `data/variants.json` im selben Format wie `build_variants` zurückgibt: `{faq_id: [umformulierung, ...]}`.

- [ ] **Step 1: Failing Tests schreiben** — `tests/test_variants.py` neu anlegen:

```python
from pipeline.variants import (
    VARIANTS_SYSTEM,
    build_variants,
    generate_variants,
    parse_variants,
)


def test_parse_variants_reads_json_array():
    text = '["Wie lange dauert der Versand?", "Was kostet der Versand?"]'
    assert parse_variants(text) == ["Wie lange dauert der Versand?", "Was kostet der Versand?"]


def test_parse_variants_strips_code_fence():
    text = '```json\n["Frage A", "Frage B"]\n```'
    assert parse_variants(text) == ["Frage A", "Frage B"]


def test_parse_variants_returns_empty_on_invalid_json():
    assert parse_variants("Das ist kein JSON.") == []


def test_parse_variants_returns_empty_for_non_list_json():
    assert parse_variants('{"question": "x"}') == []


def test_parse_variants_ignores_non_string_and_blank_items():
    text = '["Frage A", "", 42, "  ", "Frage B"]'
    assert parse_variants(text) == ["Frage A", "Frage B"]


class FakeVariantMessages:
    def __init__(self, response_text):
        self.response_text = response_text
        self.calls = []

    async def create(self, **kwargs):
        self.calls.append(kwargs)
        return type("R", (), {
            "content": [type("B", (), {"type": "text", "text": self.response_text})()]
        })()


class FakeVariantClient:
    def __init__(self, response_text):
        self.messages = FakeVariantMessages(response_text)


async def test_generate_variants_calls_llm_with_question_and_parses_result():
    client = FakeVariantClient('["Wie lange dauert der Versand?", "Was kostet der Versand?"]')
    result = await generate_variants("Wie funktioniert der Versand?", client, model="claude-haiku-4-5")
    assert result == ["Wie lange dauert der Versand?", "Was kostet der Versand?"]
    call = client.messages.calls[0]
    assert call["model"] == "claude-haiku-4-5"
    assert call["system"] == VARIANTS_SYSTEM
    assert call["messages"] == [{"role": "user", "content": "Wie funktioniert der Versand?"}]


class FlakyVariantMessages:
    """Wirft beim zweiten Call, um Fehlertoleranz pro FAQ zu pruefen."""

    def __init__(self, responses):
        self.responses = responses
        self.calls = 0

    async def create(self, **kwargs):
        self.calls += 1
        response = self.responses[self.calls - 1]
        if response is None:
            raise RuntimeError("API-Fehler")
        return type("R", (), {"content": [type("B", (), {"type": "text", "text": response})()]})()


class FlakyVariantClient:
    def __init__(self, responses):
        self.messages = FlakyVariantMessages(responses)


async def test_build_variants_skips_empty_results_and_survives_errors():
    faqs = [
        {"id": "faq-0001", "question": "Frage eins"},
        {"id": "faq-0002", "question": "Frage zwei"},
        {"id": "faq-0003", "question": "Frage drei"},
    ]
    client = FlakyVariantClient([
        '["Umformuliert eins"]',
        None,  # faq-0002: API-Fehler
        "[]",  # faq-0003: leere Antwort
    ])
    result = await build_variants(faqs, client, model="claude-haiku-4-5")
    assert result == {"faq-0001": ["Umformuliert eins"]}
```

- [ ] **Step 2: Fail sehen**

Run: `.venv\Scripts\python.exe -m pytest tests/test_variants.py -v`
Expected: ModuleNotFoundError `pipeline.variants`.

- [ ] **Step 3: Implementieren** — `pipeline/variants.py` neu anlegen:

```python
"""Query-Varianten: pro FAQ-Frage 3-5 Umformulierungen generieren (offline, einmalig).

Cached in data/variants.json -- wie der Scraper ein einmaliger, lokaler Lauf,
der Online-Service liest zur Laufzeit nie das LLM fuer diesen Zweck an.
"""
from __future__ import annotations

import json
import logging
import re
from pathlib import Path

logger = logging.getLogger("chrono24-chatbot.variants")

VARIANTS_SYSTEM = (
    "Du erhältst eine FAQ-Frage von Chrono24. Formuliere sie auf 3 bis 5 "
    "verschiedene Arten um, wie eine Nutzerin oder ein Nutzer sie im Chat "
    "stellen könnte -- umgangssprachlicher, mit anderen Wortformen, teils "
    "kürzer. Die Bedeutung darf sich nicht ändern. Antworte NUR mit einem "
    "JSON-Array von Strings, ohne Erklärung."
)

MAX_VARIANT_TOKENS = 300
VARIANTS_PATH = Path("data/variants.json")

_FENCE_RE = re.compile(r"^```(?:json)?\s*\n?(.*?)\n?```$", re.DOTALL)


def parse_variants(text: str) -> list[str]:
    stripped = text.strip()
    match = _FENCE_RE.match(stripped)
    candidate = match.group(1).strip() if match else stripped
    try:
        parsed = json.loads(candidate)
    except json.JSONDecodeError:
        return []
    if not isinstance(parsed, list):
        return []
    return [v.strip() for v in parsed if isinstance(v, str) and v.strip()]


async def generate_variants(question: str, client, model: str) -> list[str]:
    response = await client.messages.create(
        model=model,
        max_tokens=MAX_VARIANT_TOKENS,
        system=VARIANTS_SYSTEM,
        messages=[{"role": "user", "content": question}],
    )
    text = next((b.text for b in response.content if b.type == "text"), "")
    return parse_variants(text)


async def build_variants(faq_docs: list[dict], client, model: str) -> dict[str, list[str]]:
    """Ein LLM-Call pro FAQ; ein Fehler pro Frage stoppt den Gesamtlauf nicht
    (analog eval/judge.py::_run_all). FAQs ohne Varianten fehlen im Ergebnis."""
    result: dict[str, list[str]] = {}
    for doc in faq_docs:
        try:
            variants = await generate_variants(doc["question"], client, model)
        except Exception:
            logger.exception("Varianten-Generierung fehlgeschlagen fuer %s", doc["id"])
            variants = []
        if variants:
            result[doc["id"]] = variants
    return result


def main() -> None:
    import asyncio

    from app.config import settings
    from app.llm import get_client

    corpus = json.loads(settings.corpus_path.read_text(encoding="utf-8"))
    faq_docs = [d for d in corpus["documents"] if d["type"] == "faq"]
    client = get_client()

    variants = asyncio.run(build_variants(faq_docs, client, settings.model))

    VARIANTS_PATH.write_text(json.dumps(variants, ensure_ascii=False, indent=1), encoding="utf-8")
    total = sum(len(v) for v in variants.values())
    print(f"{total} Varianten für {len(variants)}/{len(faq_docs)} FAQs nach {VARIANTS_PATH} geschrieben")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Pass sehen**

Run: `.venv\Scripts\python.exe -m pytest tests/test_variants.py -v`
Expected: 9 PASSED.

- [ ] **Step 5: Commit**

```bash
git add pipeline/variants.py tests/test_variants.py
git commit -m "$(cat <<'EOF'
feat: LLM-Query-Varianten-Generator fuer FAQ-Fragen

pipeline/variants.py generiert pro FAQ 3-5 Umformulierungen per Haiku und
cached sie in data/variants.json -- offline, einmalig, wie der Scraper.
Fehler pro Frage stoppen den Lauf nicht (Muster aus eval/judge.py).

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
EOF
)"
```

---

### Task 3: Varianten-Embedding im Index (`pipeline/index.py`, `app/config.py`)

**Files:**
- Modify: `pipeline/index.py`
- Modify: `app/config.py`
- Test: `tests/test_index.py`

**Interfaces:**
- Consumes: `data/variants.json`-Format aus Task 2 (`{faq_id: [frage, ...]}`).
- Produces: `build_index(corpus_path, index_dir, encoder=None, variants_path: Path | None = None)` — Chroma-Collection-Einträge tragen jetzt `metadata={"canonical_id": ..., "category": ...}` (FAQs) bzw. `{"canonical_id": ...}` (page_chunks und Varianten). Task 4 (`app/retrieval.py`) liest `canonical_id` aus den Query-Metadaten.

- [ ] **Step 1: Failing Tests schreiben** — in `tests/test_index.py` ans Ende der Datei anhängen:

```python
def test_build_index_stores_canonical_id_and_category_metadata(tmp_path):
    corpus_path = tmp_path / "corpus.json"
    corpus_path.write_text(json.dumps({"scraped_at": "2026-08-20", "documents": [FAQ, CHUNK]}),
                           encoding="utf-8")
    index_dir = tmp_path / "index"
    build_index(corpus_path, index_dir, encoder=fake_encoder)

    import chromadb
    coll = chromadb.PersistentClient(path=str(index_dir / "chroma")).get_collection("docs")
    got = coll.get(ids=["faq-0001", "info-buyer-protection-0001"], include=["metadatas"])
    by_id = dict(zip(got["ids"], got["metadatas"]))
    assert by_id["faq-0001"] == {"canonical_id": "faq-0001", "category": "Kaufen"}
    assert by_id["info-buyer-protection-0001"] == {"canonical_id": "info-buyer-protection-0001"}


def test_build_index_embeds_variants_pointing_to_canonical_faq(tmp_path):
    corpus_path = tmp_path / "corpus.json"
    corpus_path.write_text(json.dumps({"scraped_at": "2026-08-20", "documents": [FAQ, CHUNK]}),
                           encoding="utf-8")
    variants_path = tmp_path / "variants.json"
    variants_path.write_text(json.dumps({"faq-0001": ["Wie läuft der Käuferschutz ab?"]}),
                             encoding="utf-8")
    index_dir = tmp_path / "index"

    build_index(corpus_path, index_dir, encoder=fake_encoder, variants_path=variants_path)

    import chromadb
    coll = chromadb.PersistentClient(path=str(index_dir / "chroma")).get_collection("docs")
    assert coll.count() == 3  # 2 Original-Docs + 1 Variante
    got = coll.get(ids=["faq-0001#v1"], include=["metadatas"])
    assert got["metadatas"][0] == {"canonical_id": "faq-0001"}


def test_build_index_ignores_variants_for_faq_ids_not_in_corpus(tmp_path):
    corpus_path = tmp_path / "corpus.json"
    corpus_path.write_text(json.dumps({"scraped_at": "2026-08-20", "documents": [FAQ, CHUNK]}),
                           encoding="utf-8")
    variants_path = tmp_path / "variants.json"
    variants_path.write_text(json.dumps({"faq-9999": ["Verwaiste Variante"]}), encoding="utf-8")
    index_dir = tmp_path / "index"

    build_index(corpus_path, index_dir, encoder=fake_encoder, variants_path=variants_path)

    import chromadb
    coll = chromadb.PersistentClient(path=str(index_dir / "chroma")).get_collection("docs")
    assert coll.count() == 2


def test_build_index_without_variants_path_behaves_as_before(tmp_path):
    corpus_path = tmp_path / "corpus.json"
    corpus_path.write_text(json.dumps({"scraped_at": "2026-08-20", "documents": [FAQ, CHUNK]}),
                           encoding="utf-8")
    index_dir = tmp_path / "index"

    build_index(corpus_path, index_dir, encoder=fake_encoder)

    import chromadb
    coll = chromadb.PersistentClient(path=str(index_dir / "chroma")).get_collection("docs")
    assert coll.count() == 2
```

- [ ] **Step 2: Fail sehen**

Run: `.venv\Scripts\python.exe -m pytest tests/test_index.py -v`
Expected: erste zwei neue Tests FAIL (keine Metadaten gesetzt / `variants_path` unbekanntes Argument), letzte zwei PASS (Verhalten heute schon so).

- [ ] **Step 3: Implementieren** — `pipeline/index.py`: nach `doc_search_text` (vor `DEDUPE_THRESHOLD`, Zeile 27) einfügen:

```python
def _doc_metadata(doc: dict) -> dict:
    meta = {"canonical_id": doc["id"]}
    if doc["type"] == "faq":
        meta["category"] = doc["category"]
    return meta
```

Nach `_default_encoder` (vor `build_index`, Zeile 65) einfügen:

```python
def _variant_entries(
    docs: list[dict], variants_path: Path, encoder
) -> tuple[list[str], list[list[float]], list[dict]]:
    """Zusätzliche Chroma-Einträge für LLM-generierte FAQ-Umformulierungen.

    Zeigen per canonical_id-Metadatum auf denselben Antwort-Chunk zurück;
    BM25 bleibt unangetastet -- Varianten adressieren gezielt den
    Embedding-Pfad (siehe Architektur-Begründung in variants.py).
    """
    if not variants_path.exists():
        return [], [], []
    variants: dict[str, list[str]] = json.loads(variants_path.read_text(encoding="utf-8"))
    faq_ids = {d["id"] for d in docs if d["type"] == "faq"}

    ids: list[str] = []
    texts: list[str] = []
    for faq_id, questions in variants.items():
        if faq_id not in faq_ids:
            continue  # Variante zu einem Dedupe-entfernten oder entfallenen FAQ.
        for i, question in enumerate(questions, 1):
            ids.append(f"{faq_id}#v{i}")
            texts.append(question)

    if not ids:
        return [], [], []
    embeddings = encoder(texts)
    metadatas = [{"canonical_id": vid.split("#v")[0]} for vid in ids]
    return ids, embeddings, metadatas
```

`build_index` ersetzen durch:

```python
def build_index(
    corpus_path: Path, index_dir: Path, encoder=None, variants_path: Path | None = None
) -> None:
    encoder = encoder or _default_encoder
    docs = json.loads(corpus_path.read_text(encoding="utf-8"))["documents"]
    index_dir.mkdir(parents=True, exist_ok=True)

    client = chromadb.PersistentClient(path=str(index_dir / "chroma"))
    try:
        client.delete_collection("docs")
    except NotFoundError:
        pass  # Idempotentes Aufräumen: beim ersten Lauf existiert die Collection noch nicht.
    coll = client.create_collection("docs", metadata={"hnsw:space": "cosine"})
    docs, embeddings = dedupe_docs(docs, encoder([doc_embed_text(d) for d in docs]))
    ids = [d["id"] for d in docs]
    metadatas = [_doc_metadata(d) for d in docs]

    if variants_path is not None:
        variant_ids, variant_embeddings, variant_metadatas = _variant_entries(
            docs, variants_path, encoder
        )
        ids += variant_ids
        embeddings += variant_embeddings
        metadatas += variant_metadatas

    coll.add(ids=ids, embeddings=embeddings, metadatas=metadatas)

    bm25 = BM25Okapi([tokenize(doc_search_text(d)) for d in docs])
    with open(index_dir / "bm25.pkl", "wb") as f:
        pickle.dump({"doc_ids": [d["id"] for d in docs], "bm25": bm25}, f)


if __name__ == "__main__":
    build_index(settings.corpus_path, settings.index_dir, variants_path=settings.variants_path)
    print(f"Index nach {settings.index_dir} geschrieben")
```

In `app/config.py` nach `corpus_path` (Zeile 12) einfügen:

```python
    variants_path: Path = Path("data/variants.json")
```

- [ ] **Step 4: Pass sehen**

Run: `.venv\Scripts\python.exe -m pytest tests/test_index.py -v`
Expected: alle PASSED (bestehende + 4 neue).

- [ ] **Step 5: Commit**

```bash
git add pipeline/index.py app/config.py tests/test_index.py
git commit -m "$(cat <<'EOF'
feat: Query-Varianten und Kategorie-Metadaten im Chroma-Index

build_index() nimmt optional variants_path entgegen und embedded jede
FAQ-Umformulierung als eigenen Chroma-Eintrag (id faq-000N#vM), verknuepft
per canonical_id-Metadatum mit dem Original. BM25 bleibt unveraendert.
Chroma-Eintraege tragen jetzt zusaetzlich category (FAQs).

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
EOF
)"
```

---

### Task 4: Kanonische-ID-Auflösung im Retrieval (`app/retrieval.py`)

**Files:**
- Modify: `app/retrieval.py`
- Test: `tests/test_retrieval.py`

**Interfaces:**
- Consumes: `canonical_id`-Metadatum aus Task 3.
- Produces: `_dedupe_ranking(ids: list[str]) -> list[str]` (interner Helper, keine externen Konsumenten).

- [ ] **Step 1: Failing Test schreiben** — in `tests/test_retrieval.py` ans Ende anhängen:

```python
def test_variant_hit_resolves_to_canonical_doc(tmp_path):
    """Query matcht nur die generierte Variante, nicht die Original-Frage direkt --
    Chroma liefert die Varianten-ID zurueck, der Retriever muss sie auf faq-0001
    zurueckmappen."""
    docs = [
        {"id": "faq-0001", "type": "faq", "question": "Wie funktioniert der Käuferschutz?",
         "answer": "Er sichert Zahlungen ab.", "category": "Kaufen",
         "url": "https://www.chrono24.de/info/faqs.htm"},
        {"id": "faq-0002", "type": "faq", "question": "Wie verkaufe ich eine Uhr?",
         "answer": "Über ein Verkäuferkonto.", "category": "Verkaufen",
         "url": "https://www.chrono24.de/info/faqs.htm"},
    ]
    corpus_path = tmp_path / "corpus.json"
    corpus_path.write_text(json.dumps({"scraped_at": "2026-08-20", "documents": docs}),
                           encoding="utf-8")
    variants_path = tmp_path / "variants.json"
    variants_path.write_text(json.dumps({"faq-0001": ["Was deckt der Kaeuferschutz ab?"]}),
                             encoding="utf-8")

    # Nur die Variante bekommt den Such-Vektor -- die Original-Frage liegt bewusst
    # weit weg, ein Treffer ist also nur ueber die Variante moeglich.
    vecs = {
        "Wie funktioniert der Käuferschutz?": [0.0, 0.0, 1.0],
        "Wie verkaufe ich eine Uhr?": [0.0, 1.0, 0.0],
        "Was deckt der Kaeuferschutz ab?": [1.0, 0.0, 0.0],
    }

    def encode(text):
        return vecs.get(text, [1.0, 0.0, 0.0])

    index_dir = tmp_path / "index"
    build_index(corpus_path, index_dir, encoder=lambda texts: [encode(t) for t in texts],
                variants_path=variants_path)
    retriever = Retriever(index_dir, corpus_path, encoder=encode, reranker=False)

    docs_out = retriever.retrieve("Was deckt der Kaeuferschutz ab?", top_k=5)
    assert docs_out[0].id == "faq-0001"
    assert [d.id for d in docs_out].count("faq-0001") == 1
```

- [ ] **Step 2: Fail sehen**

Run: `.venv\Scripts\python.exe -m pytest tests/test_retrieval.py -v`
Expected: FAIL — `KeyError` oder `AssertionError`, weil `res["ids"]` (roh, `faq-0001#v1`) statt der kanonischen ID verwendet wird und `self.docs["faq-0001#v1"]` nicht existiert.

- [ ] **Step 3: Implementieren** — in `app/retrieval.py` nach `rrf_fuse` (Zeile 38) einfügen:

```python
def _dedupe_ranking(ids: list[str]) -> list[str]:
    """Erster (bester) Treffer pro kanonischer ID gewinnt -- Varianten-Duplikate raus."""
    seen: set[str] = set()
    result: list[str] = []
    for doc_id in ids:
        if doc_id not in seen:
            seen.add(doc_id)
            result.append(doc_id)
    return result
```

In `Retriever.retrieve` (Zeilen 72–76) die Query- und Ranking-Zeilen ersetzen durch:

```python
    def retrieve(self, query: str, top_k: int = 5) -> list[RetrievedDoc]:
        n = min(TOP_K_CANDIDATES, len(self.doc_ids))
        res = self.collection.query(
            query_embeddings=[list(self.encoder(query))], n_results=n,
            include=["metadatas", "distances"],
        )
        vector_ranking = _dedupe_ranking([m["canonical_id"] for m in res["metadatas"][0]])
        best_sim = 1.0 - res["distances"][0][0] if res["distances"][0] else 0.0
```

(Rest der Methode ab `# Synonym-Expansion nur hier …` unverändert.)

- [ ] **Step 4: Pass sehen**

Run: `.venv\Scripts\python.exe -m pytest tests/test_retrieval.py -v`
Expected: alle PASSED (bestehende + 1 neuer Test).

- [ ] **Step 5: Vollen Testlauf gegen Regressionen prüfen**

Run: `.venv\Scripts\python.exe -m pytest tests/ -v`
Expected: alle PASSED.

- [ ] **Step 6: Commit**

```bash
git add app/retrieval.py tests/test_retrieval.py
git commit -m "$(cat <<'EOF'
feat: Varianten-Treffer im Retrieval auf kanonische FAQ-ID abbilden

Retriever.retrieve() liest jetzt canonical_id aus den Chroma-Metadaten
statt der rohen Dokument-ID und dedupliziert das Vector-Ranking, bevor es
in die RRF-Fusion geht -- ein Varianten-Treffer taucht sonst als
unbekannte ID auf und doppelt gegen sein Original.

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
EOF
)"
```

---

### Task 5: Varianten generieren, Index neu bauen, Ablation ehrlich messen

**Achtung:** Dieser Task ruft echte Haiku-API-Calls auf (~100+, geschätzt niedriger einstelliger Cent-Betrag) und verändert den committeten Index. Vor Ausführung explizit beim User nachfragen (siehe Global Constraints).

**Files:**
- Create: `data/variants.json` (committen)
- Modify: `data/index/` (Chroma + `bm25.pkl`, committen)
- Modify: `README.md` (Ablation-Tabelle — Inhalt aus den tatsächlichen Messwerten dieses Tasks)

- [ ] **Step 1: Varianten generieren**

Voraussetzung: `ANTHROPIC_API_KEY` in `.env` gesetzt (siehe README „Lokal starten").

Run: `.venv\Scripts\python.exe -m pipeline.variants`
Expected: Ausgabe `N Varianten für M/M FAQs nach data/variants.json geschrieben` (M = Anzahl FAQ-Dokumente in `data/corpus.json`).

- [ ] **Step 2: Index neu bauen**

Run: `.venv\Scripts\python.exe -m pipeline.index`
Expected: Ausgabe `Index nach data\index geschrieben`.

- [ ] **Step 3: Tuning- und Holdout-Hit-Rate messen**

Run: `.venv\Scripts\python.exe -m eval.run_eval`
Run: `.venv\Scripts\python.exe -m eval.run_eval --questions eval/questions_holdout.json`

Beide Ausgaben notieren (`Hit-Rate@5: XX% (n/33)` bzw. `.../15)`).

- [ ] **Step 4: Ehrlich vergleichen und dokumentieren**

Baseline vor diesem Task: Tuning 91 % (30/33), Holdout 87 % (13/15) (siehe README, aktueller Stand).

In `README.md` in der Ablation-Tabelle (nach Zeile 97, der Synonym-Expansion-Zeile) eine neue Zeile mit den in Step 3 gemessenen Zahlen einfügen, z. B.:

```
| + Query-Varianten (LLM-Umformulierungen, nur Embedding-Pfad) | **<gemessene Tuning-Hit-Rate>** |
```

Direkt danach im Fließtext (analog zum bestehenden Synonym-Expansion-Absatz ab Zeile 116) einen kurzen Absatz ergänzen, der den Holdout-Wert nennt und **ehrlich** einordnet — auch falls die Zahl gleich bleibt oder sinkt (Projekt-Ethos, siehe bestehende Zeilen 100–107 zum verworfenen Titel-Bonus). Beispieltext als Vorlage, mit den echten Zahlen aus Step 3 ausgefüllt:

```
Query-Varianten (`pipeline/variants.py`) generieren pro FAQ per Haiku 3–5
Umformulierungen und embedden sie zusätzlich zur Originalfrage, verknüpft
über ein `canonical_id`-Metadatum in Chroma. Ziel: Nutzerformulierungen, die
weiter von der FAQ-Frage abweichen, als es die multilingualen Embeddings
allein auffangen. Gemessen: Tuning <X> % (<n>/33), Holdout <Y> % (<n>/15).
```

Falls die Tuning- oder Holdout-Zahl gegenüber der Baseline **sinkt**: den Schritt trotzdem dokumentieren (nicht verwerfen ohne Analyse) und in `eval/run_eval.py` die Konstanten `TUNING_MIN_HIT_RATE`/`HOLDOUT_MIN_HIT_RATE` NICHT über die tatsächlich gemessene Zahl anheben — die Gate-Schwellen aus Task 1 bleiben der Sicherheitsboden, nicht die neue Bestmarke.

Falls beide Zahlen **steigen oder gleich bleiben**: `TUNING_MIN_HIT_RATE`/`HOLDOUT_MIN_HIT_RATE` in `eval/run_eval.py` optional auf `gemessene Rate - 0.03` anheben (Puffer für Rauschen), um die neue Bestmarke künftig auch als Regressionsschutz zu nutzen.

- [ ] **Step 5: Vollen Testlauf prüfen**

Run: `.venv\Scripts\python.exe -m pytest tests/ -v`
Expected: alle PASSED (Index-Änderung darf bestehende Unit-Tests nicht betreffen, die bauen ihren eigenen Test-Index in `tmp_path`).

- [ ] **Step 6: Gate lokal gegen den neuen Index prüfen**

Run: `.venv\Scripts\python.exe -m eval.run_eval --gate`
Expected: Exit-Code 0.

- [ ] **Step 7: Commit**

```bash
git add data/variants.json data/index/ README.md eval/run_eval.py
git commit -m "$(cat <<'EOF'
feat: Query-Varianten in Index eingebaut, Ablation gemessen

data/variants.json (LLM-generierte FAQ-Umformulierungen) und neu gebauter
Index committet. Ablation-Tabelle im README um die gemessenen Tuning- und
Holdout-Hit-Raten ergaenzt -- ehrlich, auch falls sie nicht steigen.

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
EOF
)"
```

---

### Task 6: Faithfulness-Gate in `eval/judge.py` + CI-Job (nur push auf main)

**Files:**
- Modify: `eval/judge.py`
- Modify: `.github/workflows/ci.yml`
- Test: `tests/test_judge.py`

**Interfaces:**
- Produces: `check_gate(summary: dict) -> list[str]` (gleiche Signaturform wie Task 1, operiert auf `aggregate()`-Ergebnis statt zwei Raten).

- [ ] **Step 1: Failing Test schreiben** — in `tests/test_judge.py` den Import um `check_gate`/`MIN_FAITHFUL_RATE` erweitern (Zeile 1–8) und ans Ende der Datei anhängen:

```python
def test_check_gate_passes_at_minimum_faithful_rate():
    assert check_gate({"faithful_rate": MIN_FAITHFUL_RATE, "n": 10, "answered_counts": {}}) == []


def test_check_gate_fails_below_minimum_faithful_rate():
    failures = check_gate({"faithful_rate": MIN_FAITHFUL_RATE - 0.01, "n": 10, "answered_counts": {}})
    assert len(failures) == 1
    assert "Faithful" in failures[0]
```

Import-Zeile:

```python
from eval.judge import (
    JUDGE_SYSTEM,
    MIN_FAITHFUL_RATE,
    aggregate,
    build_judge_prompt,
    check_gate,
    judge_one,
    parse_verdict,
)
```

- [ ] **Step 2: Fail sehen**

Run: `.venv\Scripts\python.exe -m pytest tests/test_judge.py -v`
Expected: ImportError `MIN_FAITHFUL_RATE`/`check_gate`.

- [ ] **Step 3: Implementieren** — in `eval/judge.py` nach `MAX_JUDGE_TOKENS = 300` (Zeile 19) einfügen:

```python
# Aktuell gemessen: 100 % (33/33, siehe README). Puffer fuer Einzelfrage-Rauschen
# auf dem kleinen Sample, keine Verschaerfung ohne erneuten vollen Judge-Lauf.
MIN_FAITHFUL_RATE = 0.90
```

Nach `aggregate` (Zeile 118, vor `_run_all`) einfügen:

```python
def check_gate(summary: dict) -> list[str]:
    """Prüft die Faithful-Rate gegen die Mindestschwelle. Leer = Gate bestanden."""
    failures = []
    if summary["faithful_rate"] < MIN_FAITHFUL_RATE:
        failures.append(
            f"Faithful-Rate {summary['faithful_rate']:.0%} unter Minimum {MIN_FAITHFUL_RATE:.0%}"
        )
    return failures
```

`main()` (Zeile 156–172) ersetzen durch:

```python
def main() -> None:
    import sys

    from app.llm import get_client
    from app.retrieval import Retriever

    questions = json.loads(QUESTIONS_PATH.read_text(encoding="utf-8"))
    retriever = Retriever(settings.index_dir, settings.corpus_path)
    client = get_client()

    results = asyncio.run(_run_all(questions, retriever, client))
    summary = aggregate(results)
    _print_report(results, summary)

    RESULTS_PATH.write_text(
        json.dumps({"summary": summary, "results": results}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"\nRohergebnisse geschrieben nach {RESULTS_PATH}")

    if "--gate" in sys.argv:
        failures = check_gate(summary)
        for failure in failures:
            print(f"GATE FAIL: {failure}")
        sys.exit(1 if failures else 0)
```

- [ ] **Step 4: Pass sehen**

Run: `.venv\Scripts\python.exe -m pytest tests/test_judge.py -v`
Expected: alle PASSED (bestehende + 2 neue).

- [ ] **Step 5: CI-Job ergänzen** — in `.github/workflows/ci.yml` nach dem `eval-gate`-Job (aus Task 1) anhängen:

```yaml
  quality-gate:
    if: github.event_name == 'push' && github.ref == 'refs/heads/main'
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.12"
          cache: pip
      - run: pip install -r requirements-dev.txt
      - run: python -m eval.judge --gate
        env:
          ANTHROPIC_API_KEY: ${{ secrets.ANTHROPIC_API_KEY }}
```

Hinweis für den User (nicht automatisierbar): Repo-Secret `ANTHROPIC_API_KEY` muss in den GitHub-Repo-Settings unter „Secrets and variables → Actions" gesetzt sein, sonst schlägt dieser Job bei jedem Push auf main fehl.

- [ ] **Step 6: Commit**

```bash
git add eval/judge.py tests/test_judge.py .github/workflows/ci.yml
git commit -m "$(cat <<'EOF'
feat: Faithfulness-Gate fuer eval/judge.py, CI-Job nur bei Push auf main

--gate-Modus prueft die Faithful-Rate aus aggregate() gegen eine
Mindestschwelle. Neuer CI-Job quality-gate laeuft nur bei Push auf main
(nicht bei jedem PR), weil er echte Haiku-Calls kostet.

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
EOF
)"
```

---

### Task 7: README final aktualisieren

**Files:**
- Modify: `README.md`

**Interfaces:** keine (reine Dokumentation).

- [ ] **Step 1: „Auf einen Blick"-Bullet ergänzen** — nach der Zeile „**Hybrid-Retrieval:** … 91 % Hit-Rate@5, held-out validiert (87 %)" (Zeilen 15–17) die Zahlen aktualisieren, falls Task 5 sie verändert hat, und einen Halbsatz zu Query-Varianten ergänzen, z. B.:

```
- **Hybrid-Retrieval:** BM25 + Vektorsuche (Chroma), RRF-Fusion,
  Cross-Encoder-Reranker, Synonym-Expansion, LLM-generierte
  Query-Varianten je FAQ — <aktuelle Tuning-Hit-Rate> Hit-Rate@5,
  held-out validiert (<aktuelle Holdout-Hit-Rate>)
```

- [ ] **Step 2: CI-Abschnitt ergänzen** — im Inhaltsverzeichnis (Zeile 33–42) nach `[Tests](#tests) ·` einfügen: `[CI Eval Gate](#ci-eval-gate) ·`. Im `## Tests`-Abschnitt (nach Zeile 313, vor `## Guards`) neuen Abschnitt einfügen:

```markdown
### CI Eval Gate

Zwei automatisierte Qualitäts-Regressionstests in `.github/workflows/ci.yml`,
zusätzlich zu ruff/pytest/Docker-Build:

- **`eval-gate`** (jeder Push und PR, keine API-Kosten): lädt den committeten
  Index und prüft Hit-Rate@5 gegen Tuning- und Holdout-Fragen. Unter der
  Mindestschwelle (`eval/run_eval.py::TUNING_MIN_HIT_RATE` /
  `HOLDOUT_MIN_HIT_RATE`) schlägt der Job fehl.
- **`quality-gate`** (nur bei Push auf main, kostet Haiku-API-Calls): lässt
  `eval/judge.py --gate` über alle Tuning-Fragen laufen und prüft die
  Faithful-Rate gegen `eval/judge.py::MIN_FAITHFUL_RATE`. Braucht das
  Repo-Secret `ANTHROPIC_API_KEY`.

Beide Skripte laufen auch lokal manuell:

```bash
python -m eval.run_eval --gate
python -m eval.judge --gate
```
```

- [ ] **Step 3: Commit**

```bash
git add README.md
git commit -m "$(cat <<'EOF'
docs: CI Eval Gate und Query-Varianten im README dokumentiert

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
EOF
)"
```

---

## Reihenfolge und Unabhängigkeit

Task 1 (Hit-Rate-Gate) ist unabhängig von Task 2–4 und schützt den aktuellen Stand sofort — sinnvoll zuerst. Task 2 → 3 → 4 bauen aufeinander auf (Varianten generieren → einbetten → auflösen) und müssen in dieser Reihenfolge laufen. Task 5 (echte API-Kosten) braucht Task 2–4 fertig. Task 6 (Faithfulness-Gate) ist unabhängig von Task 2–5 und könnte auch direkt nach Task 1 laufen. Task 7 (README) sinnvollerweise zuletzt, da er Zahlen aus Task 5 übernimmt.
