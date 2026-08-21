# Handover-Briefing (Stufe B) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** `POST /api/handover` erzeugt aus der Chat-History ein LLM-extrahiertes, deterministisch validiertes Übergabe-Briefing; das Frontend bietet die Übergabe per Button und nach Sackgassen an und rendert das Briefing als Ampel-Karte.

**Architecture:** Portierung aus `C:\Users\Marco\02_Portfolio\Handover Brief Generator` (extract/orchestrator/validate). Neues Modul `app/handover.py` (Zeilen-IDs, Extractor-Prompt, Orchestrator mit MAX_ATTEMPTS=2), generische Claim-Validierung in `app/faithcheck.py`, Endpoint mit Guards in `app/main.py`, Vanilla-JS-Karte in `static/`.

**Tech Stack:** FastAPI, Pydantic, slowapi, Anthropic AsyncClient (Haiku via `settings.model`), pytest, Vanilla JS.

**Spec:** `docs/superpowers/specs/2026-08-21-handover-briefing-design.md`

## Global Constraints

- Testlauf immer mit `.venv\Scripts\python.exe -m pytest …` (globales Python hat keine Abhängigkeiten).
- Deutsch für alle nutzersichtbaren Texte und Prompts.
- Frontend: ausschließlich `textContent`/`createElement`, niemals `innerHTML`.
- Keine stillen Fallbacks: Fehler werden sichtbar (HTTP 502/429), `rejected` ist ein sichtbarer Zustand.
- `data/index/`-Änderungen (Chroma mutiert beim Öffnen) nie committen: `git restore data/index/`.
- Commits: Conventional Commits, Body deutsch, Co-Authored-By-Zeile wie in den letzten Commits.

---

### Task 1: `validate_claims` in `app/faithcheck.py`

**Files:**
- Modify: `app/faithcheck.py` (Ende der Datei; `_tokenize` bei Zeile ~36)
- Test: `tests/test_faithcheck.py`

**Interfaces:**
- Consumes: `SentenceCheck`, `score_overlap`, `PASS_THRESHOLD` (vorhanden).
- Produces: `validate_claims(claims: list[dict], lines_by_id: dict[str, str]) -> list[SentenceCheck]` — `claims`-Elemente `{"text": str, "source_lines": list[str]}`; `SentenceCheck.sources` trägt hier die Zeilen-IDs (Strings). Task 3 ruft das auf.

- [ ] **Step 1: Failing Tests schreiben** — ans Ende von `tests/test_faithcheck.py` anhängen, Import oben erweitern um `validate_claims`:

```python
# --- validate_claims (Handover-Briefing) ---

LINES = {"M01": "Meine Rolex Daytona ist nach zwei Wochen immer noch nicht angekommen.",
         "M02": "Der Käuferschutz sichert deine Zahlung vollständig ab."}


def test_claim_covered_by_cited_line_is_pass():
    claims = [{"text": "Rolex Daytona ist nicht angekommen", "source_lines": ["M01"]}]
    [check] = validate_claims(claims, LINES)
    assert check.status == "PASS"
    assert check.score >= 0.5
    assert check.sources == ["M01"]


def test_claim_with_low_overlap_is_weak():
    claims = [{"text": "Kunde wartet angeblich vergeblich auf Rückerstattung des Kaufpreises",
               "source_lines": ["M01"]}]
    [check] = validate_claims(claims, LINES)
    assert check.status == "WEAK"
    assert 0 < check.score < 0.5


def test_claim_with_zero_overlap_is_fail():
    claims = [{"text": "Bitcoin Kurs steigt enorm", "source_lines": ["M02"]}]
    [check] = validate_claims(claims, LINES)
    assert check.status == "FAIL"
    assert check.score == 0.0


def test_claim_without_source_lines_is_fail():
    claims = [{"text": "Rolex Daytona ist nicht angekommen", "source_lines": []}]
    [check] = validate_claims(claims, LINES)
    assert check.status == "FAIL"


def test_claim_with_unknown_line_id_is_fail():
    claims = [{"text": "Rolex Daytona ist nicht angekommen", "source_lines": ["M99"]}]
    [check] = validate_claims(claims, LINES)
    assert check.status == "FAIL"


def test_tokenizer_ignores_line_id_tokens():
    # "M01" im Claim-Text darf den Score nicht verfälschen (analog [n]-Marker).
    claims = [{"text": "M01 Rolex Daytona ist nicht angekommen", "source_lines": ["M01"]}]
    [check] = validate_claims(claims, LINES)
    assert check.status == "PASS"


def test_claims_spanning_multiple_lines_concatenate_sources():
    claims = [{"text": "Rolex nicht angekommen, Käuferschutz sichert Zahlung ab",
               "source_lines": ["M01", "M02"]}]
    [check] = validate_claims(claims, LINES)
    assert check.status == "PASS"
```

- [ ] **Step 2: Fail sehen**

Run: `.venv\Scripts\python.exe -m pytest tests/test_faithcheck.py -q`
Expected: ImportError `validate_claims`.

- [ ] **Step 3: Implementieren** — in `app/faithcheck.py`:

`_tokenize` ersetzen durch:

```python
def _tokenize(text: str) -> set[str]:
    text = _CITATION_RE.sub(" ", text)
    tokens = set(re.findall(r"[a-zäöüß0-9]+", text.lower()))
    # Zeilen-IDs (m01) analog zu L-IDs im Schwesterprojekt herausfiltern.
    return {t for t in tokens if not re.fullmatch(r"m\d+", t)}
```

`SentenceCheck.sources`-Annotation auf `list[int] | list[str]` ändern (Chat: Zitatnummern, Briefing: Zeilen-IDs). Ans Dateiende:

```python
def validate_claims(claims: list[dict], lines_by_id: dict[str, str]) -> list[SentenceCheck]:
    """Prüft Briefing-Claims gegen die referenzierten Chat-Zeilen (Stufe B).

    Fehlende source_lines oder unbekannte Zeilen-IDs → FAIL, sonst
    Token-Overlap gegen den konkatenierten Zeilentext."""
    checks: list[SentenceCheck] = []
    for claim in claims:
        text = claim["text"]
        source_ids = claim.get("source_lines") or []
        if not source_ids or any(sid not in lines_by_id for sid in source_ids):
            checks.append(SentenceCheck(text, "FAIL", 0.0, source_ids))
            continue
        source_text = " ".join(lines_by_id[sid] for sid in source_ids)
        score = score_overlap(text, source_text)
        if score == 0.0:
            status = "FAIL"
        elif score < PASS_THRESHOLD:
            status = "WEAK"
        else:
            status = "PASS"
        checks.append(SentenceCheck(text, status, round(score, 3), source_ids))
    return checks
```

- [ ] **Step 4: Grün sehen**

Run: `.venv\Scripts\python.exe -m pytest tests/test_faithcheck.py -q` → alle grün; danach `.venv\Scripts\python.exe -m pytest tests/ -q` → komplette Suite grün (der `m\d+`-Filter darf keinen Bestandstest brechen).

- [ ] **Step 5: Commit**

```bash
git add app/faithcheck.py tests/test_faithcheck.py
git commit -m "feat: validate_claims prueft Briefing-Claims gegen Chat-Zeilen"
```

---

### Task 2: `app/handover.py` — Zeilen, Prompt, Parsing

**Files:**
- Create: `app/handover.py`
- Test: `tests/test_handover.py` (neu)

**Interfaces:**
- Consumes: nichts Projektspezifisches (reine Funktionen).
- Produces: `build_lines(messages: list[dict]) -> list[dict]` (Elemente `{"id": "M01", "actor": "Kunde"|"Bot", "text": str}`); `build_prompt(lines: list[dict], previous_failure_note: str | None = None) -> str`; `parse_response(raw_text: str) -> dict` (wirft `ValueError` bei fehlenden Pflichtfeldern, `json.JSONDecodeError` bei kaputtem JSON); `SYSTEM_PROMPT: str`; `normalize_briefing(briefing: dict) -> list[dict]` (Claims-Form aus Task 1). Task 3 baut darauf.

- [ ] **Step 1: Failing Tests schreiben** — `tests/test_handover.py` anlegen:

```python
"""Tests für die Handover-Briefing-Erzeugung (Stufe B)."""
import json

import pytest

from app.handover import (
    SYSTEM_PROMPT,
    build_lines,
    build_prompt,
    normalize_briefing,
    parse_response,
)

MESSAGES = [
    {"role": "user", "content": "Meine Rolex ist nicht angekommen."},
    {"role": "assistant", "content": "Der Käuferschutz sichert deine Zahlung ab. [1]"},
]


def test_build_lines_assigns_ids_and_actors():
    lines = build_lines(MESSAGES)
    assert lines == [
        {"id": "M01", "actor": "Kunde", "text": "Meine Rolex ist nicht angekommen."},
        {"id": "M02", "actor": "Bot",
         "text": "Der Käuferschutz sichert deine Zahlung ab. [1]"},
    ]


def test_build_prompt_contains_lines_and_failure_note():
    lines = build_lines(MESSAGES)
    prompt = build_prompt(lines, previous_failure_note="Aussage X war unbelegt")
    assert "M01 [Kunde]: Meine Rolex ist nicht angekommen." in prompt
    assert "Aussage X war unbelegt" in prompt


def test_system_prompt_is_german_chat_schema():
    assert "Chatverlauf" in SYSTEM_PROMPT
    assert "source_lines" in SYSTEM_PROMPT
    assert "claims" in SYSTEM_PROMPT


def test_parse_response_strips_markdown_fence():
    briefing = {"situation": {"text": "s", "source_lines": ["M01"]},
                "history": {"text": "h", "source_lines": ["M01"]},
                "sentiment": {"label": "frustriert", "quote": "q", "source_lines": ["M01"]},
                "open_question": {"text": "o", "source_lines": ["M01"]},
                "claims": []}
    raw = "```json\n" + json.dumps(briefing) + "\n```"
    assert parse_response(raw) == briefing


def test_parse_response_missing_field_raises():
    with pytest.raises(ValueError, match="claims"):
        parse_response(json.dumps({"situation": {}, "history": {},
                                   "sentiment": {}, "open_question": {}}))


def test_normalize_briefing_flattens_all_fields_to_claims():
    briefing = {"situation": {"text": "s", "source_lines": ["M01"]},
                "history": {"text": "h", "source_lines": ["M01", "M02"]},
                "sentiment": {"label": "frustriert", "quote": "q", "source_lines": ["M01"]},
                "open_question": {"text": "o", "source_lines": ["M02"]},
                "claims": [{"text": "c1", "source_lines": ["M01"]}]}
    claims = normalize_briefing(briefing)
    assert claims == [
        {"text": "s", "source_lines": ["M01"]},
        {"text": "h", "source_lines": ["M01", "M02"]},
        {"text": "q", "source_lines": ["M01"]},
        {"text": "o", "source_lines": ["M02"]},
        {"text": "c1", "source_lines": ["M01"]},
    ]
```

- [ ] **Step 2: Fail sehen**

Run: `.venv\Scripts\python.exe -m pytest tests/test_handover.py -q`
Expected: `ModuleNotFoundError: No module named 'app.handover'`.

- [ ] **Step 3: Implementieren** — `app/handover.py` anlegen:

```python
"""Handover-Briefing: Chat-History → geprüftes Übergabe-Briefing (Stufe B).

Portiert aus dem Handover Brief Generator (src/extract.py, src/orchestrator.py):
das LLM extrahiert ein Briefing mit Zeilen-Zitaten, der deterministische
Validator aus app/faithcheck.py prüft jede Aussage per Token-Overlap."""
from __future__ import annotations

import json

from app import faithcheck
from app.config import settings

MAX_ATTEMPTS = 2
MAX_BRIEFING_TOKENS = 1024

SYSTEM_PROMPT = """Du extrahierst ein strukturiertes Übergabe-Briefing aus einem \
Chatverlauf zwischen einem Kunden und dem FAQ-Bot eines Luxusuhren-Marktplatzes. \
Antworte ausschließlich mit JSON in exakt diesem Schema:

{
  "situation": {"text": "...", "source_lines": ["M01"]},
  "history": {"text": "...", "source_lines": ["M01", "M02"]},
  "sentiment": {"label": "...", "quote": "wörtliches Zitat aus dem Chat", "source_lines": ["M01"]},
  "open_question": {"text": "...", "source_lines": ["M01"]},
  "claims": [{"text": "...", "source_lines": ["M01"]}]
}

Jedes Feld muss über source_lines exakt die Zeilen-IDs referenzieren, aus denen \
die Aussage stammt. Erfinde nichts, das nicht durch die referenzierten Zeilen \
gedeckt ist. Falls zwei Aussagen im Chat widersprüchlich sind, gib beide als \
separate Claims mit ihren jeweiligen Quellzeilen an, statt sie zu glätten."""

_ACTOR = {"user": "Kunde", "assistant": "Bot"}
_REQUIRED_FIELDS = {"situation", "history", "sentiment", "open_question", "claims"}


def build_lines(messages: list[dict]) -> list[dict]:
    return [{"id": f"M{i:02d}", "actor": _ACTOR[m["role"]], "text": m["content"]}
            for i, m in enumerate(messages, 1)]


def build_prompt(lines: list[dict], previous_failure_note: str | None = None) -> str:
    lines_text = "\n".join(f"{l['id']} [{l['actor']}]: {l['text']}" for l in lines)
    prompt = f"Chatverlauf:\n{lines_text}\n\nErzeuge das Handover-Briefing als JSON."
    if previous_failure_note:
        prompt += f"\n\nHinweis: {previous_failure_note}"
    return prompt


def _strip_markdown_fence(raw_text: str) -> str:
    text = raw_text.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1] if "\n" in text else ""
        if text.endswith("```"):
            text = text.rsplit("```", 1)[0]
    return text.strip()


def parse_response(raw_text: str) -> dict:
    data = json.loads(_strip_markdown_fence(raw_text))
    missing = _REQUIRED_FIELDS - data.keys()
    if missing:
        raise ValueError(f"Briefing-Antwort ohne Pflichtfelder: {sorted(missing)}")
    return data


def normalize_briefing(briefing: dict) -> list[dict]:
    claims = []
    for field in ("situation", "history"):
        entry = briefing[field]
        claims.append({"text": entry["text"], "source_lines": entry["source_lines"]})
    sentiment = briefing["sentiment"]
    claims.append({"text": sentiment["quote"], "source_lines": sentiment["source_lines"]})
    entry = briefing["open_question"]
    claims.append({"text": entry["text"], "source_lines": entry["source_lines"]})
    for claim in briefing["claims"]:
        claims.append({"text": claim["text"], "source_lines": claim["source_lines"]})
    return claims
```

(Achtung Reihenfolge in `normalize_briefing`: situation, history, sentiment-quote, open_question, claims — exakt wie im Test.)

- [ ] **Step 4: Grün sehen**

Run: `.venv\Scripts\python.exe -m pytest tests/test_handover.py -q` → grün.

- [ ] **Step 5: Commit**

```bash
git add app/handover.py tests/test_handover.py
git commit -m "feat: Handover-Modul mit Zeilen-IDs, Extractor-Prompt und Parsing"
```

---

### Task 3: Orchestrator `generate_briefing`

**Files:**
- Modify: `app/handover.py` (ans Ende)
- Test: `tests/test_handover.py` (anhängen)

**Interfaces:**
- Consumes: Task-1-`validate_claims`, Task-2-Funktionen; Anthropic-AsyncClient-Form: `await client.messages.create(model=…, max_tokens=…, system=…, messages=[…])` → Response mit `.content` (Blöcke mit `.type`/`.text`) und `.usage.input_tokens`/`.usage.output_tokens` (wie `app/llm.py::rewrite_query`).
- Produces: `async generate_briefing(messages: list[dict], client) -> dict` mit Keys `status` ("ok"|"rejected"), `briefing`, `validation` (list[SentenceCheck]), `lines`, `tokens` (int); bei rejected zusätzlich `failed_claims` (list[str]). Task 4 ruft das auf.

- [ ] **Step 1: Failing Tests schreiben** — an `tests/test_handover.py` anhängen:

```python
# --- generate_briefing (Orchestrator) ---
from types import SimpleNamespace

from app.handover import generate_briefing

LINES_MESSAGES = [
    {"role": "user",
     "content": "Meine Rolex Daytona ist nach zwei Wochen immer noch nicht angekommen."},
    {"role": "assistant",
     "content": "Der Käuferschutz sichert deine Zahlung vollständig ab. [1]"},
]

VALID_BRIEFING = {
    "situation": {"text": "Rolex Daytona ist nach zwei Wochen nicht angekommen",
                  "source_lines": ["M01"]},
    "history": {"text": "Käuferschutz sichert die Zahlung vollständig ab",
                "source_lines": ["M02"]},
    "sentiment": {"label": "besorgt",
                  "quote": "nach zwei Wochen immer noch nicht angekommen",
                  "source_lines": ["M01"]},
    "open_question": {"text": "Wo ist die Rolex Daytona nach zwei Wochen",
                      "source_lines": ["M01"]},
    "claims": [],
}

BAD_BRIEFING = {**VALID_BRIEFING,
                "claims": [{"text": "Kunde verlangt sofortige Kontosperrung wegen Betrug",
                            "source_lines": ["M02"]}]}


def _response(briefing, input_tokens=100, output_tokens=50):
    return SimpleNamespace(
        content=[SimpleNamespace(type="text", text=json.dumps(briefing, ensure_ascii=False))],
        usage=SimpleNamespace(input_tokens=input_tokens, output_tokens=output_tokens))


class FakeClient:
    def __init__(self, responses):
        self.calls = []
        self._responses = list(responses)
        self.messages = self

    async def create(self, **kwargs):
        self.calls.append(kwargs)
        return self._responses[len(self.calls) - 1]


@pytest.mark.asyncio
async def test_valid_briefing_returns_ok_with_one_call():
    client = FakeClient([_response(VALID_BRIEFING)])
    result = await generate_briefing(LINES_MESSAGES, client)
    assert result["status"] == "ok"
    assert len(client.calls) == 1
    assert result["tokens"] == 150
    assert all(c.status in ("PASS", "WEAK") for c in result["validation"])
    assert result["lines"][0]["id"] == "M01"


@pytest.mark.asyncio
async def test_fail_then_valid_retries_with_failure_note():
    client = FakeClient([_response(BAD_BRIEFING), _response(VALID_BRIEFING)])
    result = await generate_briefing(LINES_MESSAGES, client)
    assert result["status"] == "ok"
    assert len(client.calls) == 2
    second_prompt = client.calls[1]["messages"][0]["content"]
    assert "unbelegte Aussage" in second_prompt
    assert "Kontosperrung" in second_prompt
    assert result["tokens"] == 300


@pytest.mark.asyncio
async def test_two_fails_returns_rejected_with_failed_claims():
    client = FakeClient([_response(BAD_BRIEFING), _response(BAD_BRIEFING)])
    result = await generate_briefing(LINES_MESSAGES, client)
    assert result["status"] == "rejected"
    assert len(client.calls) == 2
    assert any("Kontosperrung" in text for text in result["failed_claims"])
```

Hinweis: falls `pytest.mark.asyncio` im Projekt nicht greift (Config-Warnung `asyncio_mode`), das async-Testmuster aus `tests/test_llm.py` übernehmen und exakt so verwenden.

- [ ] **Step 2: Fail sehen**

Run: `.venv\Scripts\python.exe -m pytest tests/test_handover.py -q`
Expected: ImportError `generate_briefing`.

- [ ] **Step 3: Implementieren** — ans Ende von `app/handover.py`:

```python
async def generate_briefing(messages: list[dict], client) -> dict:
    """Extract → Validierung, bei FAIL ein Retry mit Fehlerhinweis, dann rejected.

    Exceptions (kaputtes JSON, API-Fehler) propagieren zum Aufrufer —
    kein stiller Fallback."""
    lines = build_lines(messages)
    lines_by_id = {line["id"]: line["text"] for line in lines}
    tokens = 0
    failure_note = None
    briefing = None
    validation: list = []
    failed: list = []

    for _ in range(MAX_ATTEMPTS):
        response = await client.messages.create(
            model=settings.model,
            max_tokens=MAX_BRIEFING_TOKENS,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": build_prompt(lines, failure_note)}],
        )
        tokens += response.usage.input_tokens + response.usage.output_tokens
        raw = next((b.text for b in response.content if b.type == "text"), "")
        briefing = parse_response(raw)
        validation = faithcheck.validate_claims(normalize_briefing(briefing), lines_by_id)
        failed = [c for c in validation if c.status == "FAIL"]
        if not failed:
            return {"status": "ok", "briefing": briefing, "validation": validation,
                    "lines": lines, "tokens": tokens}
        failure_note = ("Vorherige Antwort hatte unbelegte Aussage(n): "
                        + "; ".join(c.text for c in failed))

    return {"status": "rejected", "briefing": briefing, "validation": validation,
            "failed_claims": [c.text for c in failed], "lines": lines, "tokens": tokens}
```

- [ ] **Step 4: Grün sehen**

Run: `.venv\Scripts\python.exe -m pytest tests/test_handover.py -q` → grün.

- [ ] **Step 5: Commit**

```bash
git add app/handover.py tests/test_handover.py
git commit -m "feat: Briefing-Orchestrator mit Retry und ehrlicher Ablehnung"
```

---

### Task 4: Endpoint `POST /api/handover`

**Files:**
- Modify: `app/main.py` (Import ~Zeile 18, `create_app`-Signatur ~Zeile 54, neuer Endpoint nach dem Chat-Endpoint ~Zeile 125)
- Test: `tests/test_api.py` (anhängen)

**Interfaces:**
- Consumes: `generate_briefing` (Task 3), `ChatMessage`, `TokenBudget`, `limiter`.
- Produces: `POST /api/handover`; `create_app(…, handover_fn=None)` — Tests injizieren `handover_fn`; Response-JSON `{status, briefing, validation: [{text, status, score, sources}], lines}` (+ `failed_claims` bei rejected).

- [ ] **Step 1: Failing Tests schreiben** — an `tests/test_api.py` anhängen:

```python
# --- POST /api/handover ---

HANDOVER_RESULT = {
    "status": "ok",
    "briefing": {"situation": {"text": "s", "source_lines": ["M01"]}},
    "validation": [],
    "lines": [{"id": "M01", "actor": "Kunde", "text": "Frage"}],
    "tokens": 150,
}


def make_handover_client(tmp_path, budget=None, handover_fn=None):
    async def default_fn(messages, client):
        return dict(HANDOVER_RESULT)

    app = create_app(retriever=FakeRetriever([DOC]),
                     budget=budget or TokenBudget(tmp_path / "h.sqlite3", daily_limit=1000),
                     answer_fn=fake_answer_fn, rewrite_fn=fake_rewrite_fn,
                     llm_client=object(), handover_fn=handover_fn or default_fn)
    return TestClient(app)


HANDOVER_BODY = {"messages": [
    {"role": "user", "content": "Meine Rolex ist nicht angekommen."},
    {"role": "assistant", "content": "Der Käuferschutz sichert deine Zahlung ab. [1]"}]}


def test_handover_returns_briefing_validation_lines(tmp_path):
    response = make_handover_client(tmp_path).post("/api/handover", json=HANDOVER_BODY)
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["briefing"]["situation"]["text"] == "s"
    assert data["lines"][0]["id"] == "M01"
    assert "tokens" not in data


def test_handover_accepts_assistant_as_last_message(tmp_path):
    # Kein "letzte Nachricht muss user sein" — Übergabe nach Bot-Antwort ist Normalfall.
    response = make_handover_client(tmp_path).post("/api/handover", json=HANDOVER_BODY)
    assert response.status_code == 200


def test_handover_spends_budget(tmp_path):
    budget = TokenBudget(tmp_path / "h2.sqlite3", daily_limit=1000)
    make_handover_client(tmp_path, budget=budget).post("/api/handover", json=HANDOVER_BODY)
    assert budget.used_today() == 150


def test_handover_rejects_when_budget_empty(tmp_path):
    budget = TokenBudget(tmp_path / "h3.sqlite3", daily_limit=10)
    budget.spend(10)
    response = make_handover_client(tmp_path, budget=budget).post(
        "/api/handover", json=HANDOVER_BODY)
    assert response.status_code == 429


def test_handover_extract_failure_returns_502(tmp_path):
    async def broken_fn(messages, client):
        raise ValueError("kaputtes JSON")

    response = make_handover_client(tmp_path, handover_fn=broken_fn).post(
        "/api/handover", json=HANDOVER_BODY)
    assert response.status_code == 502
    assert "Briefing" in response.json()["detail"]


def test_handover_rate_limits_after_three_per_minute(tmp_path):
    budget = TokenBudget(tmp_path / "h4.sqlite3", daily_limit=1_000_000)
    client = make_handover_client(tmp_path, budget=budget)
    for _ in range(3):
        assert client.post("/api/handover", json=HANDOVER_BODY).status_code == 200
    assert client.post("/api/handover", json=HANDOVER_BODY).status_code == 429
```

- [ ] **Step 2: Fail sehen**

Run: `.venv\Scripts\python.exe -m pytest tests/test_api.py -q`
Expected: `TypeError: create_app() got an unexpected keyword argument 'handover_fn'`.

- [ ] **Step 3: Implementieren** — in `app/main.py`:

Import erweitern: `from app import faithcheck, handover, llm`.

`create_app`-Signatur: `def create_app(retriever=None, budget=None, answer_fn=None, rewrite_fn=None, llm_client=None, handover_fn=None) -> FastAPI:` und nach `app.state.rewrite_fn = …`: `app.state.handover_fn = handover_fn or handover.generate_briefing`.

Nach der `ChatRequest`-Klasse:

```python
class HandoverRequest(BaseModel):
    messages: list[ChatMessage] = Field(min_length=1, max_length=MAX_HISTORY_MESSAGES)
```

Nach dem Chat-Endpoint (vor dem StaticFiles-Mount):

```python
    @app.post("/api/handover")
    @limiter.limit("3/minute;10/day")
    async def handover_endpoint(request: Request, body: HandoverRequest) -> dict:
        if app.state.budget.remaining() <= 0:
            raise HTTPException(status_code=429, detail="Demo-Budget für heute erschöpft")
        client = app.state.llm_client or llm.get_client()
        try:
            result = await app.state.handover_fn(
                [m.model_dump() for m in body.messages], client)
        except Exception:
            logger.exception("Handover fehlgeschlagen")
            raise HTTPException(
                status_code=502,
                detail="Briefing-Erstellung fehlgeschlagen — bitte erneut versuchen.")
        app.state.budget.spend(result["tokens"])
        payload = {
            "status": result["status"],
            "briefing": result["briefing"],
            "validation": [{"text": c.text, "status": c.status,
                            "score": c.score, "sources": c.sources}
                           for c in result["validation"]],
            "lines": result["lines"],
        }
        if result["status"] == "rejected":
            payload["failed_claims"] = result["failed_claims"]
        return payload
```

(`result["validation"]` ist im Test-Fake eine leere Liste, im Echtbetrieb `SentenceCheck`-Objekte — die Comprehension verträgt beides.)

- [ ] **Step 4: Grün sehen**

Run: `.venv\Scripts\python.exe -m pytest tests/ -q` → komplette Suite grün.

- [ ] **Step 5: Commit**

```bash
git add app/main.py tests/test_api.py
git commit -m "feat: POST /api/handover mit Guards, Budget und ehrlichem Fehlerpfad"
```

---

### Task 5: Frontend — Button, Auto-Angebot, Briefing-Karte

**Files:**
- Modify: `static/index.html` (Button in `.input-row`), `static/app.js`, `static/style.css`

**Interfaces:**
- Consumes: Response-JSON aus Task 4; bestehende Helfer `addMessage`, `STATUS_ICON`, `history`-Array; NOT-FOUND-Text "Dazu finde ich nichts in den Chrono24-Hilfeseiten."
- Produces: UI-Verhalten; keine JS-Unit-Tests (Projekt hat keine JS-Test-Infrastruktur; Backend-Verhalten ist durch API-Tests gedeckt).

- [ ] **Step 1: Button in `static/index.html`** — in der `.input-row` neben `#send`:

```html
<button id="handover" type="button" hidden>An Support übergeben</button>
```

- [ ] **Step 2: Logik in `static/app.js`** — nach `addValidationDetails` einfügen:

```javascript
const handoverBtn = document.getElementById("handover");
const NOT_FOUND_TEXT = "Dazu finde ich nichts in den Chrono24-Hilfeseiten.";
const STATUS_LABEL = { ok: "✅ geprüft", rejected: "⛔ abgelehnt" };

function lineTooltip(lines, ids) {
  const byId = new Map(lines.map((l) => [l.id, `${l.actor}: ${l.text}`]));
  return ids.map((id) => `${id} — ${byId.get(id) || "?"}`).join("\n");
}

function briefingRow(label, text, check, lines) {
  const row = document.createElement("div");
  row.className = "briefing-row";
  const icon = check ? STATUS_ICON[check.status] : "";
  const strong = document.createElement("strong");
  strong.textContent = label + ": ";
  row.append(`${icon} `, strong, text);
  if (check && check.sources.length) {
    const ids = document.createElement("span");
    ids.className = "line-ids";
    ids.textContent = ` [${check.sources.join(", ")}]`;
    ids.title = lineTooltip(lines, check.sources);
    row.appendChild(ids);
  }
  return row;
}

function addBriefingCard(result) {
  const card = document.createElement("div");
  card.className = "briefing";
  const head = document.createElement("div");
  head.className = "briefing-head";
  head.textContent = `Übergabe-Briefing · ${STATUS_LABEL[result.status]}`;
  card.appendChild(head);

  if (result.status === "rejected") {
    const p = document.createElement("p");
    p.textContent =
      `Briefing nicht belegbar — der Validator hat ${result.failed_claims.length} ` +
      "Aussage(n) abgelehnt. Der Roh-Verlauf würde übergeben. " +
      "(Demo: es findet keine echte Weiterleitung statt.)";
    card.appendChild(p);
    messagesEl.appendChild(card);
    return;
  }

  // validation-Reihenfolge = situation, history, sentiment-quote, open_question, claims
  const v = result.validation;
  const b = result.briefing;
  card.appendChild(briefingRow("Situation", b.situation.text, v[0], result.lines));
  card.appendChild(briefingRow("Verlauf", b.history.text, v[1], result.lines));
  card.appendChild(briefingRow("Stimmung",
    `${b.sentiment.label} — „${b.sentiment.quote}"`, v[2], result.lines));
  card.appendChild(briefingRow("Offene Frage", b.open_question.text, v[3], result.lines));
  b.claims.forEach((claim, i) => {
    card.appendChild(briefingRow("Aussage", claim.text, v[4 + i], result.lines));
  });
  const legend = document.createElement("p");
  legend.className = "legend";
  legend.textContent =
    "✅ Wortlaut deckt sich mit dem Chat · 🟡 paraphrasiert · 🔴 nicht belegt · (Demo: keine echte Weiterleitung)";
  card.appendChild(legend);
  messagesEl.appendChild(card);
  messagesEl.scrollTop = messagesEl.scrollHeight;
}

async function requestHandover() {
  handoverBtn.disabled = true;
  handoverBtn.textContent = "Übergebe …";
  try {
    const response = await fetch("/api/handover", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ messages: history.slice(-20) }),
    });
    if (!response.ok) {
      const body = await response.json().catch(() => ({}));
      addMessage("bot", body.detail || "Übergabe gerade nicht möglich — bitte später erneut versuchen.");
      return;
    }
    addBriefingCard(await response.json());
  } catch {
    addMessage("bot", "Verbindungsfehler bei der Übergabe — bitte gleich nochmal versuchen.");
  } finally {
    handoverBtn.disabled = false;
    handoverBtn.textContent = "An Support übergeben";
  }
}

handoverBtn.addEventListener("click", requestHandover);

function offerHandover() {
  const div = document.createElement("div");
  div.className = "handover-offer";
  div.append("Der Bot weiß hier nicht weiter — ");
  const link = document.createElement("button");
  link.type = "button";
  link.className = "handover-link";
  link.textContent = "an einen Menschen übergeben?";
  link.addEventListener("click", requestHandover);
  div.appendChild(link);
  messagesEl.appendChild(div);
}
```

In `ask()` die Zeile `if (answer) history.push({ role: "assistant", content: answer });` ersetzen durch:

```javascript
    if (answer) {
      history.push({ role: "assistant", content: answer });
      handoverBtn.hidden = false;
      if (answer.trim().endsWith(NOT_FOUND_TEXT)) offerHandover();
    }
```

- [ ] **Step 3: CSS in `static/style.css`** — nach dem `.validation`-Block:

```css
.briefing { font-size: 0.85rem; background: var(--panel); border: 1px solid var(--muted);
  border-radius: 10px; padding: 0.6rem 0.9rem; }
.briefing-head { font-weight: 600; margin-bottom: 0.4rem; }
.briefing-row { margin-bottom: 0.3rem; }
.briefing .line-ids { color: var(--accent); cursor: help; }
.briefing .legend { margin-top: 0.4rem; font-size: 0.75rem; color: var(--muted); }
.handover-offer { font-size: 0.85rem; color: var(--muted); }
.handover-link { background: none; border: none; color: var(--accent);
  cursor: pointer; font-size: inherit; padding: 0; text-decoration: underline; }
#handover { background: var(--panel); color: var(--text); border: 1px solid var(--muted);
  border-radius: 8px; padding: 0.6rem 0.9rem; cursor: pointer; }
#handover:hover { border-color: var(--accent); }
#handover:disabled { opacity: 0.5; }
```

- [ ] **Step 4: Suite grün**

Run: `.venv\Scripts\python.exe -m pytest tests/ -q` → grün (Frontend bricht nichts serverseitig).

- [ ] **Step 5: Commit**

```bash
git add static/index.html static/app.js static/style.css
git commit -m "feat: Handover-Button, Auto-Angebot und Briefing-Karte im Chat"
```

---

### Task 6: README-Sektion

**Files:**
- Modify: `README.md` (nach der Sektion "Laufzeit-Faithfulness-Check (deterministisch)", vor "## Scraping-Ethik")

**Interfaces:** keine — Doku.

- [ ] **Step 1: Sektion einfügen**

```markdown
### Übergabe an Menschen (Handover-Briefing)

Wenn der Bot nicht weiterweiß — oder auf Knopfdruck — übergibt er den
Chatverlauf an einen menschlichen Support-Agenten: `POST /api/handover`
lässt Claude ein strukturiertes Briefing extrahieren
(Situation/Verlauf/Stimmung/offene Frage/Claims, jede Aussage mit
Zeilen-Zitaten `M01…`), und derselbe deterministische Validator prüft jede
Aussage per Token-Overlap gegen die zitierten Chat-Zeilen. Unbelegte
Aussagen führen zu genau einem Retry mit Fehlerhinweis; scheitert auch
der, wird das Briefing **abgelehnt** statt still ausgeliefert — die
Demo-Karte sagt das offen. Der Handover-Call läuft über dieselben Guards
(Rate-Limit 3/min, Tages-Token-Budget). Damit ist die Produktstory
komplett: der Bot beantwortet, was er belegen kann; was nicht, übergibt
er an einen Menschen — mit geprüftem Briefing.
```

- [ ] **Step 2: Suite grün**

Run: `.venv\Scripts\python.exe -m pytest tests/ -q` → grün.

- [ ] **Step 3: Commit**

```bash
git add README.md
git commit -m "docs: Handover-Briefing im README"
```
