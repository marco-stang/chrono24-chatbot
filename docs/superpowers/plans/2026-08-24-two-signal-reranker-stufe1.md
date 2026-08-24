# Zwei-Signal-LLM-Reranker — Stufe 1 (Implementation Plan)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Zwei-Signal-LLM-Reranker (Rangfolge + `top1_confidence`) bauen und
isoliert an den 4 bekannten Problemfällen plus einer Off-Topic-Stichprobe
testen — billig, bevor der volle Eval-Lauf (Stufe 2, eigener Plan) folgt.

**Architecture:** Neues Modul `eval/llm_reranker.py` kapselt Prompt-Bau,
Response-Parsing (mit konservativem Fallback) und die Kandidaten-Union.
Ein neues Skript `eval/run_llm_reranker_experiment.py` verkabelt das mit
einer echten `Retriever`-Instanz (deren private Helfer `_vector_candidates`
/`_bm25_candidates`/`_to_doc` wiederverwendet werden) und dem echten
Anthropic-Client. `app/retrieval.py` bleibt unverändert — das Experiment
bleibt reversibel, wie in der Spec festgelegt.

**Tech Stack:** Python 3, pytest (asyncio_mode=auto, keine Decorators
nötig), Anthropic SDK (`AsyncAnthropic`, bereits in `app/llm.py` verkabelt),
bestehende SQLite-Retrieval-Engine (`app/retrieval.py`).

**Spec:** `docs/superpowers/specs/2026-08-24-two-signal-reranker-design.md`

## Global Constraints

- Output-Format des LLM-Calls exakt `{"ranking": [...], "top1_confidence": 0-10}`
  (Spec, Abschnitt „Architektur / Datenfluss").
- `top1_confidence` bewertet ausschließlich den nach `ranking` bestplatzierten
  Kandidaten, nie die gesamte Kandidatenmenge (Spec, selber Abschnitt).
- Bei fehlendem oder außerhalb 0–10 liegendem `top1_confidence`: konservativ
  auf `0.0` fallen (Spec, Abschnitt „Fehlerbehandlung") — nie crashen.
- `max_tokens=400`, `temperature=0`, exakte Kandidatenzahl + Beispiel-Array
  im System-Prompt (Spec, Abschnitt „Fehlerbehandlung", übernommen aus dem
  Vorexperiment).
- `app/retrieval.py` wird in diesem Plan **nicht** verändert (Spec, Abschnitt
  „Reversibilität").
- Scope ist Stufe 1 (Handover-Schritt 1: die 4 Testfälle + Off-Topic-
  Stichprobe). Schwellen-Rekalibrierung und voller Eval-Lauf (Stufe 2) sind
  **nicht** Teil dieses Plans.

---

### Task 1: Prompt-Bau und Response-Parsing (reine Funktionen)

**Files:**
- Create: `eval/llm_reranker.py`
- Test: `tests/test_llm_reranker.py`

**Interfaces:**
- Produces: `_system_prompt(n: int) -> str`, `build_llm_rerank_prompt(query: str, docs: list[RetrievedDoc]) -> str`, `_parse_response(text: str, n: int) -> tuple[list[int], float]`, `union_candidates(vector_ranking: list[str], bm25_ranking: list[str]) -> list[str]` — Task 2 und Task 3 rufen diese vier auf.

- [ ] **Step 1: Failing Tests schreiben**

```python
# tests/test_llm_reranker.py
from app.retrieval import RetrievedDoc
from eval.llm_reranker import (
    _parse_response,
    _system_prompt,
    build_llm_rerank_prompt,
    union_candidates,
)

DOCS = [
    RetrievedDoc(id="faq-0001", type="faq", title="Frage A",
                 url="https://www.chrono24.de/info/faqs.htm", text="Antwort A", score=0.1),
    RetrievedDoc(id="faq-0002", type="faq", title="Frage B",
                 url="https://www.chrono24.de/info/faqs.htm", text="Antwort B", score=0.1),
]


def test_system_prompt_names_exact_candidate_count():
    prompt = _system_prompt(3)
    assert "genau 3 nummerierte" in prompt
    assert "0 bis 2" in prompt


def test_system_prompt_includes_full_example_array():
    prompt = _system_prompt(3)
    assert '"ranking": [2, 1, 0]' in prompt


def test_build_llm_rerank_prompt_numbers_candidates_from_zero():
    prompt = build_llm_rerank_prompt("Wie funktioniert der Käuferschutz?", DOCS)
    assert "[0] Frage A" in prompt
    assert "[1] Frage B" in prompt
    assert "Antwort A" in prompt
    assert "Wie funktioniert der Käuferschutz?" in prompt


def test_parse_response_accepts_valid_json():
    ranking, confidence = _parse_response('{"ranking": [1, 0], "top1_confidence": 8}', n=2)
    assert ranking == [1, 0]
    assert confidence == 8.0


def test_parse_response_falls_back_to_identity_ranking_on_incomplete_array():
    ranking, _ = _parse_response('{"ranking": [0], "top1_confidence": 7}', n=2)
    assert ranking == [0, 1]


def test_parse_response_falls_back_to_zero_confidence_when_missing():
    ranking, confidence = _parse_response('{"ranking": [1, 0]}', n=2)
    assert confidence == 0.0
    assert ranking == [1, 0]


def test_parse_response_falls_back_to_zero_confidence_when_out_of_range():
    _, confidence = _parse_response('{"ranking": [1, 0], "top1_confidence": 15}', n=2)
    assert confidence == 0.0


def test_parse_response_falls_back_to_zero_confidence_on_boolean_value():
    # bool ist in Python eine int-Subklasse -- ohne expliziten Ausschluss
    # würde True fälschlich zu 1.0 statt zum konservativen Fallback.
    _, confidence = _parse_response('{"ranking": [1, 0], "top1_confidence": true}', n=2)
    assert confidence == 0.0


def test_parse_response_falls_back_completely_on_malformed_json():
    ranking, confidence = _parse_response("not json at all", n=2)
    assert ranking == [0, 1]
    assert confidence == 0.0


def test_union_candidates_dedupes_keeping_first_occurrence():
    result = union_candidates(["a", "b"], ["b", "c"])
    assert result == ["a", "b", "c"]
```

- [ ] **Step 2: Tests laufen lassen, müssen fehlschlagen**

Run: `pytest tests/test_llm_reranker.py -v`
Expected: FAIL mit `ModuleNotFoundError: No module named 'eval.llm_reranker'`

- [ ] **Step 3: `eval/llm_reranker.py` implementieren**

```python
"""Zwei-Signal-LLM-Reranker (Rangfolge + top1_confidence) -- Experiment,
noch nicht integriert. Siehe
docs/superpowers/specs/2026-08-24-two-signal-reranker-design.md.
app/retrieval.py bleibt unveraendert, solange nur gemessen wird."""
from __future__ import annotations

import json

from app.retrieval import RetrievedDoc, _dedupe_ranking

MAX_LLM_RERANK_TOKENS = 400


def _system_prompt(n: int) -> str:
    example_ranking = list(range(n - 1, -1, -1))
    example_json = '{"ranking": [' + ", ".join(str(i) for i in example_ranking) + '], "top1_confidence": 0}'
    return (
        "Du bist ein Reranker fuer Chrono24-Hilfeseiten. Du bekommst eine "
        f"Nutzerfrage und genau {n} nummerierte Kandidaten-Dokumente (Index "
        f"0 bis {n - 1}). Sortiere die Kandidaten nach Relevanz fuer die "
        "Frage, absteigend -- der am besten passende Kandidat zuerst. "
        "Antworte NUR mit einem JSON-Objekt, keine Erklaerung, kein "
        f"Markdown drumherum, Beispiel-Format: {example_json}\n\n"
        f'"ranking" muss alle {n} Indizes genau einmal enthalten, in deiner '
        'Rangfolge. "top1_confidence" bewertet NUR den an Position 0 deiner '
        "Rangfolge platzierten Kandidaten, unabhaengig von den anderen: "
        "0 = beantwortet die Frage ueberhaupt nicht, auch nicht ansatzweise; "
        "10 = beantwortet die Frage vollstaendig und eindeutig. Werte "
        "dazwischen nach eigenem Ermessen."
    )


def build_llm_rerank_prompt(query: str, docs: list[RetrievedDoc]) -> str:
    candidates = "\n\n".join(f"[{i}] {doc.title}\n{doc.text}" for i, doc in enumerate(docs))
    return f"Frage: {query}\n\nKandidaten:\n{candidates}"


def _parse_response(text: str, n: int) -> tuple[list[int], float]:
    """Konservativer Fallback bei jedem Abweichen vom erwarteten Format --
    ein Malformed-Value darf nie zu einem falsch-positiven Treffer fuehren
    (Spec, Abschnitt Fehlerbehandlung)."""
    identity = list(range(n))
    try:
        data = json.loads(text)
    except (json.JSONDecodeError, TypeError):
        return identity, 0.0
    if not isinstance(data, dict):
        return identity, 0.0

    ranking = data.get("ranking")
    if not isinstance(ranking, list) or sorted(ranking) != identity:
        ranking = identity

    confidence = data.get("top1_confidence")
    is_plain_number = isinstance(confidence, (int, float)) and not isinstance(confidence, bool)
    if not is_plain_number or not (0 <= confidence <= 10):
        confidence = 0.0

    return ranking, float(confidence)


def union_candidates(vector_ranking: list[str], bm25_ranking: list[str]) -> list[str]:
    """Vereinigung statt RRF-Top-n-Cut -- Kandidaten-Union-Fix aus der Spec:
    ohne ihn sieht kein Reranker Kandidaten, die nur in einer der beiden
    Top-10-Listen weit vorn liegen."""
    return _dedupe_ranking(vector_ranking + bm25_ranking)
```

- [ ] **Step 4: Tests laufen lassen, müssen passen**

Run: `pytest tests/test_llm_reranker.py -v`
Expected: PASS (10 Tests)

- [ ] **Step 5: Commit**

```bash
git add eval/llm_reranker.py tests/test_llm_reranker.py
git commit -m "feat: Prompt-Bau und Response-Parsing fuer Zwei-Signal-Reranker"
```

---

### Task 2: Asynchroner LLM-Call

**Files:**
- Modify: `eval/llm_reranker.py`
- Test: `tests/test_llm_reranker.py`

**Interfaces:**
- Consumes: `_system_prompt`, `build_llm_rerank_prompt`, `_parse_response`, `MAX_LLM_RERANK_TOKENS` (Task 1, selbe Datei).
- Produces: `async def llm_two_signal_rerank(query: str, docs: list[RetrievedDoc], client) -> tuple[list[int], float]` — Task 4 (Experiment-Skript) ruft das auf.

- [ ] **Step 1: Failing Test schreiben**

```python
# an tests/test_llm_reranker.py anhängen
from eval.llm_reranker import llm_two_signal_rerank


class _FakeTextBlock:
    type = "text"

    def __init__(self, text):
        self.text = text


class _FakeResponse:
    def __init__(self, text):
        self.content = [_FakeTextBlock(text)]


class _FakeMessages:
    def __init__(self, response_text):
        self._response_text = response_text
        self.calls = []

    async def create(self, **kwargs):
        self.calls.append(kwargs)
        return _FakeResponse(self._response_text)


class _FakeClient:
    def __init__(self, response_text):
        self.messages = _FakeMessages(response_text)


async def test_llm_two_signal_rerank_returns_parsed_ranking_and_confidence():
    client = _FakeClient('{"ranking": [1, 0], "top1_confidence": 9}')
    ranking, confidence = await llm_two_signal_rerank("Frage?", DOCS, client)
    assert ranking == [1, 0]
    assert confidence == 9.0


async def test_llm_two_signal_rerank_pins_temperature_and_token_limit():
    client = _FakeClient('{"ranking": [0, 1], "top1_confidence": 5}')
    await llm_two_signal_rerank("Frage?", DOCS, client)
    call = client.messages.calls[0]
    assert call["temperature"] == 0
    assert call["max_tokens"] == 400


async def test_llm_two_signal_rerank_falls_back_on_malformed_response():
    client = _FakeClient("kaputte Antwort, kein JSON")
    ranking, confidence = await llm_two_signal_rerank("Frage?", DOCS, client)
    assert ranking == [0, 1]
    assert confidence == 0.0
```

- [ ] **Step 2: Test laufen lassen, muss fehlschlagen**

Run: `pytest tests/test_llm_reranker.py -v -k llm_two_signal_rerank`
Expected: FAIL mit `ImportError: cannot import name 'llm_two_signal_rerank'`

- [ ] **Step 3: `llm_two_signal_rerank` implementieren**

```python
# an eval/llm_reranker.py anhängen
from app.config import settings


async def llm_two_signal_rerank(
    query: str, docs: list[RetrievedDoc], client
) -> tuple[list[int], float]:
    """Gibt (Rangfolge als 0-indexierte Positionsliste, top1_confidence)
    zurueck. ranking[0] ist der Index des nach Rangfolge bestplatzierten
    Kandidaten in docs."""
    response = await client.messages.create(
        model=settings.model,
        max_tokens=MAX_LLM_RERANK_TOKENS,
        system=_system_prompt(len(docs)),
        temperature=0,
        messages=[{"role": "user", "content": build_llm_rerank_prompt(query, docs)}],
    )
    text = next((b.text for b in response.content if b.type == "text"), "").strip()
    return _parse_response(text, len(docs))
```

- [ ] **Step 4: Tests laufen lassen, müssen passen**

Run: `pytest tests/test_llm_reranker.py -v`
Expected: PASS (13 Tests)

- [ ] **Step 5: Commit**

```bash
git add eval/llm_reranker.py tests/test_llm_reranker.py
git commit -m "feat: asynchronen LLM-Call fuer Zwei-Signal-Reranker verkabeln"
```

---

### Task 3: Retriever-Wrapper (Kandidaten-Union-Fix wiederverwenden)

**Files:**
- Modify: `eval/llm_reranker.py`
- Test: `tests/test_llm_reranker.py`

**Interfaces:**
- Consumes: `union_candidates` (Task 1), `Retriever`, `RetrievedDoc`, `TOP_K_CANDIDATES` aus `app/retrieval.py` (bestehend, `Retriever._vector_candidates`/`_bm25_candidates`/`_to_doc`, `.db`, `.sim_threshold`, `.bm25_threshold` sind bereits vorhanden, siehe `app/retrieval.py:121-197`).
- Produces: `def two_signal_candidates(retriever: Retriever, query: str, audience: str | None) -> tuple[list[RetrievedDoc], bool]` — Task 4 ruft das auf. Zweiter Rückgabewert `False` heißt: Stufe-1-Gate (sim/bm25) hat schon abgelehnt, `docs` ist dann `[]`.

- [ ] **Step 1: Failing Test schreiben**

```python
# an tests/test_llm_reranker.py anhängen
import json

from pipeline.index import build_index
from app.retrieval import Retriever
from eval.llm_reranker import two_signal_candidates

_CORPUS_DOCS = [
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

_DOC_VECS = {"Käuferschutz": [1.0, 0.0, 0.0], "verkaufe": [0.0, 1.0, 0.0],
             "Versand": [0.0, 0.0, 1.0]}


def _encode_one(text):
    for key, vec in _DOC_VECS.items():
        if key in text:
            return vec
    return [-1.0, 0.0, 0.0]


def _build_retriever(tmp_path):
    corpus_path = tmp_path / "corpus.json"
    corpus_path.write_text(json.dumps({"scraped_at": "2026-08-24", "documents": _CORPUS_DOCS}),
                           encoding="utf-8")
    index_dir = tmp_path / "index"
    build_index(corpus_path, index_dir, encoder=lambda texts: [_encode_one(t) for t in texts])
    return Retriever(index_dir, corpus_path, encoder=_encode_one, reranker=False,
                     bm25_threshold=1.0)


def test_two_signal_candidates_returns_union_when_gate_open(tmp_path):
    retriever = _build_retriever(tmp_path)
    docs, gate_open = two_signal_candidates(retriever, "Wie funktioniert der Käuferschutz?", None)
    assert gate_open is True
    assert {d.id for d in docs} == {"faq-0001", "faq-0002", "info-shipping-0001"}


def test_two_signal_candidates_closes_gate_for_offtopic(tmp_path):
    retriever = _build_retriever(tmp_path)
    docs, gate_open = two_signal_candidates(retriever, "Gedicht über Katzen bitte", None)
    assert gate_open is False
    assert docs == []
```

- [ ] **Step 2: Test laufen lassen, muss fehlschlagen**

Run: `pytest tests/test_llm_reranker.py -v -k two_signal_candidates`
Expected: FAIL mit `ImportError: cannot import name 'two_signal_candidates'`

- [ ] **Step 3: `two_signal_candidates` implementieren**

```python
# an eval/llm_reranker.py anhängen
from app.retrieval import TOP_K_CANDIDATES, Retriever


def two_signal_candidates(
    retriever: Retriever, query: str, audience: str | None
) -> tuple[list[RetrievedDoc], bool]:
    """Baut die Kandidatenmenge wie Retriever.retrieve(), aber als
    Vereinigung statt RRF-Top-n-Cut (Kandidaten-Union-Fix, siehe Spec).
    Zweiter Rueckgabewert: ob Stufe 1 des bestehenden Gates (sim/bm25-
    Schwelle) ueberhaupt Kandidaten durchlaesst."""
    total = retriever.db.execute("SELECT COUNT(*) FROM vectors").fetchone()[0]
    n = min(TOP_K_CANDIDATES, total)
    vector_ranking, best_sim = retriever._vector_candidates(query, n, total, audience)
    bm25_ranking, best_bm25 = retriever._bm25_candidates(query, n, audience)
    if best_sim < retriever.sim_threshold or best_bm25 < retriever.bm25_threshold:
        return [], False
    ids = union_candidates(vector_ranking, bm25_ranking)
    docs = [retriever._to_doc(doc_id, 0.0) for doc_id in ids]
    return docs, True
```

- [ ] **Step 4: Tests laufen lassen, müssen passen**

Run: `pytest tests/test_llm_reranker.py -v`
Expected: PASS (15 Tests)

- [ ] **Step 5: Commit**

```bash
git add eval/llm_reranker.py tests/test_llm_reranker.py
git commit -m "feat: Retriever-Wrapper mit Kandidaten-Union-Fix fuer Zwei-Signal-Reranker"
```

---

### Task 4: Experiment-Skript Stufe 1

**Files:**
- Create: `eval/run_llm_reranker_experiment.py`
- Test: `tests/test_run_llm_reranker_experiment.py`

**Interfaces:**
- Consumes: `two_signal_candidates`, `llm_two_signal_rerank` (Task 2/3, `eval/llm_reranker.py`); `eval_query`, `QUESTIONS_PATH`, `OFFTOPIC_QUESTIONS_PATH` (bestehend, `eval/run_eval.py:9-11,39-47`); `classify_audience` (bestehend, `app/textproc.py:64`); `get_client` (bestehend, `app/llm.py:36-42`); `settings` (bestehend, `app/config.py`).
- Produces: `_known_miss_cases(questions: list[dict]) -> list[dict]`, `_offtopic_sample(questions: list[dict]) -> list[dict]` (reine Funktionen, testbar ohne DB/API), `async def main() -> None` (Einstiegspunkt, nicht unit-getestet — gleiches Muster wie der bestehende `__main__`-Block in `eval/run_eval.py`, der ebenfalls nur über seine reinen Helfer wie `check_gate` getestet wird, siehe `tests/test_run_eval.py`).

- [ ] **Step 1: Failing Tests schreiben**

```python
# tests/test_run_llm_reranker_experiment.py
from eval.run_llm_reranker_experiment import (
    KNOWN_MISS_IDS,
    OFFTOPIC_SAMPLE_INDICES,
    _known_miss_cases,
    _offtopic_sample,
)

_TUNING_QUESTIONS = [
    {"question": "Frage 1", "expected_doc_id": "faq-0001"},
    {"question": "Frage 2", "expected_doc_id": "faq-0098"},
    {"question": "Frage 3", "expected_doc_id": "faq-0033"},
]

_OFFTOPIC_QUESTIONS = [{"question": f"Off-Topic {i}"} for i in range(14)]


def test_known_miss_ids_matches_the_four_documented_problem_cases():
    assert KNOWN_MISS_IDS == {"faq-0098", "info-escrow-0007", "faq-0033", "faq-0162"}


def test_known_miss_cases_filters_only_documented_ids():
    cases = _known_miss_cases(_TUNING_QUESTIONS)
    assert {c["expected_doc_id"] for c in cases} == {"faq-0098", "faq-0033"}


def test_offtopic_sample_selects_configured_indices():
    sample = _offtopic_sample(_OFFTOPIC_QUESTIONS)
    assert sample == [_OFFTOPIC_QUESTIONS[i] for i in OFFTOPIC_SAMPLE_INDICES]
```

- [ ] **Step 2: Tests laufen lassen, müssen fehlschlagen**

Run: `pytest tests/test_run_llm_reranker_experiment.py -v`
Expected: FAIL mit `ModuleNotFoundError: No module named 'eval.run_llm_reranker_experiment'`

- [ ] **Step 3: `eval/run_llm_reranker_experiment.py` implementieren**

```python
"""Stufe 1 des Zwei-Signal-LLM-Reranker-Experiments: die 4 bekannten
Problemfaelle plus eine Off-Topic-Stichprobe, isoliert und billig, bevor
der volle Eval-Lauf (Stufe 2) folgt. Siehe
docs/superpowers/specs/2026-08-24-two-signal-reranker-design.md.

Kostet echte Anthropic-API-Calls -- nur manuell ausfuehren:
    python -m eval.run_llm_reranker_experiment
"""
from __future__ import annotations

import asyncio
import json

from app.config import settings
from app.llm import get_client
from app.retrieval import Retriever
from app.textproc import classify_audience
from eval.llm_reranker import llm_two_signal_rerank, two_signal_candidates
from eval.run_eval import OFFTOPIC_QUESTIONS_PATH, QUESTIONS_PATH, eval_query

KNOWN_MISS_IDS = {"faq-0098", "info-escrow-0007", "faq-0033", "faq-0162"}
# Indizes in eval/questions_offtopic.json: 0/1 klar themenfremd, 6/8
# absichtlich domaennah (Omega-Wert, eBay-Vergleich) -- genau die
# Fragetypen, die im Handover als Durchrutscher genannt sind.
OFFTOPIC_SAMPLE_INDICES = [0, 1, 6, 8]


def _known_miss_cases(questions: list[dict]) -> list[dict]:
    return [q for q in questions if q["expected_doc_id"] in KNOWN_MISS_IDS]


def _offtopic_sample(questions: list[dict]) -> list[dict]:
    return [questions[i] for i in OFFTOPIC_SAMPLE_INDICES]


async def _run_known_misses(retriever: Retriever, client) -> None:
    questions = json.loads(QUESTIONS_PATH.read_text(encoding="utf-8"))
    cases = _known_miss_cases(questions)
    print(f"-- {len(cases)} bekannte Problemfaelle --")
    for item in cases:
        query = eval_query(item)
        audience = classify_audience(query)
        docs, gate_open = two_signal_candidates(retriever, query, audience)
        if not gate_open:
            print(f"  GATE ZU (Stufe 1): {item['question']!r}")
            continue
        ranking, confidence = await llm_two_signal_rerank(query, docs, client)
        top1_id = docs[ranking[0]].id
        status = "OK" if top1_id == item["expected_doc_id"] else "MISS"
        print(f"  {status}: {item['question']!r} erwartet {item['expected_doc_id']}, "
              f"top1 {top1_id}, confidence {confidence}")


async def _run_offtopic_sample(retriever: Retriever, client) -> None:
    questions = json.loads(OFFTOPIC_QUESTIONS_PATH.read_text(encoding="utf-8"))
    sample = _offtopic_sample(questions)
    print(f"-- {len(sample)} Off-Topic-Stichprobe --")
    for item in sample:
        query = item["question"]
        audience = classify_audience(query)
        docs, gate_open = two_signal_candidates(retriever, query, audience)
        if not gate_open:
            print(f"  GATE ZU (Stufe 1, korrekt): {query!r}")
            continue
        ranking, confidence = await llm_two_signal_rerank(query, docs, client)
        top1_id = docs[ranking[0]].id
        print(f"  confidence {confidence}: {query!r} -> top1 {top1_id}")


async def main() -> None:
    retriever = Retriever(settings.index_dir, settings.corpus_path, reranker=False)
    client = get_client()
    await _run_known_misses(retriever, client)
    await _run_offtopic_sample(retriever, client)


if __name__ == "__main__":
    asyncio.run(main())
```

- [ ] **Step 4: Tests laufen lassen, müssen passen**

Run: `pytest tests/test_run_llm_reranker_experiment.py -v`
Expected: PASS (3 Tests)

- [ ] **Step 5: Vollen Testlauf des Repos gegenchecken**

Run: `pytest -q`
Expected: alle Tests grün (bisherige 163 + 18 neue aus diesem Plan)

- [ ] **Step 6: Commit**

```bash
git add eval/run_llm_reranker_experiment.py tests/test_run_llm_reranker_experiment.py
git commit -m "feat: Experiment-Skript fuer Zwei-Signal-Reranker Stufe 1"
```

---

### Task 5: Stufe 1 ausführen und Ergebnis interpretieren

Kein Code — das eigentliche Ziel dieses Plans: die 4 Problemfälle und die
Off-Topic-Stichprobe gegen den echten Haiku-Call laufen lassen, bevor über
Stufe 2 (voller Eval, Schwellen-Rekalibrierung) entschieden wird.

- [ ] **Step 1: Skript ausführen (kostet echte API-Calls)**

Run: `python -m eval.run_llm_reranker_experiment`

- [ ] **Step 2: Ausgabe prüfen**

Erwartung laut Spec:
- Alle 4 `OK` (Rangfolge löst laut Vorexperiment bereits alle drei Misses
  — vierter Fall info-escrow-0007 war der durch den Kandidaten-Union-Fix
  gelöste).
- `confidence` bei den 4 Fällen deutlich höher als bei der Off-Topic-
  Stichprobe (0/1 klar themenfremd, 6/8 domänennah) — das ist die
  Kernannahme, die Stufe 2 (Schwellen-Rekalibrierung) erst sinnvoll macht.
  Überlappen sich die Werte stark, ist das Zwei-Signal-Design selbst
  fraglich und Stufe 2 sollte nicht ungeprüft folgen.

- [ ] **Step 3: Ergebnis im Handover festhalten**

`HANDOVER-llm-reranker.md` um die Stufe-1-Ergebnisse ergänzen (neuer
Abschnitt, Rohwerte aus Step 2), bevor Stufe 2 (eigener Plan) beginnt.
