# LLM-Reranker-Integration (Ersatz mit Rollback-Flag) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Den in Stufe 1/2 gemessenen Zwei-Signal-LLM-Reranker als
Standardverhalten in Produktion schalten (`app/retrieval.py`,
`app/main.py`), mit Config-Flag-Rollback auf den bisherigen Cross-Encoder.

**Architecture:** `Retriever.retrieve()` wird `async def` und verzweigt
auf `self.use_llm_reranker` (Default aus `settings.use_llm_reranker`):
LLM-Pfad nutzt Kandidaten-Union statt RRF-Cut, Cross-Encoder-Pfad bleibt
exakt wie bisher. Die Reranker-Mechanik (Prompt, Parsing, `union_candidates`,
`llm_two_signal_rerank`) wandert von `eval/llm_reranker.py` nach
`app/llm_reranker.py`, damit `app/retrieval.py` sie importieren kann, ohne
dass `app` von `eval` abhängt.

**Tech Stack:** Python 3, pytest (`asyncio_mode=auto`), Anthropic SDK
(`AsyncAnthropic`), FastAPI/SSE, bestehende SQLite-Retrieval-Engine.

**Spec:** `docs/superpowers/specs/2026-08-24-llm-reranker-integration-design.md`

## Global Constraints

- `settings.use_llm_reranker: bool = True` — neuer Standard, per Env-Var
  `USE_LLM_RERANKER=false` auf Cross-Encoder zurückschaltbar.
- `Retriever.retrieve()` wird für ALLE Aufrufer `async def` — kein
  synchroner Aufrufpfad bleibt bestehen.
- `LLM_CONFIDENCE_THRESHOLD = 8.5` (Stufe-2-Messung: on-topic-Minimum 9.0,
  Puffer 0.5), definiert in `app/retrieval.py` neben den bestehenden
  Schwellen-Konstanten.
- `llm_two_signal_rerank` gibt ein 4-Tupel zurück:
  `(ranking, confidence, used_fallback, tokens)`.
- `app/llm_reranker.py` darf zur Laufzeit **nichts** aus `app/retrieval.py`
  importieren (zirkulärer Import) — `RetrievedDoc` nur hinter
  `TYPE_CHECKING`, `union_candidates` dedupliziert inline statt
  `_dedupe_ranking` zu importieren.
- `Retriever.__init__` lädt den Cross-Encoder (`_default_reranker()`) nur,
  wenn `reranker` nicht explizit übergeben wurde UND
  `self.use_llm_reranker` `False` ist.
- `daily_token_budget` bleibt unverändert bei 200.000.
- Bestehende Cross-Encoder-Mock-Tests (`reranker=<mock>`) bekommen
  zusätzlich `use_llm_reranker=False`, um weiterhin den alten Pfad zu
  testen — sonst würden sie beim neuen globalen Default (`True`)
  stillschweigend auf den LLM-Pfad wechseln.

---

### Task 1: Config-Flag

**Files:**
- Modify: `app/config.py`

**Interfaces:**
- Produces: `settings.use_llm_reranker: bool` — Task 3 (Retriever) konsumiert das als Default für seinen Konstruktor-Parameter.

- [ ] **Step 1: Feld ergänzen**

In `app/config.py`, nach der Zeile `rerank_model: str = "VoidFloat/chrono24-faq-reranker"`:

```python
    # Ersatz statt Cross-Encoder als Standard-Reranker (siehe Handover +
    # Spec docs/superpowers/specs/2026-08-24-llm-reranker-integration-design.md).
    # Rollback ohne Deploy: USE_LLM_RERANKER=false setzen und neu starten.
    use_llm_reranker: bool = True
```

- [ ] **Step 2: Import-Test laufen lassen**

Run: `c:/Users/Marco/02_Portfolio/chrono24-chatbot/.venv/Scripts/python.exe -c "from app.config import settings; assert settings.use_llm_reranker is True; print('ok')"`
Expected: `ok`

- [ ] **Step 3: Commit**

```bash
git add app/config.py
git commit -m "feat: Config-Flag fuer LLM-Reranker-Rollback"
```

---

### Task 2: Reranker-Modul nach `app/` verschieben

**Files:**
- Create: `app/llm_reranker.py`
- Delete: `eval/llm_reranker.py`
- Modify: `tests/test_llm_reranker.py`

**Interfaces:**
- Consumes: nichts von `app/retrieval.py` zur Laufzeit (nur `TYPE_CHECKING`-Import von `RetrievedDoc`).
- Produces: `MAX_LLM_RERANK_TOKENS`, `_system_prompt(n)`, `build_llm_rerank_prompt(query, docs)`, `_parse_response(text, n) -> tuple[list[int], float, bool]`, `union_candidates(vector_ranking, bm25_ranking) -> list[str]`, `async def llm_two_signal_rerank(query, docs, client) -> tuple[list[int], float, bool, int]` — Task 3 (`app/retrieval.py`) importiert `llm_two_signal_rerank` und `union_candidates`. Task 8 (Stufe-1/2-Skripte) importiert alle vier öffentlichen Namen.

- [ ] **Step 1: `app/llm_reranker.py` mit vollständigem Inhalt anlegen**

```python
"""Zwei-Signal-LLM-Reranker (Rangfolge + top1_confidence) fuer
Retriever.retrieve(). Siehe
docs/superpowers/specs/2026-08-24-llm-reranker-integration-design.md."""
from __future__ import annotations

import json
import re
from typing import TYPE_CHECKING

from app.config import settings

if TYPE_CHECKING:
    from app.retrieval import RetrievedDoc

MAX_LLM_RERANK_TOKENS = 400

_FENCE_RE = re.compile(r"^```(?:json)?\s*\n?(.*?)\n?```$", re.DOTALL)


def _strip_code_fence(text: str) -> str:
    """Strips markdown code fence (```json or ```) from LLM response if present.

    Handles variants: ```json\n{...}\n``` and ```\n{...}\n```
    If no fence is present, returns the text unchanged.
    """
    match = _FENCE_RE.match(text.strip())
    return match.group(1) if match else text


def _system_prompt(n: int) -> str:
    example_ranking = list(range(n - 1, -1, -1))
    example_json = '{"ranking": [' + ", ".join(str(i) for i in example_ranking) + '], "top1_confidence": 0}'
    return (
        "Du bist ein Reranker fuer Chrono24-Hilfeseiten. Du bekommst eine "
        f"Nutzerfrage und genau {n} nummerierte Kandidaten-Dokumente (Index "
        f"0 bis {n - 1}). Sortiere die Kandidaten nach Relevanz fuer die "
        "Frage, absteigend -- der am besten passende Kandidat zuerst. "
        "Antworte NUR mit einem JSON-Objekt, keine Erklaerung, kein "
        "Markdown drumherum, Beispiel-Format: " + example_json + "\n\n"
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


def _parse_response(text: str, n: int) -> tuple[list[int], float, bool]:
    """Konservativer Fallback bei jedem Abweichen vom erwarteten Format --
    ein Malformed-Value darf nie zu einem falsch-positiven Treffer fuehren.
    Dritter Rueckgabewert `used_fallback`: True, wenn ranking und/oder
    confidence nicht valide geparst werden konnten -- sonst waere eine
    echte confidence von 0.0 nicht von einem Parse-Fehler zu
    unterscheiden."""
    identity = list(range(n))
    try:
        stripped_text = _strip_code_fence(text)
        data = json.loads(stripped_text)
    except (json.JSONDecodeError, TypeError):
        return identity, 0.0, True
    if not isinstance(data, dict):
        return identity, 0.0, True

    ranking = data.get("ranking")
    # bool ist eine int-Subklasse -- ohne den Ausschluss wuerde z.B.
    # [True, 0] als gueltige Indexliste durchgehen. sorted() faellt bei
    # gemischten Typen (z.B. [0, "1"]) sonst mit TypeError um, deshalb erst
    # der Typ-Check, dann erst sorted().
    is_index_list = isinstance(ranking, list) and all(
        isinstance(i, int) and not isinstance(i, bool) for i in ranking
    )
    ranking_fallback = not is_index_list or sorted(ranking) != identity
    if ranking_fallback:
        ranking = identity

    confidence = data.get("top1_confidence")
    is_plain_number = isinstance(confidence, (int, float)) and not isinstance(confidence, bool)
    confidence_fallback = not is_plain_number or not (0 <= confidence <= 10)
    if confidence_fallback:
        confidence = 0.0

    return ranking, float(confidence), ranking_fallback or confidence_fallback


def union_candidates(vector_ranking: list[str], bm25_ranking: list[str]) -> list[str]:
    """Vereinigung statt RRF-Top-n-Cut -- Kandidaten-Union-Fix: ohne ihn
    sieht kein Reranker Kandidaten, die nur in einer der beiden Top-10-
    Listen weit vorn liegen. Dedupe hier inline statt ueber
    app.retrieval._dedupe_ranking importiert, damit dieses Modul zur
    Laufzeit nichts aus app.retrieval braucht (zirkulaerer Import, siehe
    Spec: app.retrieval importiert umgekehrt aus diesem Modul)."""
    seen: set[str] = set()
    result: list[str] = []
    for doc_id in vector_ranking + bm25_ranking:
        if doc_id not in seen:
            seen.add(doc_id)
            result.append(doc_id)
    return result


async def llm_two_signal_rerank(
    query: str, docs: list[RetrievedDoc], client
) -> tuple[list[int], float, bool, int]:
    """Gibt (Rangfolge als 0-indexierte Positionsliste, top1_confidence,
    used_fallback, tokens) zurueck. ranking[0] ist der Index des nach
    Rangfolge bestplatzierten Kandidaten in docs. used_fallback ist True,
    wenn die Antwort nicht valide geparst werden konnte (siehe
    _parse_response) -- Aufrufer sollten das nicht mit einer echten
    confidence von 0.0 verwechseln. tokens = verbrauchte Input- + Output-
    Tokens, fuers Tagesbudget-Tracking."""
    response = await client.messages.create(
        model=settings.model,
        max_tokens=MAX_LLM_RERANK_TOKENS,
        system=_system_prompt(len(docs)),
        temperature=0,
        messages=[{"role": "user", "content": build_llm_rerank_prompt(query, docs)}],
    )
    text = next((b.text for b in response.content if b.type == "text"), "").strip()
    ranking, confidence, used_fallback = _parse_response(text, len(docs))
    tokens = response.usage.input_tokens + response.usage.output_tokens
    return ranking, confidence, used_fallback, tokens
```

- [ ] **Step 2: `eval/llm_reranker.py` löschen**

```bash
git rm eval/llm_reranker.py
```

- [ ] **Step 3: `tests/test_llm_reranker.py` anpassen**

Diese Datei importiert aktuell von `eval.llm_reranker` und testet u.a.
`two_signal_candidates` (entfällt, seine Logik wandert in Task 3 direkt in
`Retriever.retrieve()`) sowie `llm_two_signal_rerank` mit dem alten
3-Tupel. Öffne die Datei und wende diese Änderungen an:

1. Ersetze den Import-Block am Dateianfang. Alt:
```python
from app.retrieval import RetrievedDoc
from eval.llm_reranker import (
    _parse_response,
    _system_prompt,
    build_llm_rerank_prompt,
    union_candidates,
)
```
Neu:
```python
from app.llm_reranker import (
    _parse_response,
    _system_prompt,
    build_llm_rerank_prompt,
    union_candidates,
)
from app.retrieval import RetrievedDoc
```

2. Jedes weitere `from eval.llm_reranker import ...` (z.B. für
   `llm_two_signal_rerank`, `two_signal_candidates`) wird zu
   `from app.llm_reranker import ...` — außer `two_signal_candidates`,
   das entfällt komplett (siehe Punkt 4).

3. Alle Tests für `llm_two_signal_rerank`, die aktuell ein 3-Tupel
   entpacken (`ranking, confidence = await llm_two_signal_rerank(...)`
   bzw. `ranking, confidence, used_fallback = ...`), entpacken jetzt ein
   4-Tupel. Beispiel — alt:
```python
async def test_llm_two_signal_rerank_returns_parsed_ranking_and_confidence():
    client = _FakeClient('{"ranking": [1, 0], "top1_confidence": 9}')
    ranking, confidence = await llm_two_signal_rerank("Frage?", DOCS, client)
    assert ranking == [1, 0]
    assert confidence == 9.0
```
   Prüfe für jeden solchen Test, ob die Fake-Response-Objekte
   (`_FakeResponse`/`_FakeMessages`) bereits ein `usage`-Attribut mit
   `input_tokens`/`output_tokens` liefern (falls Task-1-Stufe1-Fake-
   Klassen das nicht taten, ergänzen — siehe `tests/test_llm.py:43-51`
   für das Referenzmuster `FakeUsage`). Passe die Assertions entsprechend
   an ein 4-Tupel an, z.B.:
```python
async def test_llm_two_signal_rerank_returns_parsed_ranking_and_confidence():
    client = _FakeClient('{"ranking": [1, 0], "top1_confidence": 9}')
    ranking, confidence, used_fallback, tokens = await llm_two_signal_rerank("Frage?", DOCS, client)
    assert ranking == [1, 0]
    assert confidence == 9.0
    assert used_fallback is False
    assert tokens == 95  # falls FakeUsage input=80, output=15 wie in tests/test_llm.py
```
   (Exakten Tokenwert an die tatsächlich in dieser Datei verwendete
   Fake-Usage anpassen — falls keine vorhanden ist, ergänze eine
   `_FakeUsage`-Klasse mit `input_tokens = 80` und `output_tokens = 15`
   analog zu `tests/test_llm.py`, und einen neuen Test, der explizit
   `tokens == 95` prüft.)

4. Entferne jeden Test, der `two_signal_candidates` importiert oder
   aufruft, komplett (inklusive der zugehörigen Hilfs-Fixtures
   `_CORPUS_DOCS`, `_DOC_VECS`, `_encode_one`, `_build_retriever`, falls
   sie ausschließlich von diesen Tests genutzt werden — prüfe vor dem
   Löschen, ob eine andere Test-Funktion in derselben Datei sie noch
   braucht).

- [ ] **Step 4: Tests laufen lassen**

Run: `c:/Users/Marco/02_Portfolio/chrono24-chatbot/.venv/Scripts/python.exe -m pytest tests/test_llm_reranker.py -v`
Expected: alle verbleibenden Tests PASS, keine `ImportError`

- [ ] **Step 5: Zirkulär-Import-Check**

Run: `c:/Users/Marco/02_Portfolio/chrono24-chatbot/.venv/Scripts/python.exe -c "import app.llm_reranker; import app.retrieval; print('ok')"`
Expected: `ok` (kein `ImportError`/`AttributeError` durch Zirkularität)

- [ ] **Step 6: Commit**

```bash
git add app/llm_reranker.py tests/test_llm_reranker.py
git commit -m "refactor: Reranker-Modul von eval/ nach app/ verschoben (Ersatz-Vorbereitung)"
```

---

### Task 3: `Retriever` — async, Union-Fix, LLM-Gate

**Files:**
- Modify: `app/retrieval.py`
- Test: `tests/test_retrieval.py`

**Interfaces:**
- Consumes: `llm_two_signal_rerank`, `union_candidates` aus `app.llm_reranker` (Task 2); `settings.use_llm_reranker` (Task 1).
- Produces: `async def Retriever.retrieve(self, query, top_k=5, audience=None, client=None) -> tuple[list[RetrievedDoc], int]`; `Retriever.__init__(..., use_llm_reranker: bool = settings.use_llm_reranker)`; Konstante `LLM_CONFIDENCE_THRESHOLD = 8.5`. Task 4 (Testmigration), Task 5 (`eval/run_eval.py`), Task 6 (`app/main.py`), Task 7 (`eval/judge.py`), Task 8 (Stufe-1/2-Skripte) konsumieren dieses neue `retrieve()`.

- [ ] **Step 1: Failing Test schreiben**

An `tests/test_retrieval.py` anhängen (nach dem letzten bestehenden Test):

```python
from app.retrieval import LLM_CONFIDENCE_THRESHOLD


class _FakeRerankClient:
    def __init__(self, response_text):
        self._response_text = response_text

    class _FakeTextBlock:
        type = "text"

        def __init__(self, text):
            self.text = text

    class _FakeUsage:
        input_tokens = 50
        output_tokens = 10

    class _FakeResponse:
        def __init__(self, text):
            self.content = [_FakeRerankClient._FakeTextBlock(text)]
            self.usage = _FakeRerankClient._FakeUsage()

    class _FakeMessages:
        def __init__(self, response_text):
            self._response_text = response_text

        async def create(self, **kwargs):
            return _FakeRerankClient._FakeResponse(self._response_text)

    @property
    def messages(self):
        return _FakeRerankClient._FakeMessages(self._response_text)


async def test_retrieve_uses_llm_reranker_when_flag_enabled(tmp_path):
    corpus_path = tmp_path / "corpus.json"
    corpus_path.write_text(json.dumps({"scraped_at": "2026-08-24", "documents": DOCS}),
                           encoding="utf-8")
    index_dir = tmp_path / "index"
    build_index(corpus_path, index_dir, encoder=lambda texts: [encode_one(t) for t in texts])
    retriever = Retriever(index_dir, corpus_path, encoder=encode_one, reranker=False,
                          bm25_threshold=1.0, use_llm_reranker=True)
    client = _FakeRerankClient('{"ranking": [1, 0, 2], "top1_confidence": 9}')
    docs, tokens = await retriever.retrieve("Wie funktioniert der Käuferschutz?", client=client)
    assert docs[0].id == "faq-0002"
    assert docs[0].rerank_score == 9.0
    assert tokens == 60


async def test_retrieve_llm_path_abstains_below_confidence_threshold(tmp_path):
    corpus_path = tmp_path / "corpus.json"
    corpus_path.write_text(json.dumps({"scraped_at": "2026-08-24", "documents": DOCS}),
                           encoding="utf-8")
    index_dir = tmp_path / "index"
    build_index(corpus_path, index_dir, encoder=lambda texts: [encode_one(t) for t in texts])
    retriever = Retriever(index_dir, corpus_path, encoder=encode_one, reranker=False,
                          bm25_threshold=1.0, use_llm_reranker=True)
    low_confidence = LLM_CONFIDENCE_THRESHOLD - 1
    client = _FakeRerankClient(
        '{"ranking": [0, 1, 2], "top1_confidence": ' + str(low_confidence) + '}')
    docs, tokens = await retriever.retrieve("Wie funktioniert der Käuferschutz?", client=client)
    assert docs == []
    assert tokens == 60


async def test_retrieve_llm_path_abstains_on_parse_fallback(tmp_path):
    corpus_path = tmp_path / "corpus.json"
    corpus_path.write_text(json.dumps({"scraped_at": "2026-08-24", "documents": DOCS}),
                           encoding="utf-8")
    index_dir = tmp_path / "index"
    build_index(corpus_path, index_dir, encoder=lambda texts: [encode_one(t) for t in texts])
    retriever = Retriever(index_dir, corpus_path, encoder=encode_one, reranker=False,
                          bm25_threshold=1.0, use_llm_reranker=True)
    client = _FakeRerankClient("kaputte Antwort, kein JSON")
    docs, tokens = await retriever.retrieve("Wie funktioniert der Käuferschutz?", client=client)
    assert docs == []
    assert tokens == 60


async def test_retrieve_llm_path_gate_fires_before_llm_call(tmp_path):
    corpus_path = tmp_path / "corpus.json"
    corpus_path.write_text(json.dumps({"scraped_at": "2026-08-24", "documents": DOCS}),
                           encoding="utf-8")
    index_dir = tmp_path / "index"
    build_index(corpus_path, index_dir, encoder=lambda texts: [encode_one(t) for t in texts])
    retriever = Retriever(index_dir, corpus_path, encoder=encode_one, reranker=False,
                          bm25_threshold=1.0, use_llm_reranker=True)

    class _ExplodingClient:
        @property
        def messages(self):
            raise AssertionError("LLM darf bei Off-Topic nicht aufgerufen werden")

    docs, tokens = await retriever.retrieve("Gedicht über Katzen bitte",
                                            client=_ExplodingClient())
    assert docs == []
    assert tokens == 0


async def test_retrieve_cross_encoder_path_returns_zero_tokens(tmp_path):
    retriever = make_retriever(tmp_path, reranker=neutral_reranker, use_llm_reranker=False)
    docs, tokens = await retriever.retrieve("Wie funktioniert der Käuferschutz?")
    assert docs
    assert tokens == 0


def test_init_skips_cross_encoder_load_when_llm_reranker_enabled(tmp_path, monkeypatch):
    def exploding_default_reranker():
        raise AssertionError("Cross-Encoder darf bei use_llm_reranker=True nicht geladen werden")

    monkeypatch.setattr("app.retrieval._default_reranker", exploding_default_reranker)
    corpus_path = tmp_path / "corpus.json"
    corpus_path.write_text(json.dumps({"scraped_at": "2026-08-24", "documents": DOCS}),
                           encoding="utf-8")
    index_dir = tmp_path / "index"
    build_index(corpus_path, index_dir, encoder=lambda texts: [encode_one(t) for t in texts])
    # Kein reranker=-Override, use_llm_reranker=True -> darf _default_reranker
    # nicht aufrufen. Waere das Verhalten falsch, wuerde die Zeile oben den
    # Test mit AssertionError zum Scheitern bringen.
    Retriever(index_dir, corpus_path, encoder=encode_one, bm25_threshold=1.0,
             use_llm_reranker=True)
```

Ergänze `make_retriever`s Signatur um den neuen Parameter (Modify statt
neuer Funktion):

Alt:
```python
def make_retriever(tmp_path, reranker, rerank_threshold=-6.0):
    ...
    return Retriever(index_dir, corpus_path, encoder=encode_one, reranker=reranker,
                     bm25_threshold=1.0, rerank_threshold=rerank_threshold)
```
Neu:
```python
def make_retriever(tmp_path, reranker, rerank_threshold=-6.0, use_llm_reranker=False):
    ...
    return Retriever(index_dir, corpus_path, encoder=encode_one, reranker=reranker,
                     bm25_threshold=1.0, rerank_threshold=rerank_threshold,
                     use_llm_reranker=use_llm_reranker)
```

- [ ] **Step 2: Tests laufen lassen, müssen fehlschlagen**

Run: `c:/Users/Marco/02_Portfolio/chrono24-chatbot/.venv/Scripts/python.exe -m pytest tests/test_retrieval.py -v -k "llm_reranker or llm_path or cross_encoder_path"`
Expected: FAIL — `TypeError: retrieve() got an unexpected keyword argument 'client'` oder `ImportError: cannot import name 'LLM_CONFIDENCE_THRESHOLD'` (Retriever kennt `use_llm_reranker`/`client`/neue Konstante noch nicht)

- [ ] **Step 3: `app/retrieval.py` implementieren**

Am Kopf der Datei, nach den bestehenden Imports, ergänzen:

```python
from app.llm_reranker import llm_two_signal_rerank, union_candidates
```

Nach `RERANK_THRESHOLD = 2.9` (bestehende Konstante), neue Konstante
ergänzen:

```python
# Stufe-2-Messung (siehe HANDOVER-llm-reranker.md, Integrationsentscheidung):
# on-topic-Minimum 9.0, off-topic-Maximum 9.0 (exakte Ueberlappung), Puffer
# 0.5 wie bei den anderen Schwellen. Ersetzt RERANK_THRESHOLD im LLM-Pfad --
# RERANK_THRESHOLD selbst bleibt fuer den Cross-Encoder-Rollback-Pfad
# bestehen.
LLM_CONFIDENCE_THRESHOLD = 8.5
```

`Retriever.__init__`-Signatur (aktuell Zeilen 122-125) ändern. Alt:
```python
    def __init__(self, index_dir: Path, corpus_path: Path, encoder=None, reranker=None,
                 sim_threshold: float = SIM_THRESHOLD,
                 bm25_threshold: float = BM25_THRESHOLD,
                 rerank_threshold: float = RERANK_THRESHOLD):
        """reranker: Callable[(query, texts) -> scores] | None (Default-Modell) | False (aus)."""
        self.sim_threshold = sim_threshold
        self.bm25_threshold = bm25_threshold
        self.rerank_threshold = rerank_threshold
        self.encoder = encoder or _default_encoder()
        if reranker is False:
            self.reranker = None
        else:
            self.reranker = reranker or _default_reranker()
        self.db = _connect(Path(index_dir) / "hybrid.db")
        corpus = json.loads(Path(corpus_path).read_text(encoding="utf-8"))
        self.docs = {d["id"]: d for d in corpus["documents"]}
```
Neu:
```python
    def __init__(self, index_dir: Path, corpus_path: Path, encoder=None, reranker=None,
                 sim_threshold: float = SIM_THRESHOLD,
                 bm25_threshold: float = BM25_THRESHOLD,
                 rerank_threshold: float = RERANK_THRESHOLD,
                 use_llm_reranker: bool = settings.use_llm_reranker):
        """reranker: Callable[(query, texts) -> scores] | None (Default-Modell) | False (aus).
        Nur relevant, wenn use_llm_reranker=False -- der LLM-Pfad ignoriert
        reranker komplett und braucht stattdessen einen client-Parameter
        bei retrieve()."""
        self.sim_threshold = sim_threshold
        self.bm25_threshold = bm25_threshold
        self.rerank_threshold = rerank_threshold
        self.use_llm_reranker = use_llm_reranker
        self.encoder = encoder or _default_encoder()
        if reranker is False:
            self.reranker = None
        elif reranker is not None:
            self.reranker = reranker
        elif self.use_llm_reranker:
            # Cross-Encoder-Load und HF_TOKEN-Abhaengigkeit entfallen im
            # LLM-Pfad komplett -- kein reranker=-Override, kein Bedarf.
            self.reranker = None
        else:
            self.reranker = _default_reranker()
        self.db = _connect(Path(index_dir) / "hybrid.db")
        corpus = json.loads(Path(corpus_path).read_text(encoding="utf-8"))
        self.docs = {d["id"]: d for d in corpus["documents"]}
```

`Retriever.retrieve` (aktuell Zeilen 199-231) ersetzen. Alt:
```python
    def retrieve(self, query: str, top_k: int = 5,
                 audience: str | None = None) -> list[RetrievedDoc]:
        total = self.db.execute("SELECT COUNT(*) FROM vectors").fetchone()[0]
        n = min(TOP_K_CANDIDATES, total)

        # Harter Pre-Filter vor der RRF-Fusion (kein Score-Abzug): Kandidaten
        # der falschen Rolle fliegen aus beiden Rankings komplett raus, statt
        # nur schlechter bewertet zu werden -- Unterschied zum verworfenen
        # weichen Rollen-Malus (siehe README). Nur bei eindeutiger
        # Klassifikation aktiv; "neutral" (Default bei Dokumenten ohne Feld)
        # passiert den Filter für beide Rollen. Jetzt als SQL-WHERE in beiden
        # Teil-Queries statt Python-seitigem Filtern (Schritt 2, siehe
        # corpus-storage-rethink-design.md).
        vector_ranking, best_sim = self._vector_candidates(query, n, total, audience)
        bm25_ranking, best_bm25 = self._bm25_candidates(query, n, audience)

        # Stufe 1 des Gates: billig, vor dem Reranker.
        if best_sim < self.sim_threshold or best_bm25 < self.bm25_threshold:
            return []

        fused = rrf_fuse([vector_ranking, bm25_ranking])
        candidates = sorted(fused.items(), key=lambda kv: kv[1], reverse=True)[:n]
        docs = [self._to_doc(doc_id, score) for doc_id, score in candidates]
        if self.reranker is not None:
            scores = self.reranker(query, [f"{d.title}\n{d.text}" for d in docs])
            for doc, score in zip(docs, scores):
                doc.rerank_score = round(float(score), 4)
            docs.sort(key=lambda d: d.rerank_score, reverse=True)
            # Stufe 2: passt selbst der beste Kandidat laut Cross-Encoder
            # eindeutig nicht, lieber leer als raten.
            if docs and docs[0].rerank_score < self.rerank_threshold:
                return []
        return docs[:top_k]
```
Neu:
```python
    async def retrieve(self, query: str, top_k: int = 5,
                       audience: str | None = None, client=None) -> tuple[list[RetrievedDoc], int]:
        """Gibt (Dokumente, verbrauchte Reranker-Tokens) zurueck. Tokens
        sind 0 im Cross-Encoder-Pfad (self.use_llm_reranker=False)."""
        total = self.db.execute("SELECT COUNT(*) FROM vectors").fetchone()[0]
        n = min(TOP_K_CANDIDATES, total)

        # Harter Pre-Filter vor der Kandidaten-Fusion (kein Score-Abzug):
        # Kandidaten der falschen Rolle fliegen aus beiden Rankings komplett
        # raus, statt nur schlechter bewertet zu werden. Nur bei eindeutiger
        # Klassifikation aktiv; "neutral" passiert den Filter fuer beide
        # Rollen. Als SQL-WHERE in beiden Teil-Queries.
        vector_ranking, best_sim = self._vector_candidates(query, n, total, audience)
        bm25_ranking, best_bm25 = self._bm25_candidates(query, n, audience)

        # Stufe 1 des Gates: billig, vor jedem Reranking (LLM oder
        # Cross-Encoder) -- muss fuer beide Pfade identisch bleiben.
        if best_sim < self.sim_threshold or best_bm25 < self.bm25_threshold:
            return [], 0

        if self.use_llm_reranker:
            ids = union_candidates(vector_ranking, bm25_ranking)
            docs = [self._to_doc(doc_id, 0.0) for doc_id in ids]
            ranking, confidence, used_fallback, tokens = await llm_two_signal_rerank(
                query, docs, client)
            if used_fallback or confidence < LLM_CONFIDENCE_THRESHOLD:
                return [], tokens
            ordered = [docs[i] for i in ranking]
            # Konfidenz gilt nur fuer Position 0 (siehe Spec) -- nur dort
            # gesetzt, fuer SSE-Observability analog zum rerank_score im
            # Cross-Encoder-Pfad.
            ordered[0].rerank_score = round(confidence, 4)
            return ordered[:top_k], tokens

        fused = rrf_fuse([vector_ranking, bm25_ranking])
        candidates = sorted(fused.items(), key=lambda kv: kv[1], reverse=True)[:n]
        docs = [self._to_doc(doc_id, score) for doc_id, score in candidates]
        if self.reranker is not None:
            scores = self.reranker(query, [f"{d.title}\n{d.text}" for d in docs])
            for doc, score in zip(docs, scores):
                doc.rerank_score = round(float(score), 4)
            docs.sort(key=lambda d: d.rerank_score, reverse=True)
            # Stufe 2: passt selbst der beste Kandidat laut Cross-Encoder
            # eindeutig nicht, lieber leer als raten.
            if docs and docs[0].rerank_score < self.rerank_threshold:
                return [], 0
        return docs[:top_k], 0
```

- [ ] **Step 4: Tests laufen lassen, müssen passen**

Run: `c:/Users/Marco/02_Portfolio/chrono24-chatbot/.venv/Scripts/python.exe -m pytest tests/test_retrieval.py -v`
Expected: die 6 neuen Tests PASS. Alle bisherigen Tests in dieser Datei
schlagen jetzt fehl (sync-Aufrufe gegen eine jetzt-async-Methode) — das
ist erwartet, Task 4 behebt das. Notiere die Fehlerzahl für Task 4s
Verifikation.

- [ ] **Step 5: Commit**

```bash
git add app/retrieval.py tests/test_retrieval.py
git commit -m "feat: Retriever.retrieve() async mit LLM-Reranker-Pfad und Rollback-Flag"
```

---

### Task 4: Bestehende Cross-Encoder-Tests migrieren

**Files:**
- Modify: `tests/test_retrieval.py`

**Interfaces:**
- Consumes: `async def Retriever.retrieve(...) -> tuple[list[RetrievedDoc], int]` (Task 3).

Alle Tests in dieser Datei, die vor Task 3 bereits existierten (nicht die
6 neuen aus Task 3), rufen `retriever.retrieve(...)` noch synchron und
ohne Tupel-Entpacken auf. Jede betroffene Funktion braucht: `async def`
davor, `await` vor dem `.retrieve(...)`-Aufruf, und das Ergebnis als
`docs, _ = await ...` entpackt (Tokens werden in diesen Tests nicht
geprüft, außer wo unten explizit anders angegeben). Jede direkte
`Retriever(...)`-Konstruktion (nicht über `make_retriever`) bekommt
zusätzlich `use_llm_reranker=False`.

- [ ] **Step 1: Jede betroffene Testfunktion anpassen**

`test_retrieve_finds_matching_faq` — alt:
```python
def test_retrieve_finds_matching_faq(retriever):
    docs = retriever.retrieve("Wie funktioniert der Käuferschutz?")
```
neu:
```python
async def test_retrieve_finds_matching_faq(retriever):
    docs, _ = await retriever.retrieve("Wie funktioniert der Käuferschutz?")
```

`test_retrieve_returns_empty_for_offtopic` — alt:
```python
def test_retrieve_returns_empty_for_offtopic(retriever):
    docs = retriever.retrieve("Gedicht über Katzen bitte")
```
neu:
```python
async def test_retrieve_returns_empty_for_offtopic(retriever):
    docs, _ = await retriever.retrieve("Gedicht über Katzen bitte")
```

`test_reranker_reorders_candidates` — alt:
```python
def test_reranker_reorders_candidates(tmp_path):
    def prefer_selling(query, texts):
        return [2.0 if "Verkäuferkonto" in t else 1.0 for t in texts]

    retriever = make_retriever(tmp_path, reranker=prefer_selling)
    docs = retriever.retrieve("Wie funktioniert der Käuferschutz?")
```
neu:
```python
async def test_reranker_reorders_candidates(tmp_path):
    def prefer_selling(query, texts):
        return [2.0 if "Verkäuferkonto" in t else 1.0 for t in texts]

    retriever = make_retriever(tmp_path, reranker=prefer_selling, use_llm_reranker=False)
    docs, _ = await retriever.retrieve("Wie funktioniert der Käuferschutz?")
```

`test_reranker_false_keeps_rrf_order` — alt:
```python
def test_reranker_false_keeps_rrf_order(tmp_path):
    retriever = make_retriever(tmp_path, reranker=False)
    docs = retriever.retrieve("Wie funktioniert der Käuferschutz?")
```
neu:
```python
async def test_reranker_false_keeps_rrf_order(tmp_path):
    retriever = make_retriever(tmp_path, reranker=False, use_llm_reranker=False)
    docs, _ = await retriever.retrieve("Wie funktioniert der Käuferschutz?")
```

`test_gate_fires_before_reranker` — alt:
```python
def test_gate_fires_before_reranker(tmp_path):
    def exploding_reranker(query, texts):
        raise AssertionError("Reranker darf bei Off-Topic nicht laufen")

    retriever = make_retriever(tmp_path, reranker=exploding_reranker)
    assert retriever.retrieve("Gedicht über Katzen bitte") == []
```
neu:
```python
async def test_gate_fires_before_reranker(tmp_path):
    def exploding_reranker(query, texts):
        raise AssertionError("Reranker darf bei Off-Topic nicht laufen")

    retriever = make_retriever(tmp_path, reranker=exploding_reranker, use_llm_reranker=False)
    docs, _ = await retriever.retrieve("Gedicht über Katzen bitte")
    assert docs == []
```

`test_variant_hit_resolves_to_canonical_doc` — die
`Retriever(index_dir, corpus_path, encoder=encode, reranker=False, bm25_threshold=float("-inf"))`-Zeile
bekommt `use_llm_reranker=False` ergänzt; die Aufrufzeile
```python
    docs_out = retriever.retrieve("Was deckt der Kaeuferschutz ab?", top_k=5)
```
wird zu
```python
    docs_out, _ = await retriever.retrieve("Was deckt der Kaeuferschutz ab?", top_k=5)
```
und die Funktionssignatur bekommt `async def` davor.

`test_vector_path_yields_n_distinct_canonical_docs_after_dedupe` — analog:
`Retriever(...)`-Konstruktion bekommt `use_llm_reranker=False`, Aufruf
```python
    docs_out = retriever.retrieve("Xylophon Quietscheentchen Zauberstab", top_k=num_faqs)
```
wird zu
```python
    docs_out, _ = await retriever.retrieve("Xylophon Quietscheentchen Zauberstab", top_k=num_faqs)
```
Funktion bekommt `async def`.

`test_gate_abstains_when_only_bm25_is_low` — `Retriever(...)`-Konstruktion
bekommt `use_llm_reranker=False`. Alt:
```python
    assert retriever.retrieve("Wie backe ich einen Hefezopf?") == []
    # Gegenprobe: eine echte Frage mit demselben Vektor bleibt durch.
    assert retriever.retrieve("Wie funktioniert der Käuferschutz?")
```
neu:
```python
    docs, _ = await retriever.retrieve("Wie backe ich einen Hefezopf?")
    assert docs == []
    # Gegenprobe: eine echte Frage mit demselben Vektor bleibt durch.
    docs, _ = await retriever.retrieve("Wie funktioniert der Käuferschutz?")
    assert docs
```
Funktion bekommt `async def`.

`test_gate_abstains_when_reranker_rejects_every_candidate` — alt:
```python
def test_gate_abstains_when_reranker_rejects_every_candidate(tmp_path):
    """..."""
    def rejects_all(query, texts):
        return [-9.0 for _ in texts]

    retriever = make_retriever(tmp_path, reranker=rejects_all)
    assert retriever.retrieve("Wie funktioniert der Käuferschutz?") == []
```
neu:
```python
async def test_gate_abstains_when_reranker_rejects_every_candidate(tmp_path):
    """..."""
    def rejects_all(query, texts):
        return [-9.0 for _ in texts]

    retriever = make_retriever(tmp_path, reranker=rejects_all, use_llm_reranker=False)
    docs, _ = await retriever.retrieve("Wie funktioniert der Käuferschutz?")
    assert docs == []
```

`test_retrieve_with_audience_excludes_wrong_role` — `Retriever(...)`-
Konstruktion bekommt `use_llm_reranker=False`. Alt:
```python
    # Ohne audience-Filter finden beide Rollen den gleichen Kandidatenpool.
    docs_out = retriever.retrieve("Wie funktioniert der Schutz?")
    assert {d.id for d in docs_out} == {"faq-buyer", "faq-seller"}

    # Mit hartem Filter verschwindet das Verkaeufer-Dokument komplett.
    docs_out = retriever.retrieve("Wie funktioniert der Schutz?", audience="kaeufer")
```
neu:
```python
    # Ohne audience-Filter finden beide Rollen den gleichen Kandidatenpool.
    docs_out, _ = await retriever.retrieve("Wie funktioniert der Schutz?")
    assert {d.id for d in docs_out} == {"faq-buyer", "faq-seller"}

    # Mit hartem Filter verschwindet das Verkaeufer-Dokument komplett.
    docs_out, _ = await retriever.retrieve("Wie funktioniert der Schutz?", audience="kaeufer")
```
Funktion bekommt `async def`.

`test_gate_keeps_candidates_when_reranker_is_merely_unsure` — alt:
```python
def test_gate_keeps_candidates_when_reranker_is_merely_unsure(tmp_path):
    """..."""
    def unsure(query, texts):
        return [-5.0 for _ in texts]

    retriever = make_retriever(tmp_path, reranker=unsure)
    assert retriever.retrieve("Wie funktioniert der Käuferschutz?")
```
neu:
```python
async def test_gate_keeps_candidates_when_reranker_is_merely_unsure(tmp_path):
    """..."""
    def unsure(query, texts):
        return [-5.0 for _ in texts]

    retriever = make_retriever(tmp_path, reranker=unsure, use_llm_reranker=False)
    docs, _ = await retriever.retrieve("Wie funktioniert der Käuferschutz?")
    assert docs
```

`test_rrf_fuse_rewards_docs_in_both_rankings` bleibt unverändert (testet
`rrf_fuse` direkt, keine `Retriever`-Instanz beteiligt).

- [ ] **Step 2: Tests laufen lassen, müssen passen**

Run: `c:/Users/Marco/02_Portfolio/chrono24-chatbot/.venv/Scripts/python.exe -m pytest tests/test_retrieval.py -v`
Expected: alle Tests in dieser Datei PASS (die 6 aus Task 3 + alle hier
migrierten)

- [ ] **Step 3: Commit**

```bash
git add tests/test_retrieval.py
git commit -m "test: bestehende Retriever-Tests auf async migriert"
```

---

### Task 5: `eval/run_eval.py` + `tests/test_eval.py`

**Files:**
- Modify: `eval/run_eval.py`
- Modify: `tests/test_eval.py`

**Interfaces:**
- Consumes: `async def Retriever.retrieve(...) -> tuple[list[RetrievedDoc], int]` (Task 3).
- Produces: `async def hit_rate_at_k(...)`, `async def hit_rate_at_k_with_audience(...)`, `async def abstention_rate(...)` — unverändertes Rückgabeformat `tuple[float, list[dict]]`, nur die Funktionen selbst werden async.

- [ ] **Step 1: `eval/run_eval.py` anpassen**

`hit_rate_at_k` — alt:
```python
def hit_rate_at_k(retriever, questions: list[dict], k: int = 5) -> tuple[float, list[dict]]:
    misses = []
    hits = 0
    for item in questions:
        ids = [d.id for d in retriever.retrieve(eval_query(item), top_k=k)]
        if item["expected_doc_id"] in ids:
            hits += 1
        else:
            misses.append({**item, "got": ids})
    return hits / len(questions), misses
```
neu:
```python
async def hit_rate_at_k(retriever, questions: list[dict], k: int = 5) -> tuple[float, list[dict]]:
    misses = []
    hits = 0
    for item in questions:
        docs, _ = await retriever.retrieve(eval_query(item), top_k=k)
        ids = [d.id for d in docs]
        if item["expected_doc_id"] in ids:
            hits += 1
        else:
            misses.append({**item, "got": ids})
    return hits / len(questions), misses
```

`hit_rate_at_k_with_audience` — alt:
```python
def hit_rate_at_k_with_audience(
    retriever, questions: list[dict], k: int = 5
) -> tuple[float, list[dict]]:
    """..."""
    from app.textproc import classify_audience

    misses = []
    hits = 0
    for item in questions:
        query = eval_query(item)
        audience = classify_audience(query)
        ids = [d.id for d in retriever.retrieve(query, top_k=k, audience=audience)]
        if item["expected_doc_id"] in ids:
            hits += 1
        else:
            misses.append({**item, "got": ids, "audience": audience})
    return hits / len(questions), misses
```
neu:
```python
async def hit_rate_at_k_with_audience(
    retriever, questions: list[dict], k: int = 5, client=None
) -> tuple[float, list[dict]]:
    """..."""
    from app.textproc import classify_audience

    misses = []
    hits = 0
    for item in questions:
        query = eval_query(item)
        audience = classify_audience(query)
        docs, _ = await retriever.retrieve(query, top_k=k, audience=audience, client=client)
        ids = [d.id for d in docs]
        if item["expected_doc_id"] in ids:
            hits += 1
        else:
            misses.append({**item, "got": ids, "audience": audience})
    return hits / len(questions), misses
```

`abstention_rate` — alt:
```python
def abstention_rate(retriever, questions: list[dict]) -> tuple[float, list[dict]]:
    """..."""
    false_hits = []
    abstained = 0
    for item in questions:
        docs = retriever.retrieve(item["question"])
        if not docs:
            abstained += 1
        else:
            false_hits.append({**item, "got_id": docs[0].id, "got_title": docs[0].title})
    return abstained / len(questions), false_hits
```
neu:
```python
async def abstention_rate(retriever, questions: list[dict], client=None) -> tuple[float, list[dict]]:
    """..."""
    false_hits = []
    abstained = 0
    for item in questions:
        docs, _ = await retriever.retrieve(item["question"], client=client)
        if not docs:
            abstained += 1
        else:
            false_hits.append({**item, "got_id": docs[0].id, "got_title": docs[0].title})
    return abstained / len(questions), false_hits
```

Der `__main__`-Block braucht einen echten Client und `asyncio.run`. Alt
(kompletter Block):
```python
if __name__ == "__main__":
    import sys

    from app.config import settings
    from app.retrieval import Retriever

    retriever = Retriever(settings.index_dir, settings.corpus_path)

    if "--gate" in sys.argv:
        tuning = json.loads(QUESTIONS_PATH.read_text(encoding="utf-8"))
        holdout = json.loads(HOLDOUT_QUESTIONS_PATH.read_text(encoding="utf-8"))
        offtopic = json.loads(OFFTOPIC_QUESTIONS_PATH.read_text(encoding="utf-8"))
        # Misst den harten audience-Filter mit (Schritt 1,
        # corpus-storage-rethink-design.md) -- abstention_rate bleibt bewusst
        # ohne Klassifikation, off-topic-Fragen haben keine Rolle.
        tuning_rate, tuning_misses = hit_rate_at_k_with_audience(retriever, tuning)
        holdout_rate, _ = hit_rate_at_k_with_audience(retriever, holdout)
        abstain_rate, false_hits = abstention_rate(retriever, offtopic)
        # Punktzahl plus 95-%-Intervall: bei 14 bis 33 Fragen sagt die Zahl
        # allein zu wenig, und ohne die Spanne daneben wird ueber Unterschiede
        # gestritten, die vollstaendig im Rauschen liegen.
        print(f"Tuning-Hit-Rate@5:  {format_rate(round(tuning_rate * len(tuning)), len(tuning))}")
        print(f"Holdout-Hit-Rate@5: {format_rate(round(holdout_rate * len(holdout)), len(holdout))}")
        print(f"Abstention-Rate:    {format_rate(len(offtopic) - len(false_hits), len(offtopic))}")
        for miss in tuning_misses:
            print(f"  TUNING MISS ({miss['audience']}): {miss['question']!r} erwartet "
                  f"{miss['expected_doc_id']}, bekam {miss['got']}")
        for false_hit in false_hits:
            print(f"  FALSE HIT: {false_hit['question']!r} -> "
                  f"{false_hit['got_id']} ({false_hit['got_title']!r})")
        failures = check_gate(tuning_rate, holdout_rate, abstain_rate)
        for failure in failures:
            print(f"GATE FAIL: {failure}")
        sys.exit(1 if failures else 0)

    questions = json.loads(_questions_path(sys.argv).read_text(encoding="utf-8"))
    if "--with-rewrite" in sys.argv:
        questions = _rewrite_questions(questions)
    rate, misses = hit_rate_at_k(retriever, questions)
    print(f"Hit-Rate@5: {format_rate(len(questions) - len(misses), len(questions))}")
    for miss in misses:
        print(f"  MISS: {miss['question']!r} erwartet {miss['expected_doc_id']}, bekam {miss['got']}")
```
neu:
```python
async def _run_gate(retriever, client) -> int:
    tuning = json.loads(QUESTIONS_PATH.read_text(encoding="utf-8"))
    holdout = json.loads(HOLDOUT_QUESTIONS_PATH.read_text(encoding="utf-8"))
    offtopic = json.loads(OFFTOPIC_QUESTIONS_PATH.read_text(encoding="utf-8"))
    tuning_rate, tuning_misses = await hit_rate_at_k_with_audience(retriever, tuning, client=client)
    holdout_rate, _ = await hit_rate_at_k_with_audience(retriever, holdout, client=client)
    abstain_rate, false_hits = await abstention_rate(retriever, offtopic, client=client)
    print(f"Tuning-Hit-Rate@5:  {format_rate(round(tuning_rate * len(tuning)), len(tuning))}")
    print(f"Holdout-Hit-Rate@5: {format_rate(round(holdout_rate * len(holdout)), len(holdout))}")
    print(f"Abstention-Rate:    {format_rate(len(offtopic) - len(false_hits), len(offtopic))}")
    for miss in tuning_misses:
        print(f"  TUNING MISS ({miss['audience']}): {miss['question']!r} erwartet "
              f"{miss['expected_doc_id']}, bekam {miss['got']}")
    for false_hit in false_hits:
        print(f"  FALSE HIT: {false_hit['question']!r} -> "
              f"{false_hit['got_id']} ({false_hit['got_title']!r})")
    failures = check_gate(tuning_rate, holdout_rate, abstain_rate)
    for failure in failures:
        print(f"GATE FAIL: {failure}")
    return 1 if failures else 0


async def _run_plain(retriever, argv: list[str]) -> None:
    questions = json.loads(_questions_path(argv).read_text(encoding="utf-8"))
    if "--with-rewrite" in argv:
        questions = _rewrite_questions(questions)
    rate, misses = await hit_rate_at_k(retriever, questions)
    print(f"Hit-Rate@5: {format_rate(len(questions) - len(misses), len(questions))}")
    for miss in misses:
        print(f"  MISS: {miss['question']!r} erwartet {miss['expected_doc_id']}, bekam {miss['got']}")


if __name__ == "__main__":
    import asyncio
    import sys

    from app.config import settings
    from app.llm import get_client
    from app.retrieval import Retriever

    async def _main() -> int:
        retriever = Retriever(settings.index_dir, settings.corpus_path)
        client = get_client() if settings.use_llm_reranker else None
        if "--gate" in sys.argv:
            return await _run_gate(retriever, client)
        await _run_plain(retriever, sys.argv)
        return 0

    sys.exit(asyncio.run(_main()))
```

- [ ] **Step 2: `tests/test_eval.py` anpassen**

`StubRetriever` — alt:
```python
class StubRetriever:
    def __init__(self, mapping):
        self.mapping = mapping

    def retrieve(self, query, top_k=5):
        return [RetrievedDoc(id=i, type="faq", title=f"Titel {i}", url="u", text="x", score=0.1)
                for i in self.mapping.get(query, [])]
```
neu:
```python
class StubRetriever:
    def __init__(self, mapping):
        self.mapping = mapping

    async def retrieve(self, query, top_k=5, audience=None, client=None):
        docs = [RetrievedDoc(id=i, type="faq", title=f"Titel {i}", url="u", text="x", score=0.1)
                for i in self.mapping.get(query, [])]
        return docs, 0
```

Beide Testfunktionen (`test_hit_rate_counts_hits_in_top_k`,
`test_abstention_rate_counts_questions_with_no_hits`) bekommen `async def`
davor, und ihre Aufrufe von `hit_rate_at_k(...)`/`abstention_rate(...)`
bekommen `await` davor:

```python
async def test_hit_rate_counts_hits_in_top_k():
    ...
    rate, misses = await hit_rate_at_k(retriever, questions, k=5)
    ...


async def test_abstention_rate_counts_questions_with_no_hits():
    ...
    rate, false_hits = await abstention_rate(retriever, questions)
    ...
```

- [ ] **Step 3: Tests laufen lassen**

Run: `c:/Users/Marco/02_Portfolio/chrono24-chatbot/.venv/Scripts/python.exe -m pytest tests/test_eval.py tests/test_run_eval.py -v`
Expected: alle PASS

- [ ] **Step 4: Commit**

```bash
git add eval/run_eval.py tests/test_eval.py
git commit -m "feat: eval/run_eval.py auf async Retriever migriert"
```

---

### Task 6: `app/main.py` + `tests/test_api.py`

**Files:**
- Modify: `app/main.py`
- Modify: `tests/test_api.py`

**Interfaces:**
- Consumes: `async def Retriever.retrieve(...) -> tuple[list[RetrievedDoc], int]` (Task 3).

- [ ] **Step 1: `app/main.py` anpassen**

Alt (Zeile 104):
```python
                docs = app.state.retriever.retrieve(standalone, audience=audience)
```
Neu:
```python
                docs, rerank_tokens = await app.state.retriever.retrieve(
                    standalone, audience=audience, client=client)
                if rerank_tokens:
                    app.state.budget.spend(rerank_tokens)
```

- [ ] **Step 2: `tests/test_api.py` anpassen**

`FakeRetriever` — alt:
```python
class FakeRetriever:
    def __init__(self, docs):
        self.docs = docs

    def retrieve(self, query, top_k=5, audience=None):
        return self.docs
```
neu:
```python
class FakeRetriever:
    def __init__(self, docs):
        self.docs = docs

    async def retrieve(self, query, top_k=5, audience=None, client=None):
        return self.docs, 0
```

- [ ] **Step 3: Tests laufen lassen**

Run: `c:/Users/Marco/02_Portfolio/chrono24-chatbot/.venv/Scripts/python.exe -m pytest tests/test_api.py -v`
Expected: alle PASS

- [ ] **Step 4: Commit**

```bash
git add app/main.py tests/test_api.py
git commit -m "feat: app/main.py ruft async Retriever.retrieve() auf, trackt Reranker-Tokens"
```

---

### Task 7: `eval/judge.py` + `tests/test_judge.py`

**Files:**
- Modify: `eval/judge.py`
- Modify: `tests/test_judge.py`

**Interfaces:**
- Consumes: `async def Retriever.retrieve(...) -> tuple[list[RetrievedDoc], int]` (Task 3).

- [ ] **Step 1: `eval/judge.py` anpassen**

Alt (Zeile 90):
```python
    docs = retriever.retrieve(standalone, top_k=5)
```
Neu:
```python
    docs, _ = await retriever.retrieve(standalone, top_k=5)
```

- [ ] **Step 2: `tests/test_judge.py` anpassen**

`FakeRetriever` — alt:
```python
class FakeRetriever:
    def __init__(self, docs):
        self.docs = docs

    def retrieve(self, query, top_k=5):
        return self.docs
```
neu:
```python
class FakeRetriever:
    def __init__(self, docs):
        self.docs = docs

    async def retrieve(self, query, top_k=5, audience=None, client=None):
        return self.docs, 0
```

- [ ] **Step 3: Tests laufen lassen**

Run: `c:/Users/Marco/02_Portfolio/chrono24-chatbot/.venv/Scripts/python.exe -m pytest tests/test_judge.py -v`
Expected: alle PASS

- [ ] **Step 4: Commit**

```bash
git add eval/judge.py tests/test_judge.py
git commit -m "feat: eval/judge.py ruft async Retriever.retrieve() auf"
```

---

### Task 8: Stufe-1/2-Skripte auf `app.llm_reranker` umstellen

**Files:**
- Modify: `eval/run_llm_reranker_experiment.py`
- Modify: `eval/run_llm_reranker_stufe2.py`
- Modify: `tests/test_run_llm_reranker_experiment.py`
- Modify: `tests/test_run_llm_reranker_stufe2.py`

**Interfaces:**
- Consumes: `app.llm_reranker.llm_two_signal_rerank(...) -> tuple[list[int], float, bool, int]` (Task 2, neues 4-Tupel).

Diese vier Dateien importierten bisher `two_signal_candidates` aus
`eval.llm_reranker` (jetzt entfernt, Task 2/3) und entpacken
`llm_two_signal_rerank`s Ergebnis als 3-Tupel. Beide Skripte nutzen eigene
Retriever-Wrapper-Funktionen, die dieselbe Union-Kandidaten-Logik jetzt
direkt selbst bauen müssen (die Logik ist nach `Retriever.retrieve()`
gewandert, aber diese Skripte greifen bewusst auf die privaten
`_vector_candidates`/`_bm25_candidates`-Helfer zu, nicht auf das neue
`retrieve()`, weil sie unabhängig vom `use_llm_reranker`-Flag immer den
LLM-Pfad testen wollen, egal was `settings.use_llm_reranker` gerade ist).

- [ ] **Step 1: `eval/run_llm_reranker_experiment.py` anpassen**

Import-Zeile — alt:
```python
from eval.llm_reranker import llm_two_signal_rerank, two_signal_candidates
```
neu:
```python
from app.llm_reranker import llm_two_signal_rerank, union_candidates
from app.retrieval import TOP_K_CANDIDATES
```

Neue lokale Hilfsfunktion ergänzen (ersetzt die entfallene
`two_signal_candidates`, gleiches Verhalten):
```python
def _two_signal_candidates(retriever, query, audience):
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

Jeder Aufruf von `two_signal_candidates(retriever, query, audience)` wird
zu `_two_signal_candidates(retriever, query, audience)`.

Jede Zeile `ranking, confidence = await llm_two_signal_rerank(query, docs, client)`
wird zu `ranking, confidence, used_fallback, _tokens = await llm_two_signal_rerank(query, docs, client)`
(falls die Datei `used_fallback` schon aus einem 3-Tupel entpackt hat,
wird daraus ein 4-Tupel-Unpack mit zusätzlichem `_tokens`).

`Retriever(settings.index_dir, settings.corpus_path, reranker=False)` in
`main()` bekommt zusätzlich `use_llm_reranker=False` (dieses Skript nutzt
weiterhin seine eigene `_two_signal_candidates`-Funktion statt
`retriever.retrieve()`, der Wert von `use_llm_reranker` ist hier ohne
Wirkung, aber explizit `False` vermeidet den überflüssigen Cross-Encoder-
Skip-Zweig-Verwirrung und macht die Intention klar).

- [ ] **Step 2: `eval/run_llm_reranker_stufe2.py` analog anpassen**

Gleiches Muster: Import von `two_signal_candidates` durch die lokale
`_two_signal_candidates`-Hilfsfunktion (Step 1) ersetzen — dieses Skript
importiert bereits `TOP_K_CANDIDATES` und `Retriever` aus `app.retrieval`,
ergänze nur den `union_candidates`-Import aus `app.llm_reranker` statt
`eval.llm_reranker`, und füge dieselbe `_two_signal_candidates`-Funktion
hinzu (dupliziert aus Step 1 — beide Skripte sind eigenständige, jeweils
für sich lauffähige Experiment-Skripte). `two_signal_result` in dieser
Datei ruft `two_signal_candidates` auf — Aufruf wird zu
`_two_signal_candidates`. Jedes
`ranking, confidence, used_fallback = await llm_two_signal_rerank(...)`
wird zu `ranking, confidence, used_fallback, _tokens = await llm_two_signal_rerank(...)`.

- [ ] **Step 3: `tests/test_run_llm_reranker_experiment.py` prüfen**

Diese Datei testet nur die reinen Funktionen `_known_miss_cases` und
`_offtopic_sample`, die von der Reranker-Änderung nicht betroffen sind —
falls sie `two_signal_candidates` oder `llm_two_signal_rerank` gar nicht
importiert, ist keine Änderung nötig. Prüfe per Grep, ob einer der beiden
Namen vorkommt; falls ja, Importzeile analog zu Step 1 anpassen.

- [ ] **Step 4: `tests/test_run_llm_reranker_stufe2.py` anpassen**

Diese Datei importiert `two_signal_result` und ruft es über einen
`_FakeClient` auf, der `llm_two_signal_rerank`-Antworten simuliert — die
Tests prüfen nicht `two_signal_candidates` direkt (das war intern in
`eval/llm_reranker.py` gekapselt). Falls `two_signal_result`s
Implementierung in `eval/run_llm_reranker_stufe2.py` selbst
`two_signal_candidates` aufruft (siehe Datei), passt sich das automatisch
an, sobald Step 2 die lokale `_two_signal_candidates`-Funktion einführt —
keine Teständerung nötig, außer die Tests importieren `union_candidates`
direkt aus `eval.llm_reranker` (dann Importzeile auf `app.llm_reranker`
ändern).

- [ ] **Step 5: Tests laufen lassen**

Run: `c:/Users/Marco/02_Portfolio/chrono24-chatbot/.venv/Scripts/python.exe -m pytest tests/test_run_llm_reranker_experiment.py tests/test_run_llm_reranker_stufe2.py -v`
Expected: alle PASS

- [ ] **Step 6: Commit**

```bash
git add eval/run_llm_reranker_experiment.py eval/run_llm_reranker_stufe2.py tests/test_run_llm_reranker_experiment.py tests/test_run_llm_reranker_stufe2.py
git commit -m "refactor: Stufe-1/2-Skripte auf app.llm_reranker und neues 4-Tupel umgestellt"
```

---

### Task 9: CI-Gate auf den LLM-Pfad umstellen

**Files:**
- Modify: `.github/workflows/ci.yml`

- [ ] **Step 1: `eval-gate`-Job anpassen**

Alt:
```yaml
  eval-gate:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.12"
          cache: pip
      - uses: actions/cache@v4
        with:
          path: ~/.cache/huggingface
          key: hf-models-${{ hashFiles('app/config.py') }}
          restore-keys: |
            hf-models-
      - run: pip install -r requirements-dev.txt
      # Index wird nicht mehr committet (siehe README, "Pipeline neu bauen") --
      # ohne API-Kosten, Embedding laeuft lokal.
      - run: python -m pipeline.index
      - run: python -m eval.run_eval --gate
```
Neu:
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
      # Index wird nicht mehr committet (siehe README, "Pipeline neu bauen") --
      # ohne API-Kosten, Embedding laeuft lokal.
      - run: python -m pipeline.index
      # Testet ab jetzt den LLM-Reranker-Pfad (settings.use_llm_reranker
      # Standard True) statt des Cross-Encoders -- das Gate soll pruefen,
      # was tatsaechlich live laeuft. Kostet ab jetzt echte API-Calls pro
      # Lauf (siehe HANDOVER-llm-reranker.md, Budget-Messung). Kein
      # HF-Modell-Cache-Step mehr noetig -- der Cross-Encoder wird bei
      # use_llm_reranker=True nicht geladen.
      - run: python -m eval.run_eval --gate
        env:
          ANTHROPIC_API_KEY: ${{ secrets.ANTHROPIC_API_KEY }}
```

- [ ] **Step 2: YAML-Syntax prüfen**

Run: `c:/Users/Marco/02_Portfolio/chrono24-chatbot/.venv/Scripts/python.exe -c "import yaml; yaml.safe_load(open('.github/workflows/ci.yml')); print('ok')"`
Expected: `ok` (falls `pyyaml` nicht installiert ist, stattdessen die Datei
manuell auf konsistente Einrückung prüfen — Leerzeichen, kein Tab, wie im
Rest der Datei)

- [ ] **Step 3: Commit**

```bash
git add .github/workflows/ci.yml
git commit -m "ci: eval-gate testet ab jetzt den LLM-Reranker-Pfad statt des Cross-Encoders"
```

---

### Task 10: Voller Testlauf + manuelle Verifikation

Kein neuer Code — Abschlussprüfung, dass die gesamte Migration
konsistent ist.

- [ ] **Step 1: Vollen Testlauf ausführen**

Run: `c:/Users/Marco/02_Portfolio/chrono24-chatbot/.venv/Scripts/python.exe -m pytest -q`
Expected: alle Tests PASS, 0 Failures

- [ ] **Step 2: Ruff**

Run: `c:/Users/Marco/02_Portfolio/chrono24-chatbot/.venv/Scripts/python.exe -m ruff check .`
Expected: `All checks passed!`

- [ ] **Step 3: Manueller Smoke-Test gegen die echte API**

Run: `c:/Users/Marco/02_Portfolio/chrono24-chatbot/.venv/Scripts/python.exe -m eval.run_eval --gate`
(braucht `ANTHROPIC_API_KEY` in der Umgebung, kostet echte API-Calls für
alle 62 Fragen — bewusst der erste End-to-End-Beweis, dass der neue
Produktionspfad wirklich funktioniert, nicht nur die gemockten Tests)

Erwartung laut Stufe-2-Messung: Tuning-Hit-Rate@5 100 %, Holdout-Hit-
Rate@5 100 %, Abstention-Rate ~93 %. Weicht das Ergebnis stark ab
(insbesondere `used_fallback`/Parse-Fehler-Häufungen), vor dem nächsten
Schritt untersuchen statt zu ignorieren.

- [ ] **Step 4: Ergebnis im Handover festhalten**

`HANDOVER-llm-reranker.md` um einen Abschnitt „Integration abgeschlossen"
ergänzen: Datum, Ergebnis aus Step 3, Verweis auf den Rollback-Weg
(`USE_LLM_RERANKER=false`).
