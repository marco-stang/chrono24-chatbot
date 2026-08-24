# Design: LLM-Reranker-Integration (Ersatz mit Rollback-Flag)

**Stand:** 2026-08-24 · Vorgänger: `HANDOVER-llm-reranker.md`,
`docs/superpowers/specs/2026-08-24-two-signal-reranker-design.md`

## Problem

Stufe 1 und Stufe 2 haben gezeigt: der Zwei-Signal-LLM-Reranker löst alle
vier Dauer-Misses (Tuning-Hit-Rate 91 %→100 %) bei fast unveränderter
Abstention (100 %→93 %, ein zusätzlicher Fehlalarm bei 14 Off-Topic-Fragen).
Drei Kaskaden-Heuristiken (Cross-Encoder-Score-Schwelle, Score-Abstand,
LLM/Cross-Encoder-Einigkeit) wurden mit echten Messungen widerlegt — keine
trennt zuverlässig zwischen echten Treffern und dem einen Fehlalarm. Ersatz
(LLM-Reranker läuft bei jeder Anfrage) ist die einzige belegte Option.

Diese Spec beschreibt die tatsächliche Integration in `app/retrieval.py`
und `app/main.py` — bisher bewusst unverändert gehalten, um das Experiment
reversibel zu halten. Das ändert sich jetzt: echte Produktionsänderung,
live für Nutzer der Demo.

## Entscheidungen aus dem Brainstorming

- **Ersatz, keine Kaskade.** Cross-Encoder-Signale (Score-Höhe, Score-
  Abstand Top1/Top2, Übereinstimmung mit dem LLM) trennen den einen
  gemessenen Fehlalarm nicht zuverlässig von echten Treffern — siehe
  Handover-Abschnitt „Integrationsentscheidung: drei Kaskaden-Heuristiken
  widerlegt".
- **Rollback per Config-Flag, nicht Git-Revert.** `settings.use_llm_reranker:
  bool = True`. Bei `False` unverändertes Cross-Encoder-Verhalten.
- **`Retriever.retrieve()` wird `async def`.** Ein Retrieval-Pfad statt
  zwei parallelen Implementierungen. Alle Aufrufer (Produktion, Eval-
  Skripte, Tests) werden entsprechend migriert.
- **Tagesbudget bleibt bei 200.000**, trotz gemessener ~74 % geringerer
  Kapazität (siehe unten) — Portfolio-Demo ohne Umsatzdruck, Rate-Limit
  (50/Tag/IP) war ohnehin der engere Deckel pro Nutzer.
- **CI-Gate testet künftig den LLM-Pfad**, nicht mehr nur den
  Cross-Encoder — das Gate soll prüfen, was tatsächlich live läuft.

## Architektur

### `app/retrieval.py`

`Retriever.retrieve()` wird `async def retrieve(self, query, top_k=5,
audience=None, client=None) -> tuple[list[RetrievedDoc], int]`. Zweiter
Rückgabewert: verbrauchte Tokens (0 im Cross-Encoder-Zweig).

`Retriever.__init__` bekommt einen neuen Parameter `use_llm_reranker: bool
= settings.use_llm_reranker`, gespeichert als `self.use_llm_reranker` —
gleiches Muster wie die bestehenden `sim_threshold`/`bm25_threshold`/
`rerank_threshold`-Parameter (Konstante als Default, pro Instanz
überschreibbar). **Wichtig:** `retrieve()` verzweigt auf
`self.use_llm_reranker`, nicht auf das globale `settings.use_llm_reranker`
direkt — sonst würden alle bestehenden Tests in `tests/test_retrieval.py`,
die einen Mock-`reranker=`-Callable übergeben, um gezielt den Cross-
Encoder-Zweig zu testen, beim neuen globalen Default (`True`) still-
schweigend auf den LLM-Zweig umspringen, obwohl sie das nie erwarten. Die
bestehenden Tests übergeben weiterhin nur `reranker=` (Cross-Encoder-
Callable) und müssen zusätzlich `use_llm_reranker=False` an
`make_retriever`/`Retriever(...)` übergeben, um beim alten Verhalten zu
bleiben.

Verzweigung nach `self.use_llm_reranker`:

- **`True` (Standard):** Kandidatenmenge per Union-Fix
  (`_vector_candidates` + `_bm25_candidates`, vereinigt statt RRF-Top-n-
  Cut — Logik aus `eval/llm_reranker.py::two_signal_candidates` wandert
  hierher bzw. wird von hier wiederverwendet). Reranking per
  `llm_two_signal_rerank(query, docs, client)`. Gate: `top1_confidence <
  LLM_CONFIDENCE_THRESHOLD` (neue Konstante, Wert `8.5`, siehe Stufe-2-
  Messung im Handover) → leer zurückgeben. `used_fallback=True` wird wie
  eine zu niedrige Konfidenz behandelt (konservativ abstinieren), nicht
  wie ein Erfolg.
- **`False` (Rollback):** unverändertes Verhalten — RRF-Fusion + Cross-
  Encoder + `RERANK_THRESHOLD`, exakt wie vor dieser Spec.

`Retriever.__init__` lädt den Cross-Encoder (`_default_reranker()`) nur
noch, wenn `reranker` nicht explizit übergeben wurde **und**
`self.use_llm_reranker` (siehe oben) `False` ist — sonst entfällt der
Modell-Load und die `HF_TOKEN`-Abhängigkeit für den Standardfall komplett.
Bestehende Test-Konstruktion mit explizitem `reranker=`-Argument bleibt
unverändert funktionsfähig (Override sticht immer, unabhängig von
`use_llm_reranker`).

`llm_two_signal_rerank` in `eval/llm_reranker.py` gibt künftig ein
4-Tupel zurück: `(ranking, confidence, used_fallback, tokens)` —
`tokens = response.usage.input_tokens + response.usage.output_tokens`,
analog zu `app/llm.py::rewrite_query`.

### `app/main.py`

Der bestehende Aufruf

```python
docs = app.state.retriever.retrieve(standalone, audience=audience)
```

wird zu

```python
docs, rerank_tokens = await app.state.retriever.retrieve(
    standalone, audience=audience, client=client)
if rerank_tokens:
    app.state.budget.spend(rerank_tokens)
```

`client` ist an dieser Stelle im bestehenden Code bereits erzeugt (vor dem
`rewrite_fn`-Aufruf) — keine neue Client-Erzeugung nötig.

### Eval-Infrastruktur

`eval/run_eval.py`s `hit_rate_at_k`, `hit_rate_at_k_with_audience`,
`abstention_rate` werden `async def`, rufen `await retriever.retrieve(...)`
auf. Der `__main__`-Block (inkl. `--gate`) wird entsprechend mit
`asyncio.run(...)` verkabelt und braucht ab jetzt einen echten Client
(analog zu `_rewrite_questions`, das bereits `get_client()` nutzt).

`eval/run_llm_reranker_experiment.py` und `eval/run_llm_reranker_stufe2.py`
werden an das neue 4-Tupel von `llm_two_signal_rerank` angepasst
(zusätzlicher `tokens`-Rückgabewert, an den Aufrufstellen entpackt, aber
nicht weiter verwendet — diese Skripte tracken kein Produktionsbudget).

`eval/judge.py:90` (`judge_one`, genutzt vom `quality-gate`-CI-Job) ruft
`retriever.retrieve(standalone, top_k=5)` bereits aus einer `async def`
heraus auf, aber unawaited — bekommt `await` und entpackt das neue
`(docs, tokens)`-Tupel (Tokens hier ebenfalls nicht weiterverwendet, der
Judge trackt kein Produktionsbudget).

### CI (`.github/workflows/ci.yml`)

`eval-gate`-Job: `ANTHROPIC_API_KEY: ${{ secrets.ANTHROPIC_API_KEY }}` als
Env-Var für den `python -m eval.run_eval --gate`-Schritt ergänzen (Secret
existiert bereits, wird vom `quality-gate`-Job verwendet). Der
`hf-models`-Cache-Step wird für diesen Job überflüssig (kein Cross-Encoder-
Load mehr bei `use_llm_reranker=True`) — bleibt aber unschädlich, wenn er
nicht entfernt wird; Entfernen ist Aufräumen, kein funktionales Muss.

## Testmigration

Jede Testdatei, die `Retriever.retrieve()` direkt oder über
`eval/run_eval.py`s Hilfsfunktionen aufruft, braucht `await` und
`async def test_...` (pytest, `asyncio_mode=auto` — kein Decorator nötig):

- `tests/test_retrieval.py` — größter Block, alle bestehenden
  `retriever.retrieve(...)`-Aufrufe. Bestehende Cross-Encoder-Mock-Tests
  (über `make_retriever`, das `reranker=` an `Retriever(...)` durchreicht)
  bekommen zusätzlich `use_llm_reranker=False`, damit sie weiterhin den
  Cross-Encoder-Zweig testen — sonst nur syntaktisch async (`await` +
  `async def test_...`).
- `tests/test_eval.py:7-13` — `StubRetriever.retrieve(query, top_k=5)`
  gibt aktuell eine reine Liste zurück, wird `async def` und gibt
  `(docs, 0)` zurück (Aufrufer entpacken jetzt ein Tupel).
- `tests/test_api.py:18-23` — `FakeRetriever.retrieve(query, top_k=5,
  audience=None)` gibt aktuell `self.docs` zurück, wird `async def` und
  gibt `(self.docs, 0)` zurück.
- `tests/test_judge.py:95-99` — `FakeRetriever.retrieve(query, top_k=5)`,
  gleiche Anpassung wie die beiden Stubs oben.
- `tests/test_llm_reranker.py`, `tests/test_run_llm_reranker_experiment.py`,
  `tests/test_run_llm_reranker_stufe2.py` — an das neue 4-Tupel von
  `llm_two_signal_rerank` anpassen.

Neue Tests für diese Spec:
- `Retriever.retrieve()` mit `use_llm_reranker=True`: Union-Kandidaten
  statt RRF-Cut, Gate bei `top1_confidence < LLM_CONFIDENCE_THRESHOLD`,
  `used_fallback=True` führt zu leerem Ergebnis, Tokens werden
  zurückgegeben.
- `Retriever.retrieve()` mit `use_llm_reranker=False`: unverändertes
  Cross-Encoder-Verhalten, Tokens `== 0`.
- `Retriever.__init__` lädt den Cross-Encoder nicht, wenn
  `use_llm_reranker=True` und kein `reranker=`-Override übergeben wurde.
- `app/main.py`: `budget.spend()` wird mit den Reranker-Tokens aufgerufen.

## Fehlerbehandlung

`used_fallback=True` (Parse-Fehler trotz Fence-Stripping-Fix) wird im
Produktionspfad wie eine zu niedrige Konfidenz behandelt — leeres Ergebnis,
keine Antwort, kein stiller Fehltreffer. Das ist strenger als during Stufe
1/2 (dort nur geloggt), weil hier ein echter Nutzer die Antwort sieht.

## Nicht Teil dieser Spec

- Keine Änderung an `daily_token_budget` (bleibt 200.000, Entscheidung
  siehe oben).
- Keine weitere Kaskaden-Forschung — als widerlegt dokumentiert.
- Kein Entfernen des Cross-Encoder-Codepfads oder des `rerank_model`-
  Settings-Felds — bleibt als Rollback-Pfad bestehen.
