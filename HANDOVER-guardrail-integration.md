# Handover: Guardrail-Integration aus dem Handover Brief Generator

Stand: 2026-08-21. Geschrieben aus einer Session im Schwesterprojekt
`C:\Users\Marco\02_Portfolio\Handover Brief Generator` — dieses Dokument
übergibt den dort entwickelten Plan an eine Session in diesem Repo.

Update 2026-08-21: Stufe A und Stufe B sind umgesetzt (siehe README); dieses Dokument beschreibt den ursprünglichen Plan.

## Kontext: die beiden Projekte

- **Dieses Repo (chrono24-chatbot):** RAG-Chatbot über die Chrono24-Hilfeseiten.
  FastAPI + SSE-Streaming, Hybrid-Retrieval (BM25 + Vektor + RRF + Cross-Encoder,
  88 % Hit-Rate@5), Offline-Eval per LLM-Judge (100 % faithful), Guards
  (Rate-Limit, Token-Budget). Quellenangaben als `[1]`–`[5]`-Marker.
- **Handover Brief Generator:** erzeugt aus einem Support-Ticketverlauf bei der
  Übergabe ein Briefing mit Zitierpflicht. Kernidee: **Faithfulness by
  construction** — das LLM liefert Claims mit `source_lines`, ein
  deterministischer Validator (`src/validate.py`) prüft jede Aussage per
  Token-Overlap gegen die zitierten Zeilen (PASS ≥ 0.5 / WEAK / FAIL; FAIL
  führt zu Retry und danach Ablehnung statt stillem Ausliefern).

## Der Befund, der die Integration motiviert

Der Chatbot erzwingt seine Zitierpflicht **nur im Prompt**
([app/llm.py](app/llm.py), `SYSTEM_PROMPT`: "Belege Aussagen mit den
Quellennummern … Erfinde nichts") — zur Laufzeit prüft nichts, ob eine
Antwort tatsächlich durch die zitierten Dokumente gedeckt ist. Faithfulness
wird nur offline per LLM-Judge gemessen (`eval/judge.py`), und der Judge ist
derselbe Modelltyp wie der Bot (im README ehrlich als Bias-Risiko notiert).
Genau diese Lücke füllt der deterministische Validator des Schwesterprojekts:
zur Laufzeit, ohne zusätzlichen LLM-Call, unbestechlich.

## Stufe A: Laufzeit-Validator für Bot-Antworten (zuerst umsetzen)

Ziel: Nach jedem gestreamten Antwortende wird die Antwort deterministisch
gegen die zitierten Quellen geprüft; das Ergebnis geht als eigenes SSE-Event
ans Frontend und wird dort als Ampel angezeigt.

Umsetzungsskizze:

1. **Portieren:** `score_overlap` + `_tokenize` aus
   `..\Handover Brief Generator\src\validate.py` in ein neues Modul
   `app/faithcheck.py` übernehmen (Regex-Tokenizer `[a-zäöüß0-9]+`,
   Schwelle 0.5). Die `L\d+`-Filterung dort durch eine `\[\d+\]`-Filterung
   für Zitatmarker ersetzen.
2. **Antwort zerlegen:** fertige Antwort in Sätze splitten; pro Satz
   `[n]`-Zitate parsen. Satz mit Zitat → Overlap gegen den Text des
   zitierten `RetrievedDoc` (Index aus dem `docs`-Array des Requests);
   Satz mit Tatsachenform ohne Zitat → WEAK-Markierung.
   Floskeln/Verweigerungssatz ("Dazu finde ich nichts …") überspringen.
3. **Event:** in [app/main.py](app/main.py) nach dem Antwort-Stream ein
   Event `{"type": "validation", "sentences": [{text, status, score,
   sources}]}` senden (analog zum bestehenden `sources`-Event).
4. **UI:** [static/app.js](static/app.js) rendert pro Satz eine kleine
   Ampel (✅ / 🟡), plus Einzeiler-Legende wie im Schwesterprojekt:
   "✅ Wortlaut deckt sich mit der Quelle · 🟡 paraphrasiert oder ohne
   Zitat". Kein Blockieren der Antwort in Stufe A — nur sichtbar machen.
5. **Tests:** Unit-Tests für Satz-Splitting, Zitat-Parsing, Overlap-Status;
   ein API-Test, dass das `validation`-Event kommt (bestehendes
   Test-Muster in [tests/test_api.py](tests/test_api.py) mit
   `create_app(answer_fn=…)`).

Bewusste Grenzen (dokumentieren, nicht verstecken): Token-Overlap ist ein
grobes Maß — Paraphrasen erscheinen gelb, obwohl sie inhaltlich stimmen
können. Das ist im Schwesterprojekt genauso und dort Teil der Erzählung:
der Validator misst, statt zu vertrauen.

## Stufe B: Eskalation mit Handover-Briefing (danach)

Ziel: Wenn der Bot nicht weiterweiß (heute endet das in der Sackgasse
"Dazu finde ich nichts in den Chrono24-Hilfeseiten.") oder auf Knopfdruck,
wird der Chatverlauf an einen menschlichen Agenten übergeben — mit einem
geprüften Briefing statt rohem Verlauf.

Umsetzungsskizze:

1. **Endpoint** `POST /api/handover`: nimmt die Chat-History, vergibt
   Zeilen-IDs (`M01`, `M02`, …), ruft Claude mit dem Extractor-Schema des
   Schwesterprojekts (`situation/history/sentiment/open_question/claims`,
   je mit `source_lines`) auf — Prompt-Vorlage:
   `..\Handover Brief Generator\src\extract.py`.
2. **Validator drüber** (aus Stufe A vorhanden): FAIL → ein Retry mit
   Fehlerhinweis, danach ablehnen — Orchestrator-Logik siehe
   `..\Handover Brief Generator\src\orchestrator.py` (MAX_ATTEMPTS = 2).
3. **UI:** Button "An Support übergeben" im Chat; Briefing als Karte mit
   Ampel pro Aussage (Vorbild: `_render_briefing_card` in
   `..\Handover Brief Generator\app.py`).
4. Guards beachten: der Handover-Call kostet Tokens → ins Tagesbudget
   (`app/guards.py`) einbuchen, Rate-Limit gilt über den bestehenden
   Decorator.

Damit ist die Produktstory komplett: Bot beantwortet, was er belegen kann;
was nicht, übergibt er an einen Menschen — mit geprüftem Briefing.

## Optional, Stufe C (im anderen Repo, nicht hier)

Im Handover Brief Generator könnte ein Ticket (T-2087 bietet sich an — die
Rückgabefrist-Frage ist eine echte FAQ-Frage) den Akt 1 als
FAQ-Bot-Gespräch erzählen: Kunde fragt den Bot, Bot antwortet belegt,
Eskalation an den Menschen, weil der Bot nicht weiterweiß. Reine
Daten-Kosmetik am Ticket-JSON, kein Code. Erst sinnvoll, wenn A/B stehen.

## Portfolio-Verzahnung

- README beider Projekte: je ein Querverweis-Absatz ("Derselbe
  deterministische Faithfulness-Validator läuft in beiden Projekten").
- marco-os (`data/projects.js`): beide Einträge aufeinander verweisen
  lassen — ein Guardrail-Muster, mehrfach angewandt, wirkt wie Methode.

## Empfohlene Reihenfolge

1. Stufe A (klein, rein additiv, kein LLM-Call): `app/faithcheck.py` +
   SSE-Event + UI-Ampel + Tests.
2. Stufe B: `/api/handover` + Briefing-Karte.
3. READMEs/Portfolio verzahnen; Stufe C optional.
