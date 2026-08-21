# Design: Handover-Briefing (Stufe B der Guardrail-Integration)

Datum: 2026-08-21 · Status: vom Nutzer freigegeben (Abschnitte 1–4 einzeln bestätigt)
Kontext: `HANDOVER-guardrail-integration.md` (Stufe A — Laufzeit-Faithfulness-Check — ist umgesetzt, Commit `ce74d88`).

## Ziel

Wenn der Bot nicht weiterweiß oder der Nutzer es wünscht, wird der Chatverlauf
an einen menschlichen Support-Agenten übergeben — als **geprüftes Briefing**
statt rohem Verlauf. Das LLM extrahiert ein strukturiertes Briefing mit
Zeilen-Zitaten; der deterministische Validator aus Stufe A prüft jede Aussage
per Token-Overlap gegen die zitierten Chat-Zeilen. FAIL → ein Retry mit
Fehlerhinweis, danach ehrliche Ablehnung statt stillem Ausliefern.

Gewählter Ansatz: **schlanke Portierung** aus dem Schwesterprojekt
`..\Handover Brief Generator` (`src/extract.py`, `src/orchestrator.py`,
`src/validate.py`). Kein Shared-Package (YAGNI), kein SSE (Briefing entsteht
als Ganzes, Retry-Schleife macht Streaming sinnlos).

## Entscheidungen (vom Nutzer bestätigt)

- **Trigger:** Button "An Support übergeben" immer sichtbar sobald der Verlauf
  eine Bot-Antwort enthält; nach einer NOT-FOUND-Antwort zusätzlich ein
  aktives Übergabe-Angebot.
- **Rejected-UX:** ehrlich ablehnen — Karte "Briefing nicht belegbar", kein
  Rendern unbelegter Briefing-Felder.
- **Kein Precomputed-Fallback** (existiert hier nicht): Fehler → HTTP-Fehler
  mit ehrlicher Meldung.

## 1. Architektur & Datenfluss

Neues Modul **`app/handover.py`**:

- `build_lines(messages) -> list[dict]` — Chat-History (user/assistant) bekommt
  Zeilen-IDs `M01, M02, …`; Elemente `{"id", "actor", "text"}`,
  actor `"Kunde"` / `"Bot"`.
- `SYSTEM_PROMPT` — deutsches Extractor-Schema wie im Schwesterprojekt
  (`situation/history/sentiment/open_question/claims`, je mit `source_lines`),
  angepasst von "Support-Ticketverlauf" auf "Chatverlauf zwischen Kunde und
  FAQ-Bot eines Luxusuhren-Marktplatzes".
- `build_prompt(lines, previous_failure_note)` und `parse_response(raw)`
  (Markdown-Fence-Strip, Pflichtfelder-Check) — direkte Ports.
- `generate_briefing(messages, client) -> dict` — Orchestrator:
  `MAX_ATTEMPTS = 2`; pro Versuch Extract → `validate_claims`; FAIL-Claims →
  Retry mit Fehlerhinweis ("Vorherige Antwort hatte unbelegte Aussage(n): …");
  nach zwei Fehlversuchen `{"status": "rejected", "failed_claims": […]}`.
  Rückgabe enthält `tokens` (Summe input+output aller Versuche) fürs Budget.
  Exceptions propagieren zum Endpoint.

**`app/faithcheck.py`** erweitert:

- `validate_claims(claims, lines_by_id) -> list[SentenceCheck]` — Port von
  `validate_claim`/`normalize_briefing_to_claims`: Overlap des Claim-Texts
  gegen die per `source_lines` referenzierten `M`-Zeilen, gleiche
  0.5-Schwelle; fehlende/leere `source_lines` oder unbekannte Zeilen-ID →
  FAIL (Score 0).
- `_tokenize` filtert zusätzlich `m\d+`-Tokens (analog `L\d+` im Original),
  damit Zeilen-IDs im Claim-Text den Score nicht verfälschen.

Datenfluss: Frontend `POST /api/handover` mit `{messages}` → Zeilen-IDs →
Claude (Haiku, `settings.model`, bestehender Async-Client) → deterministische
Validierung → JSON-Response → Briefing-Karte im Chat.

## 2. API-Endpoint & Guards

`POST /api/handover` in `app/main.py`:

- Request: `HandoverRequest(messages: list[ChatMessage])` — gleiche Grenzen
  wie Chat (max 20 Nachrichten, 4000 Zeichen), aber **ohne** die Regel
  "letzte Nachricht muss vom Nutzer sein" (Übergabe nach Bot-Antwort ist der
  Normalfall). Mindestens 1 Nachricht.
- Guards: `@limiter.limit("3/minute;10/day")` (strenger als Chat — jeder Call
  ist mindestens ein LLM-Call, bei Retry zwei). Budget-Check vorab
  (`remaining() <= 0` → 429 "Demo-Budget für heute erschöpft"); nach dem Lauf
  `budget.spend(tokens)`.
- Response 200:

  ```json
  {
    "status": "ok",
    "briefing": {"situation": {"text": "…", "source_lines": ["M01"]}, "…": "…"},
    "validation": [{"text": "…", "status": "PASS", "score": 0.83, "sources": ["M01"]}],
    "lines": [{"id": "M01", "actor": "Kunde", "text": "…"}]
  }
  ```

  `lines` wird mitgeliefert, damit das Frontend Quellzeilen-Tooltips rendern
  kann, ohne die ID-Vergabe zu duplizieren.
- `status: "rejected"` ist **kein** HTTP-Fehler: 200 mit `failed_claims` —
  die Ablehnung ist Feature, nicht Fehler.
- Fehlerpfad: Exception im Extract (kaputtes JSON nach Fence-Strip,
  API-Fehler) → HTTP 502 "Briefing-Erstellung fehlgeschlagen — bitte erneut
  versuchen." Geloggt per `logger.exception`. Kein stilles Degradieren.

## 3. Frontend

`static/app.js` + `static/index.html` + `static/style.css`:

- **Button** "An Support übergeben" neben dem Senden-Button; sichtbar sobald
  `history` eine Bot-Antwort enthält; während des Calls disabled
  ("Übergebe …").
- **Auto-Angebot:** endet eine Bot-Antwort mit der NOT-FOUND-Formulierung,
  erscheint darunter "Der Bot weiß hier nicht weiter — an einen Menschen
  übergeben?" mit demselben Handler.
- **Briefing-Karte** `addBriefingCard(result)` (Vorbild
  `_render_briefing_card` im Schwesterprojekt):
  - Kopf "Übergabe-Briefing" + Status-Badge (✅ geprüft / ⛔ abgelehnt).
  - Felder Situation, Verlauf, Stimmung (Label + wörtliches Zitat), Offene
    Frage — je mit Ampel aus `validation` und Quellzeilen-IDs als
    `title`-Tooltip mit dem Zeilentext; Claims als Liste, gleiche Logik.
  - `rejected`: keine Briefing-Felder — Karte "Briefing nicht belegbar — der
    Validator hat n Aussage(n) abgelehnt. Der Roh-Verlauf würde übergeben."
    (Demo: keine echte Weiterleitung; die Karte sagt das ehrlich dazu.)
  - Fehler (429/502): Meldung als Bot-Nachricht, kein Kartentorso.
  - Ausschließlich `textContent`/`createElement` — kein `innerHTML`.
  - Legenden-Einzeiler wie im Faithfulness-Panel.
- CSS: `.briefing`-Karte im Panel-Stil, Badge-Farben, Claim-Zeilen.

## 4. Tests & Fehlerbehandlung

`tests/test_handover.py` (FakeClient-Muster wie `tests/test_judge.py`):

- `build_lines`: IDs `M01…`, Actor-Zuordnung.
- `parse_response`: Markdown-Fence; fehlende Pflichtfelder → `ValueError`.
- `generate_briefing`: (a) valide → `ok`, ein Call; (b) FAIL dann valide →
  `ok`, zwei Calls, Fehlerhinweis im zweiten Prompt; (c) zweimal FAIL →
  `rejected` mit `failed_claims`; (d) Token-Summe über Versuche korrekt.

`tests/test_faithcheck.py` erweitert: `validate_claims` PASS/WEAK/FAIL-Grenzen,
fehlende `source_lines` → FAIL, unbekannte ID → FAIL, `m\d+`-Tokenfilter.

`tests/test_api.py` erweitert: Happy Path (200, `ok`, `validation` + `lines`),
Budget belastet, Budget leer → 429, Extract-Exception → 502, Rate-Limit
3/minute.

## Grenzen (dokumentieren, nicht verstecken)

Token-Overlap bleibt ein grobes Maß: korrekt paraphrasierte Claims erscheinen
WEAK. Der Judge-Bias-Hinweis aus dem README gilt hier nicht — die Validierung
ist deterministisch, aber die **Extraktion** bleibt LLM-Arbeit: der Validator
garantiert Belegbarkeit, nicht Vollständigkeit des Briefings.

## Nach Umsetzung

README-Sektion (Produktstory: "Bot beantwortet, was er belegen kann; was
nicht, übergibt er an einen Menschen — mit geprüftem Briefing"),
Querverweis-Absätze in beiden READMEs, marco-os-Verzahnung
(`data/projects.js`). Stufe C (Ticket T-2087 im Schwesterprojekt) optional,
separates Repo.
