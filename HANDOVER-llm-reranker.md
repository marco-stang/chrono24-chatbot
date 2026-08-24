# Handover: LLM-Reranker löst alle drei Dauer-Misses — braucht aber ein Zwei-Signal-Design

**Stand:** 2026-08-23 · `main` bei `2a7907a`, **elf Commits lokal, nicht gepusht**
(zuletzt gepusht vermutlich `db0f968` oder früher) · 163 Tests grün · ruff sauber ·
Working Tree clean

Vorgänger-Handover: `HANDOVER-eval-set-groesse.md` (Eval-Set-Größe, teilweise erledigt),
`HANDOVER-guardrail-integration.md`, `HANDOVER-query-varianten-eval-gate.md`. Dieses
Dokument behandelt genau einen offenen Punkt: das LLM-Reranker-Experiment aus README
(„LLM als Reranker: Intention statt Nomen") ist gemessen, aber nicht produktionsreif.

---

## Das Ergebnis in einem Satz

Ein Haiku-Call als finaler Reranker (Rangfolge statt Score) löst alle drei Misses, die
die gesamte bisherige Ablationsreihe nicht geknackt hat — **Tuning-Hit-Rate@5 100 %
(33/33)** —, reißt aber die Abstention-Rate von 100 % auf **36 % (5/14)** ein, weil
Rangfolge ein relatives Signal ist und keine absolute Konfidenz trägt.

| Metrik | Vorher (Finetune-Cross-Encoder) | LLM-Reranker (Rangfolge) |
|---|---|---|
| Tuning-Hit-Rate@5 | 91 % (30/33) | **100 % (33/33)** |
| Holdout-Hit-Rate@5 | 100 % (15/15) | 100 % (15/15) |
| Abstention-Rate | 100 % (14/14) | **36 % (5/14)** |

---

## Was schon gemessen ist (bitte nicht wiederholen)

1. **Voraussetzung: Kandidaten-Union-Fix.** Der bestehende RRF-Top-n-Cut (`app/retrieval.py`,
   `candidates = sorted(fused.items(), ...)[:n]`) schneidet faq-0098 (BM25-Rang 8,
   Vektor-Rang 207) und info-escrow-0007 (BM25-Rang 6, Vektor-Rang 426) schon *vor* jedem
   Reranking raus — verifiziert direkt. Ohne die Kandidatenmenge auf die Vereinigung beider
   Top-10-Listen zu erweitern (`_dedupe_ranking(vector_ranking + bm25_ranking)` statt des
   RRF-Fusions-Cutoffs), sieht **kein** Reranker diese Dokumente je, auch kein LLM.
2. **LLM mit unabhängigen 0–10-Scores pro Kandidat:** löst beide Rollen-Misses (Score 9–10,
   Platz 1), erzeugt aber einen neuen Miss (DAC7/faq-0162) — bei Themenclustern mit
   mehreren ähnlich passenden Geschwister-FAQs vergibt Haiku gehäuft Höchst-Scores
   (4-facher Gleichstand bei 10), die eigentlich korrekte, allgemeine Antwort verliert
   gegen ein spezifischeres Geschwister. Derselbe Fehlermodus wie der alte Certified-Miss
   (faq-0033 vs. faq-0048), nur durch den größeren Kandidatenpool neu ausgelöst.
3. **LLM mit expliziter Rangfolge statt Scores** (JSON-Array der Dokumentnummern,
   absteigend nach Relevanz) erzwingt Tiebreaks strukturell — löst zusätzlich DAC7, und
   Certified landet auf Platz 5 von 15 (zählt als Hit@5). **Alle drei Misses gelöst.**
4. **Robustheitsproblem gefunden und behoben:** Haiku lieferte anfangs oft nur
   Teil-Rangfolgen (z. B. 2 von 16 Indizes) trotz expliziter Anweisung. Fix: exakte
   Kandidatenzahl im System-Prompt nennen, Beispiel-Array mitgeben, `max_tokens` 200 → 400.
   Seitdem keine Parsing-Warnungen mehr in allen 66 Testläufen.

**Bereits ausgeschlossen als „nur nicht genug Kandidaten":** der Kandidaten-Union-Fix
allein (ohne LLM, mit dem Basis-Cross-Encoder) wurde separat gemessen — löst nichts,
siehe README-Ablationstabelle. Es ist die Kombination aus Kandidaten-Fix *und* echtem
Sprachverständnis, die den Unterschied macht.

---

## Warum das kein einfacher Merge ist

Die Abstention-Regression ist kein Bug, sondern eine Eigenschaft von Rangfolgen: der
beste von zehn komplett themenfremden Kandidaten bekommt denselben hohen Rang-Score wie
ein echter Treffer. Das bestehende `RERANK_THRESHOLD`-Gate setzt „hoher Score = sicher
relevant" voraus — das gilt für Rangfolgen nicht mehr.

**Sortier-Signal und Konfidenz-Signal sind zwei verschiedene Dinge.** Ein Prompt, der
beides liefert, ist der naheliegende nächste Schritt:

- Rangfolge fürs Sortieren (wie jetzt).
- Zusätzlich eine **absolute** Einschätzung — z. B. „ist der bestplatzierte Kandidat
  tatsächlich relevant, ja/nein" oder ein zweiter 0–10-Wert mit expliziter Verankerung
  („0 = beantwortet die Frage überhaupt nicht, auch nicht der beste Kandidat"), getrennt
  von der relativen Sortierung.
- Alternative: zweistufig — erst Rangfolge, dann ein zweiter, kleinerer Call nur für den
  Top-1-Kandidaten mit einer echten Ja/Nein-Frage („beantwortet [1] die Frage?"). Kostet
  einen zweiten Call, aber nur auf den bereits sortierten Top-Kandidaten, nicht auf alle.

Beide Varianten sind ungetestet — das ist die eigentliche Aufgabe dieses Handovers.

---

## Kosten- und Latenz-Realität

- **Ø 1,1–1,3 s zusätzliche Latenz pro Anfrage mit Kandidaten** (gemessen über 66 Calls,
  `claude-haiku-4-5`, `temperature=0`).
- **Echte, laufende API-Kosten pro Anfrage** — anders als der bestehende Rewrite-Call, der
  bei deutschen Fragen ohne Chatverlauf komplett entfällt (`app/llm.py::rewrite_query`),
  würde dieser Call bei praktisch jeder On-Topic-Anfrage laufen. Kein Vergleich mit dem
  Finetune-Ansatz (einmaliger Trainingsaufwand, danach lokal und kostenlos).
- Für diese Messrunde: ~130 Haiku-Calls insgesamt (Score-Variante + Rank-Variante + Debug-
  Checks + voller Eval-Lauf), Session-Gesamtkosten laut Cost-Tracking ~20 $ inklusive aller
  anderen Experimente dieser Sitzung (Finetune-Training separat, lokal/kostenlos).
- Bei Live-Traffic: **Tagesbudget-Guard (`daily_token_budget`, aktuell 200.000) würde durch
  einen LLM-Reranker-Call pro Anfrage deutlich schneller aufgebraucht** — das Budget wurde
  für Query-Rewrite (selten) + Antwort (immer) dimensioniert, nicht für einen dritten Call
  pro Anfrage. Vor jeder Integration neu kalkulieren.

---

## Vorgeschlagene nächste Schritte

1. **Zwei-Signal-Prompt entwerfen und isoliert testen** — erst an den vier bekannten
   Problemfällen (faq-0098, info-escrow-0007, faq-0033, faq-0162) plus ein paar Off-Topic-
   Stichproben, billig, bevor der volle Eval-Lauf folgt (Muster aus dieser Sitzung: erst
   4 Calls, dann 66).
2. **Schwellen-Rekalibrierung für das neue Konfidenz-Signal**, analog zur Methodik beim
   Cross-Encoder-Finetune (on-topic-Minimum vs. off-topic-Maximum, `RERANK_THRESHOLD`
   entsprechend setzen) — aber jetzt auf dem neuen, separaten Konfidenz-Wert, nicht auf dem
   Rang-Score.
3. **Vollen Eval-Lauf** (Tuning + Holdout + Abstention) mit dem neuen Design, gegen die
   Zahlen oben vergleichen.
4. **Erst danach über Integration entscheiden** — inklusive Tagesbudget-Neukalkulation und
   der Frage, ob der LLM-Reranker den Cross-Encoder ersetzt oder als Kaskade dahinter läuft
   (z. B. Cross-Encoder für die Mehrheit der Anfragen, LLM nur wenn der Cross-Encoder
   unsicher ist — würde Kosten senken, ist aber ein dritter, noch ungetesteter Entwurf).

---

## Fallstricke aus dieser Sitzung

- **Scratchpad-Skripte sind flüchtig** (Session-gebundenes Temp-Verzeichnis) und nicht Teil
  dieses Repos — die hier beschriebenen Prompts/Ergebnisse müssen aus diesem Dokument
  rekonstruiert werden, nicht aus altem Code. Kernbestandteile: System-Prompt mit exakter
  Kandidatenzahl + Beispiel-Array (siehe „Robustheitsproblem" oben), `max_tokens=400`,
  `temperature=0`, Kandidatenliste als `[i] {title}\n{text}`-Block.
- **`Retriever.retrieve()` monkeypatchen statt `app/retrieval.py` ändern**, solange nur
  gemessen und nicht integriert wird — hält Experimente reversibel und das Repo sauber.
  Beispiel-Pattern: private Helfer `_vector_candidates`/`_bm25_candidates` der aktuellen
  SQLite-Engine wiederverwenden, nur die Kandidaten-Auswahl (Union statt Top-n-Cut) und den
  Reranker-Callable ersetzen.
- **`rerank_threshold=-999` beim Messen der reinen Rangfolge setzen**, sonst verfälscht die
  (für das neue Signal falsch kalibrierte) alte Schwelle die Hit-Rate-Messung selbst.
- **Parallele Sessions können denselben Working Tree verändern** (diese Sitzung erlebte
  genau das: ein uncommitteter `retrieval.py`-Diff verschwand, weil eine zweite Sitzung
  die Datei zwischenzeitlich komplett neu geschrieben hat). Vor jedem größeren Edit
  `git status`/`git log` prüfen, wenn eine Sitzung länger pausiert oder ein langer
  Hintergrund-Job lief.

---

## Stufe 1: erste Messung des Zwei-Signal-Designs (2026-08-24)

Umgesetzt nach `docs/superpowers/specs/2026-08-24-two-signal-reranker-design.md`
und `docs/superpowers/plans/2026-08-24-two-signal-reranker-stufe1.md` — neues
Modul `eval/llm_reranker.py` (Prompt, Parsing, Kandidaten-Union,
`two_signal_candidates`-Wrapper) plus `eval/run_llm_reranker_experiment.py`
für den isolierten Stufe-1-Lauf. `app/retrieval.py` unverändert.

**Fallstrick beim ersten Lauf:** Haiku verpackt seine JSON-Antwort trotz
expliziten Verbots im Prompt regelmäßig in ` ```json ... ``` `-Markdown-Fences.
`json.loads` scheiterte daran bei **jedem** der 8 ersten API-Calls — der
Parse-Fallback griff durchgehend, keiner der Calls lieferte ein echtes
Signal, das Geld für den ersten Lauf war verloren. Fix: `_strip_code_fence()`
in `eval/llm_reranker.py` vor `json.loads`, 5 neue Tests. Für jeden künftigen
LLM-Reranker-Prompt einplanen, nicht nur für dieses Experiment.

**Ergebnis nach dem Fix** (4 bekannte Problemfälle + 4 Off-Topic-Stichprobe):

| Fall | Ergebnis |
|---|---|
| faq-0098 (Käuferschutz-Gebühr) | OK, confidence 9.0 |
| faq-0162 (DAC7) | OK, confidence 9.0 |
| faq-0033 (Certified) | OK, confidence 9.0 |
| info-escrow-0007 (Escrow-Versand) | OK, confidence 9.0 |
| "Wie backe ich einen Hefezopf?" | Gate zu (Stufe 1, vor jedem LLM-Call) |
| "Welches Wetter … Karlsruhe?" | Gate zu (Stufe 1, vor jedem LLM-Call) |
| "Omega Seamaster … Wert?" (domänennah) | Kandidat durch, confidence **9.0** |
| "eBay-Rückgabepolitik vs. Chrono24" (domänennah) | Kandidat durch, confidence 2.0 |

Alle vier Dauer-Misses bestätigt gelöst, mit klar hoher Konfidenz. Die
Off-Topic-Trennung ist aber nicht sauber: von den zwei domänennahen
Durchrutschern, die die bestehende Stufe-1-Schwelle passieren, bekommt einer
(eBay-Vergleich) korrekt niedrige Konfidenz, der andere (Omega-Bewertung)
dieselbe Konfidenz wie ein echter Treffer — **die im Design-Dokument
benannte Überlappungs-Gefahr tritt real auf**, wenn auch nur bei einem von
zwei geprüften Grenzfällen. Zu wenig Stichprobe (nur 4 Off-Topic-Fragen), um
daraus zu schließen, ob das Design insgesamt trägt — das ist genau die
Aufgabe von Stufe 2 (voller Eval-Lauf, Schwellen-Rekalibrierung), die als
nächstes ansteht.

---

## Nicht gepusht

Elf Commits liegen lokal auf `main`, keiner davon gepusht. Reihenfolge (neueste zuerst):

```
2a7907a feat: Reranker-Finetune auf Kaeufer/Verkaeufer-Hard-Negatives integriert
1ae583f refactor: Chroma + Hand-BM25 durch SQLite (FTS5 + sqlite-vec) ersetzt
dca39a2 feat: audience-Feld als harter Rollenfilter vor der RRF-Fusion
494c3fd docs: Rollen-Malus-Befund praezisiert, Verzahnung mit dem Rollenfilter-Design
3153489 docs: Rollenfilter-Design korrigiert - Heuristik statt Rewrite-Call-Erweiterung
dba4d89 docs: Handover-Kopf auf den Stand nach Schritt 4 gebracht
0cf2dda feat: Eval-Ausgabe zeigt Konfidenzintervalle und warnt bei zu kleiner Stichprobe
610a17f docs: Design fuer Rollenfilter (Schritt 1) und Engine-Konsolidierung (Schritt 2)
e62288a docs: Handover fuer die Eval-Set-Groesse
a930bd0 docs: Konfidenzintervalle ergaenzt, Kategorie im Embedding verworfen
db0f968 docs: Wurzel des Vektorpfad-Problems gefunden, zwei Gegenmittel verworfen
```

Vor dem Push: `HF_TOKEN` muss als Render-Env-Var gesetzt sein (siehe README,
Deployment-Abschnitt) — der Finetune-Commit macht das private HF-Hub-Modell zur
Voraussetzung fürs Booten. Ohne den Token startet der Service nicht.
