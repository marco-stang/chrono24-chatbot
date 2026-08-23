# Handover: Das Eval-Set ist zu klein

**Stand:** 2026-08-24 · `main` bei `0cf2dda`, **vier Commits lokal, nicht gepusht**
(zuletzt gepusht: `13c09d1`) · 162 Tests grün · ruff sauber · Working Tree clean

**Schritt 4 ist bereits umgesetzt** (`eval/stats.py`) — jeder Eval-Lauf druckt jetzt sein
Konfidenzintervall und warnt selbst, wenn die Stichprobe zu klein ist. Offen sind
Schritte 1–3.

Vorgänger-Handover: `HANDOVER-query-varianten-eval-gate.md` — dort stehen Architektur,
Schwellen und die Ablationshistorie. Dieses Dokument behandelt genau einen offenen Punkt.

---

## Das Problem in einem Satz

Die Zahlen, mit denen dieses Projekt wirbt, stammen aus Stichproben, die zu klein sind,
um sie zu tragen.

95-%-Wilson-Intervalle für den aktuellen Stand:

| Zahl | Wert | 95-%-Intervall | Breite |
|---|---|---|---|
| Tuning-Hit-Rate@5 | 91 % (30/33) | [76 %, 97 %] | 20 pp |
| Held-out-Hit-Rate@5 | 100 % (15/15) | [80 %, 100 %] | 20 pp |
| **Abstention-Rate** | **50 % (7/14)** | **[27 %, 73 %]** | **46 pp** |
| Faithful-Rate | 92,1 % (Mittel aus 5 Läufen, Spanne 85–97 %) | — | — |

„100 % auf dem Held-out-Set" belegt statistisch nur **„mindestens 80 %"**. Die
Abstention-Rate ist mit 46 Punkten Breite kaum mehr als eine Richtungsangabe.

**Die wichtigste Folge:** Bei 33 Fragen ist ein Unterschied erst ab rund **vier Fragen**
überhaupt vom Zufall unterscheidbar. Das erklärt rückwirkend, warum die
README-Ablationstabelle inzwischen zwölf Einträge mit „neutral" trägt — ein Teil davon
sind womöglich echte kleine Effekte, die diese Stichprobe nicht auflösen kann. Man weiß
es nicht, und das ist der Punkt.

---

## Was schon gemessen ist (bitte nicht wiederholen)

Zwölf Ansätze, die drei Tuning-Misses zu beheben, sind gemessen und verworfen — alle in
der README-Ablationstabelle mit Zahlen. Die wichtigsten für die Einordnung:

| Ansatz | Ergebnis |
|---|---|
| Stärkerer Reranker `bge-reranker-v2-m3` (568M) | 0 von 3 Misses, Latenz 0,7 s → 6,7–10,9 s |
| Embedder `multilingual-e5-base`, Merge fair kalibriert | **exakt dieselben drei Misses** |
| Doc2Query für alle 132 page_chunks (568 Fragen) | Hit-Rate unverändert |
| Kategorie im Embedding-Text (3 Varianten) | dieselben drei Misses |
| `TOP_K_CANDIDATES`, `RRF_K`, Pfad-Gewichte | neutral oder schlechter |
| Rollen-Malus über die FAQ-**Kategorie**, weich, nach dem Ranking | neutral — **aber kein harter Test**, siehe Kasten unten |

**Die Diagnose ist abgeschlossen:** `paraphrase-multilingual-MiniLM-L12-v2` ist ein
Paraphrase-Modell und belohnt Nomen-Überlappung statt Frage-Antwort-Passung (faq-0121
bekommt 0.870, weil es „Privatverkäufer" und „meine Uhr" teilt, ohne die Frage zu
beantworten; faq-0098 bekommt 0.495, obwohl seine Variante fast die Frage selbst ist).
Dass ein asymmetrisch trainiertes Modell an denselben drei Fällen scheitert, schließt
Modelle von der Stange als Lösung aus.

**Dieses Handover ist deshalb bewusst kein weiterer Optimierungsversuch.** Es geht darum,
die Messung tragfähig zu machen — nicht die Zahl zu heben.

> **Achtung, eine Idee ist nicht widerlegt.** Parallel zu dieser Messrunde ist ein Design
> entstanden: `docs/superpowers/specs/2026-08-23-corpus-storage-rethink-design.md` —
> ein `audience`-Feld (kaeufer/verkaeufer/neutral) auf allen Dokumenten, als **harter
> Filter vor der Fusion**. Der oben verworfene Rollen-Malus ist *nicht* dasselbe: er
> benutzte die FAQ-Kategorie als Rollen-Ersatz und zog Punkte nach dem Ranking ab. 132 der
> 318 Dokumente tragen gar keine Kategorie — darunter info-escrow-0007, einer der beiden
> Rollen-Fälle. Der Malus konnte dieses Dokument nie erreichen. Die Zeile in der
> Ablationstabelle darf also nicht als „Rollen-Idee tot" gelesen werden.

---

## Ziel

| Set | heute | Ziel | Priorität |
|---|---|---|---|
| `eval/questions_offtopic.json` | 14 | 60 | **hoch** — mit Abstand das schwächste |
| `eval/questions_holdout.json` | 15 | 60 | hoch |
| `eval/questions.json` (Tuning) | 33 | 100 | mittel |

Bei 60 Off-Topic-Fragen schrumpft das Intervall der Abstention-Rate von 46 auf ~25 pp,
bei 100 Tuning-Fragen das der Hit-Rate von 20 auf 11 pp.

---

## Der methodische Fallstrick (der eigentliche Inhalt dieser Aufgabe)

**LLM-generierte Eval-Fragen aus den eigenen Dokumenten sind systematisch zu leicht.**
Sie übernehmen den Wortlaut des Zieldokuments, und genau auf Wortlaut-Überlappung
reagiert dieses Embedding-Modell (siehe Diagnose oben). Ein so erzeugtes Set würde die
Hit-Rate hochtreiben, ohne dass sich am System etwas ändert — die Zahl stiege, die
Wahrheit nicht.

Es gibt dafür schon einen Beleg im Repo: `data/variants.json` enthält 885
LLM-Umformulierungen der FAQ-Fragen. Die sind **im Index** und dürfen deshalb nie als
Eval-Fragen dienen — das wäre ein direkter Zirkelschluss.

**Regeln, die deshalb einzuhalten sind:**

1. **Generierte Fragen niemals mit den handgeschriebenen mischen.** Eigene Dateien,
   eigene Zahl im Report. Der Vergleich der beiden Zahlen *ist* die Bias-Messung: liegt
   das generierte Set bei 98 % und das handgeschriebene bei 91 %, ist bewiesen, dass es
   zu leicht ist — und das gehört so ins README.
2. **Generierungs-Prompt gegen Wortlaut-Übernahme.** Nutzersprache statt Dokumentsprache,
   Umgangston, andere Wortwahl als der Quelltext. Das mildert den Bias, beseitigt ihn
   nicht — deshalb Regel 1.
3. **Off-Topic-Fragen brauchen keine Quelldokumente** und sind darum am unbedenklichsten
   generierbar. Wichtig ist dort die Mischung: eindeutig fachfremd *und* absichtlich nah
   am Domänenvokabular (Uhrenbewertung, Konkurrenzvergleich, Versicherung). Die aktuellen
   14 Fragen sind ein gutes Muster — die 7 Durchrutscher sind genau die domänennahen.
4. **Gold-Labels prüfen, nicht glauben.** Bei den drei bestehenden Misses wurden die
   Labels von Hand gegen den Korpus geprüft und waren korrekt (faq-0098 nennt wörtlich
   die 6,5 % Provision). Generierte Labels müssen dieselbe Prüfung durchlaufen — sonst
   misst man Label-Fehler statt Retrieval-Fehler. Es gibt dafür ein Muster im
   Schwesterprojekt (Handover Brief Generator, `scripts/data_gen_run.py`): jeder
   generierte Fakt wird vor dem Speichern gegen die Quelle validiert, Ungedecktes
   verworfen.
5. **Nicht-deutsche Fragen brauchen ein `rewritten`-Feld** (offline erzeugt, siehe
   `eval/run_eval.py::eval_query`) — sonst misst die Eval einen Pfad, den es in
   Produktion nicht gibt.

---

## Vorgeschlagene Schritte

1. **Off-Topic-Set auf 60** (`eval/questions_offtopic.json`). Keine Gold-Labels nötig,
   kein Zirkelschluss-Risiko. Format ist heute `[{"question": "..."}]`. Danach
   `MIN_ABSTENTION_RATE` neu setzen — Regressionsboden mit Puffer, nicht Bestmarke.
2. **Held-out auf 60** als getrenntes generiertes Set, z. B.
   `eval/questions_generated.json`, mit eigener Zahl im Report. Das bestehende
   handgeschriebene Held-out-Set bleibt unverändert bestehen und behält seine eigene Zahl.
3. **Bias-Differenz messen und ins README schreiben.** Hit-Rate auf handgeschrieben vs.
   generiert. Wenn die Differenz groß ist, ist das ein Befund, kein Makel.
4. ~~**Konfidenzintervalle mitdrucken**~~ — **erledigt.** `eval/stats.py` mit
   `wilson_interval()` und `format_rate()`, eingebunden in `eval/run_eval.py` (alle drei
   Raten) und `eval/judge.py` (Faithful-Rate). Ab 20 Punkten Intervallbreite hängt die
   Ausgabe selbst einen Warnhinweis an; aktuell trifft das **alle drei** Zahlen. 8 Tests.
5. Erst danach die CI-Schwellen nachziehen.

Nach Schritt 1–3 sollte die Warnung bei Off-Topic und Held-out verschwinden — das ist
das messbare Abnahmekriterium für diese Aufgabe.

---

## Warum diese Aufgabe vor dem Rollenfilter kommt

Das Rollenfilter-Design zielt auf die zwei Kandidaten-Misses — also **2 von 33 Fragen**
oder 6 Prozentpunkte. Das aktuelle Tuning-Set hat ein Konfidenzintervall von **20 Punkten
Breite**. Selbst wenn der Filter beide Fälle löst, ist der Effekt **statistisch nicht von
Null zu unterscheiden**: 30/33 und 32/33 haben überlappende Intervalle.

Daraus folgt nicht, dass der Filter nicht gebaut werden soll — sondern dass sein Erfolg
anders belegt werden muss als über die Gesamt-Hit-Rate:

- **Gezielt prüfen**, ob faq-0098 und info-escrow-0007 im gefilterten Kandidatenpool
  auftauchen und korrekt ranken (das steht so schon im Design unter „Test").
- **Keine Regression** auf allen anderen Fragen — das ist das, was die Gesamtzahl
  tatsächlich beantworten kann.
- Für eine Aussage wie „der Filter hebt die Hit-Rate" braucht es **erst das größere
  Eval-Set**.

Wer beide Aufgaben in einer Sitzung macht: **Eval-Set zuerst**, sonst ist das Ergebnis
des Filters nicht interpretierbar.

---

## Kosten

- Off-Topic-Generierung: ~1 Haiku-Call pro Charge, vernachlässigbar.
- Generiertes Held-out-Set mit Label-Validierung: ~100–200 Haiku-Calls, wenige Cent.
- Judge-Läufe zum Gegenprüfen: **~66 Haiku-Calls pro Lauf**, und wegen der Streuung
  (85–97 % über sieben Läufe) braucht eine belastbare Aussage mehrere Läufe.
- Reindex kostet kein API-Geld (Embedding-Modell ist lokal).

---

## Fallstricke aus der letzten Sitzung

- **Eval-Läufe sind langsam.** Der Cross-Encoder läuft auf CPU; ein `TOP_K`-Sweep oder
  fünf Judge-Läufe brauchen deutlich mehr als 10 Minuten. **Immer im Hintergrund starten**,
  sonst läuft der Vordergrund-Timeout ab.
- **Chroma mutiert committete Binärdateien beim bloßen Öffnen.** Nach jedem Lauf
  `git restore data/index/` — **außer nach einem bewussten Reindex**, dort setzt das
  restore die SQLite auf einen Stand zurück, der auf ein gelöschtes Segment zeigt.
- **Jeder Reindex legt ein neues HNSW-Segment-Verzeichnis an** und lässt das alte als
  Waise liegen. Prüfen mit `select id from segments` gegen `ls data/index/chroma/`.
- **Ein laufender `uvicorn` sperrt `chroma.sqlite3`.**
- **Experimente gegen einen Wegwerf-Index außerhalb des Repos** laufen lassen
  (`pipeline.index.doc_embed_text` monkeypatchen, `build_index` in ein
  Scratch-Verzeichnis). So bleibt `data/index/` unangetastet, bis eine Variante wirklich
  gewinnt.
- **Merge-Schwelle beim Embedder-Vergleich mitkalibrieren.** `DEDUPE_THRESHOLD = 0.95` ist
  auf das aktuelle Modell getunt; e5 entfernte damit 18 statt 5 Dokumente und riss
  Zieldokumente mit. Der erste e5-Vergleich war deshalb ungültig — gleiche Duplikat-Anzahl
  herstellen, sonst vergleicht man zwei verschiedene Korpora.
- **`temperature=0` ist keine Determinismus-Garantie.** Alle drei Claude-Calls stehen auf
  0; zwei Läufe bei identischem Code ergeben trotzdem 91 % und 94 %.

---

## Offener Nebenpunkt

`git stash@{0}` enthält den fertigen, getesteten Doc2Query-Code (`generate_variants` nimmt
ein doc statt einer Frage, `_variant_entries` öffnet sich für alle Doktypen). Gemessen
neutral und deshalb nicht gemerged. Wer mit besserem Generierungs-Prompt nachlegen will,
findet dort den Ausgangspunkt; die erzeugten Fragen lagen unter
`scratchpad/variants_d2q.json` (Scratch ist flüchtig, ggf. neu erzeugen).

---

## Nicht gepusht

- **chrono24-chatbot:** `db0f968`, `a930bd0`, `e62288a` und `0cf2dda` liegen lokal auf `main`.
- **marco-os:** drei Commits lokal auf `master` (`e774d16` Projektknoten, `0451dd2`
  Portrait-Layout, `15cf2f8` Zahlen-Korrektur). Ein Push dort geht direkt live auf
  GitHub Pages. Offener Befund: mobil überlappt unten noch ein Label-Paar
  („Interview Cockpit" / „Document Auto-Classifier"), Desktop ist sauber. Der Graph
  rendert in ein `<canvas>`, Labels sind keine DOM-Knoten — Kollisionen lassen sich nur
  per Screenshot prüfen, nicht per `getBoundingClientRect`.

Wenn das Eval-Set wächst, tragen beide marco-os-Knoten veraltete Zahlen
(`stats`-Feld und Beschreibungstext) — vor dem Push nachziehen.
