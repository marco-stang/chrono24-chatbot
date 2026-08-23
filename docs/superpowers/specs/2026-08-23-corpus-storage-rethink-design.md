# Corpus-Speicherung neu denken: Rollenfilter + Engine-Konsolidierung

## Kontext

Retrieval-Tuning ist an eine Wand gelaufen: zwoelf Hebel gemessen (README-Ablationstabelle),
alle verworfen oder neutral. Drei Misses bleiben reproduzierbar, zwei davon geteilt mit den
Faithfulness-Fehlern. Diagnose (README, Abschnitt "Warum der Vektorpfad hier versagt"):
das Embedding-Modell (`paraphrase-multilingual-MiniLM-L12-v2`) ist ein Paraphrase-Modell,
misst Satzaehnlichkeit statt Frage-beantwortet-Frage. Ein komplett anderer Embedder
(`multilingual-e5-base`) traf exakt dieselben drei Faelle — kein Modell-von-der-Stange-Fix mehr.

Konkretes Muster in zwei der drei Faelle: **Rollenverwechslung Kaeufer/Verkaeufer** bei
geteiltem Vokabular. Ziel-Dokument liegt im Vektor-Ranking auf Platz 76 bzw. 130, ein
Konkurrenzdokument gleicher Wortwahl aber falscher Rolle liegt vorn und gewinnt die
RRF-Fusion. Ein bereits getesteter "Rollen-Malus" (weicher Score-Abzug auf FAQ-*Kategorie*
nach dem Ranking) war bei Gewicht 0.7/0.5 neutral, bei 0.3 leicht negativ — kein klarer
Gewinn, aber auch kein harter Test der eigentlichen Idee.

Nutzer will das Projekt von Grund auf neu denken, speziell die Art, wie der Corpus
gespeichert wird. Ziel laut Brainstorming-Antworten: (a) die drei Misses strukturell loesen,
(b) das System vereinfachen (Chroma + Hand-BM25 + RRF-Fusion + Varianten-Mapping ist viel
bewegliche Teile), (c) eine Architektur, die auch mit echtem Chrono24-Umfang tragen wuerde —
ohne den Corpus selbst jetzt zu vergroessern. Portfolio-Wirkung war explizit nicht der
Treiber.

## Entscheidung

Zwei Schritte, in dieser Reihenfolge. Schritt 1 ist der Umfang **dieser** Implementierung.
Schritt 2 ist beschrieben, aber bewusst zurueckgestellt (eigene Session/PR), weil er nur
lohnt, wenn Schritt 1s Schema feststeht.

### Schritt 1 — Rollenfeld als harter Pre-Filter (dieser Umbau)

**Aenderung an `data/corpus.json`:** neues Pflichtfeld `audience` pro FAQ- und
Seiten-Chunk-Dokument, Werte `kaeufer` | `verkaeufer` | `neutral`. Population einmalig
per LLM-Klassifizierungslauf ueber alle ~150 Dokumente, danach Stichprobenkontrolle von
Hand (gleiches Muster wie die handkuratierte Synonymliste in `app/textproc.py`).

**Aenderung an `pipeline/index.py`:** `audience` wird wie andere Metadaten in Chroma
mitindexiert (analog zum bereits vorhandenen `category`-Feld, das aktuell *nicht* mehr
indexiert wird — `audience` hier bewusst schon, weil es fuer einen harten Filter gebraucht
wird, nicht nur als Text-Zusatz).

**Aenderung an `app/main.py`/Query-Rewrite:** der bestehende `rewrite_query`-Call (Haiku,
uebersetzt bereits nicht-deutsche Fragen) bekommt ein zusaetzliches Ausgabefeld
`query_audience: kaeufer|verkaeufer|neutral` — kein neuer LLM-Call, nur ein erweitertes
JSON-Schema am bestehenden Call.

**Aenderung an `app/retrieval.py`:** vor der RRF-Fusion, wenn `query_audience != neutral`:
Kandidatenmenge (sowohl BM25- als auch Vektor-Pfad) auf Dokumente mit
`audience in {query_audience, neutral}` einschraenken. Echter Ausschluss aus der
Kandidatenmenge, kein Score-Abzug — Unterschied zum bereits verworfenen Rollen-Malus, der
weich und auf der falschen Ebene (Kategorie statt Rolle, nach statt vor dem Ranking) ansetzte.

**Fehlerbehandlung:** Klassifiziert `rewrite_query` eine Anfrage falsch als `kaeufer` oder
`verkaeufer`, wenn sie eigentlich `neutral` waere, killt der harte Filter potenziell einen
echten Treffer statt Rauschen zu entfernen — anders als beim Malus gibt es hier keine sanfte
Abstufung. Deshalb: bei Unsicherheit soll `rewrite_query` `neutral` bevorzugen (im Prompt
explizit machen), und das bestehende Eval-Gate (`TUNING_MIN_HIT_RATE`,
`HOLDOUT_MIN_HIT_RATE`) bleibt das Sicherheitsnetz gegen eine Regression.

**Test:** `eval/run_eval.py` gegen den bestehenden Tuning-Satz (33 Fragen) und Held-out-Satz
(15 Fragen) — keine neue Infrastruktur, ein neuer Eintrag in der README-Ablationstabelle.
Zusaetzlich gezielt pruefen, ob die zwei als "Kandidaten-Problem" diagnostizierten Misses
(faq-0098-Fall, escrow-0007-Fall) jetzt im gefilterten Kandidatenpool auftauchen und korrekt
ranken. Ergebnis ehrlich dokumentieren, auch wenn neutral oder negativ — Projekt-Ethos.

**Rollback:** rein additive Aenderung auf dem bestehenden Chroma+BM25-Stack, per
`git revert` vollstaendig entfernbar ohne Seiteneffekt auf andere Komponenten.

### Schritt 2 — Engine-Konsolidierung: SQLite + FTS5 + sqlite-vec (zurueckgestellt)

Ersetzt Chroma + Hand-BM25 durch eine einzelne SQLite-Datei: FTS5-Virtualtabelle liefert
BM25 nativ (ersetzt `app/textproc.py`s Hand-BM25), `sqlite-vec`-Extension liefert
Vektor-Suche (ersetzt Chroma). Ein Schema (`documents`: id, type, question, text, category,
audience, canonical_id, source_url) statt zwei synchronisierter Doc-ID-Raeume. RRF-Fusion
bleibt Python-Code, aber ohne Dual-Store-Sync-Aufwand; der Rollenfilter aus Schritt 1 wird
zu einer SQL-`WHERE`-Klausel in beiden Teil-Queries.

Zusaetzliche strukturelle Aenderung: Index nicht mehr committen, sondern beim Docker-Build
aus `data/corpus.json` neu erzeugen (Embedding laeuft lokal, kostenlos, deterministisch
genug fuer einen Build-Schritt). Das entfernt den dokumentierten Schmerzpunkt "Chroma
mutiert committete Indexdateien beim bloßen Öffnen" (`git restore data/index/`-Tanz) durch
Wegfall der Praemisse statt durch Disziplin.

**Bricht eine dokumentierte Annahme:** README sagt aktuell, `eval-gate` "laedt den
committeten Index" ohne API-Kosten. Mit Schritt 2 braucht CI einen Build-Schritt davor
(Sekunden, weiterhin keine API-Kosten, da Embedding lokal) — muss im
`.github/workflows/ci.yml` ergaenzt werden.

**Offenes Risiko, vor Festlegung zu pruefen:** `sqlite-vec` ist juenger und kleiner als
Chroma. Vor Beginn von Schritt 2 pruefen: Windows-Wheel fuer lokale Entwicklung,
Linux-Wheel fuer Docker-Deploy, Kompatibilitaet der Embedding-Dimension.

**Warum zurueckgestellt:** lohnt sich unabhaengig vom Ausgang von Schritt 1 fuer die Ziele
(b) und (c), sollte aber erst starten, wenn das `audience`-Schema aus Schritt 1 feststeht,
um es nicht zweimal zu migrieren.

## Verworfen: GraphRAG / Wissensgraph ueber Kaeufer-/Verkaeufer-Flows

Ueberdimensioniert fuer die tatsaechliche Diagnose. Das Problem ist eine binaere
Unterscheidung (Kaeufer vs. Verkaeufer), kein komplexes Beziehungsgeflecht zwischen
Entitaeten. YAGNI — nicht weiterverfolgt, es sei denn neue Evidenz zeigt ein Problem, das
tatsaechlich Graph-Struktur braucht.

## Umfang dieser Implementierung

Nur Schritt 1. Schritt 2 ist dokumentiert fuer eine spaetere Session, kein Teil des
aktuellen Implementierungsauftrags.
