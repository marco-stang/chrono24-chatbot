# Chrono24-FAQ-Chatbot

Ein RAG-Chatbot, der Fragen zu Kauf, Verkauf, Käuferschutz und Versand auf
Basis der öffentlichen Chrono24-Hilfeseiten beantwortet — mit Quellenangaben
und sichtbaren Retrieval-Treffern statt einer Blackbox-Antwort. Was er nicht
belegen kann, übergibt er an Menschen: als automatisch erzeugtes Briefing,
dessen Aussagen ein deterministischer Validator gegen den Chatverlauf prüft.

**Demo-Link:** folgt (Deploy steht noch aus).

![Übergabe-Demo: Szenario läuft in die Sackgasse, der Bot übergibt mit live geprüftem Briefing, der Support löst den Fall](docs/img/demo-briefing.png)

**Auf einen Blick:**

- **Hybrid-Retrieval:** BM25 + Vektorsuche (Chroma), RRF-Fusion,
  Cross-Encoder-Reranker, Synonym-Expansion, LLM-generierte Query-Varianten
  je FAQ — 91 % Hit-Rate@5,
  held-out validiert (93 %)
- **Antworten nur aus Kontext** (Claude Haiku) mit `[n]`-Quellenangaben;
  ohne Beleg sagt der Bot „weiß ich nicht" — abgesichert in drei Schichten
  (Retrieval-Gate, Prompt, Faithcheck), jede einzeln gemessen, siehe
  [Konfidenz-Gate](#konfidenz-gate-für-themenfremde-fragen)
- **Laufzeit-Faithfulness-Check:** jeder Antwortsatz wird deterministisch
  gegen die zitierten Quellen geprüft — Ampel im UI, kein zweiter LLM-Call
- **Handover-Briefing:** Übergabe an Menschen mit validierten Aussagen;
  Unbelegbares wird abgelehnt statt still ausgeliefert
- **Geführte Übergabe-Demo:** drei Szenarien an einer Betreuer-Zeitachse
  (Bot → Tier-1 → Tier-2), Briefing-Prüfung als sichtbare Animation,
  Kino-Modus zum automatischen Abspielen
- **Guards:** IP-Rate-Limits und globales Tages-Token-Budget

**Inhalt:**
[Architektur](#architektur) ·
[Warum Hybrid-RAG](#warum-hybrid-rag) ·
[Faithfulness-Check](#laufzeit-faithfulness-check-deterministisch) ·
[Handover](#übergabe-an-menschen-handover-briefing) ·
[Übergabe-Demo](#die-übergabe-demo) ·
[Scraping-Ethik](#scraping-ethik) ·
[Lokal starten](#lokal-starten) ·
[Tests](#tests) ·
[CI Eval Gate](#ci-eval-gate) ·
[Guards](#guards) ·
[Deployment](#deployment-render)

## Disclaimer

Inoffizielles Portfolio-Projekt. Nicht mit Chrono24 verbunden, nicht von
Chrono24 autorisiert oder unterstützt. Antworten ohne Gewähr; Quelle ist
ausschließlich der öffentliche Hilfebereich von chrono24.de.

## Architektur

Zwei getrennte Läufe: eine lokale Offline-Pipeline baut den Index, der
Online-Service liest ihn nur noch — er scrapt zur Laufzeit nie.

```
Offline-Pipeline (lokal, einmalig bzw. bei Bedarf)
───────────────────────────────────────────────────
scraper (Playwright) → data/raw/*.html
parser               → data/corpus.json   (versioniert im Repo)
indexer              → data/index/        (Chroma + BM25, versioniert)

Online-Service (Docker)
───────────────────────────────────────────────────
Browser-Chat-UI (statisches HTML/JS, von FastAPI ausgeliefert)
   ↓ POST /api/chat (SSE-Stream zurück)
FastAPI
   ├─ Retrieval: BM25 + Vektor → RRF-Fusion → Top 5
   ├─ Claude Haiku: Antwort nur aus Kontext, mit Quellen
   └─ Guards: IP-Rate-Limit, Tages-Token-Budget
```

Der Server braucht zur Laufzeit kein Scraping und keinen Zugriff auf
Chrono24 — nur den fertigen Index aus dem Repo und den Anthropic-API-Key.
`data/corpus.json` ist ein zwischengespeicherter Snapshot öffentlicher
Chrono24-Hilfeseiten zu Demo-Zwecken, kein Live-Datenzugriff.

## Warum Hybrid-RAG

Jedes FAQ-Paar bleibt als eigenes Dokument erhalten statt in generische
Chunks zerschnitten zu werden — das erhält die Q&A-Struktur, und die
Vektorsuche embedded gezielt die Frage, nicht die Antwort, weil Nutzer:innen
selbst Fragen stellen (Frage-auf-Frage-Matching). BM25 fängt exakte
Begriffe (Produktnamen, Fachbegriffe) ab, die Embeddings manchmal
verwässern; ein RRF-Fusion aus beiden Rankings kombiniert die Stärken.

Gemessen mit einem handgeschriebenen 33-Fragen-Set (26 FAQ- und 7
Seiten-Chunk-Ziele, 3 davon englisch), iterativ verbessert — jede Stufe
wurde einzeln gemessen, auch die, die erstmal nichts bringt:

| Retrieval-Stufe | Hit-Rate@5 |
|---|---|
| Baseline: BM25 + Vektor + RRF | 76 % (25/33) |
| + Near-Duplicate-Merge der Seiten-Chunks (−5 Docs) | 73 % (24/33) |
| + Cross-Encoder-Reranker (mmarco-mMiniLMv2) über die RRF-Top-10 | **88 % (29/33)** |
| + Query-Übersetzung nicht-deutscher Fragen vor BM25 | 88 % (29/33) |
| A: `TOP_K_CANDIDATES` 10 → 25 (mehr Kandidaten fürs Reranking) | 88 % (29/33) |
| B: FAQ-Embedding auf Frage+Antwort statt nur Frage umgestellt | 82 % (27/33) |
| A+B kombiniert | 88 % (29/33) |
| **Gewinner: Status quo** (`TOP_K_CANDIDATES=10`, FAQ-Embedding = Frage) | **88 % (29/33)** |
| + handkuratierte Synonym-Expansion der Query (nur BM25-Pfad) | **91 % (30/33)** |
| + Query-Varianten (LLM-Umformulierungen, nur Embedding-Pfad) | 91 % (30/33) |
| verworfen: Titel-Exaktheits-Bonus auf den Rerank-Score (α = 0.5–4) | 88 % → 85 % |
| verworfen: Seitentitel zusätzlich ins `page_chunk`-Embedding | 91 % → 88 % |
| verworfen: FAQ-Kategorie in den Rerank-Text | 91 % (30/33), exakt neutral |

A und A+B erreichen exakt dieselbe Trefferquote wie der Status quo, nur mit
anderer Miss-Verteilung — kein echter Gewinn, nur verschobene Fehler bei
mehr Rechenaufwand (2,5× mehr Kandidaten fürs Reranking). B verschlechtert
sich sogar: die Antwort im Embedding verwässert das gezielte
Frage-auf-Frage-Matching, das oben als Designentscheidung begründet ist.
Bei Gleichstand gewinnt laut Entscheidungsregel die einfachste Option — das
ist hier der unveränderte Status quo, also blieb Code und Index exakt wie
vor dem Experiment.

Der Dedupe-Schritt kostet solo einen BM25-Randfall, verhindert aber, dass
fast-identische Chunks die Top-5 verstopfen — zusammen mit dem Reranker
ist er neutral bis positiv. Die Query-Übersetzung ändert auf diesem Set
keinen Zähler (zwei der drei englischen Fragen trafen schon über die
multilingualen Embeddings), macht den Live-Pfad für englische Fragen aber
robust, weil BM25 sonst am deutschen Korpus vorbeiläuft.

Die Synonym-Expansion (`QUERY_SYNONYMS` in `app/textproc.py`) kam nach
einer Diagnose der damals 4 Misses dazu: Alltagswörter der Nutzerfragen
(„bezahlen", „zurückschicken") ergänzen im BM25-Pfad ihre FAQ-Pendants
(„kostet", „Rückgabe") — dasselbe Muster wie ein Elasticsearch-Synonym-
Filter, bewusst klein gehalten, Embeddings bleiben unangetastet. Das holt
den „zurückschicken"-Miss ohne einen einzigen Fall zu kippen (Held-out
unverändert 13/15). Ehrlicher Vorbehalt: die Liste entstand beim Anschauen
der Misses — die 91 % bleiben eine Tuning-Set-Zahl. Ein im selben Zug
gemessener Titel-Exaktheits-Bonus (generische Titel wie „Was kostet …"
belohnen) wurde verworfen: wirkungslos bei kleinen Gewichten, ab α = 4
kippt er einen bisher richtigen Fall.

Query-Varianten (`pipeline/variants.py`) generieren pro FAQ per Haiku 3–5
Umformulierungen und embedden sie zusätzlich zur Originalfrage, verknüpft
über ein `canonical_id`-Metadatum in Chroma. Ziel: Nutzerformulierungen, die
weiter von der FAQ-Frage abweichen, als es die multilingualen Embeddings
allein auffangen.

Die erste Messung (Tuning 91 %, Holdout 87 % — beide exakt auf dem Stand vor
dem Experiment) hatte einen Bug als Ursache, keinen echten Nulleffekt:
`Retriever.retrieve` begrenzte die Chroma-Anfrage auf `TOP_K_CANDIDATES = 10`
Treffer — aus der jetzt viel größeren Collection (Original- plus bis zu 5
Varianten-Einträge pro FAQ) —, und zwar *bevor* die Varianten per
canonical_id auf ihr Original zurückgemappt wurden. Mehrere Varianten
derselben Frage konnten sich so mehrere der 10 Rohplätze teilen und
kollabierten danach auf denselben Kandidaten: der Vektorpfad lieferte damit
oft nur noch 2–4 statt 10 unterschiedliche Dokumente an die RRF-Fusion — ein
stiller Rückschritt gegenüber dem Stand vor den Varianten, der den
vermeintlichen Nulleffekt der Varianten erst erzeugt hat (Gewinn durch mehr
Kandidaten und Verlust durch die Verdrängung hoben sich gegenseitig auf).

Der Fix überfetcht jetzt (`n * (1 + MAX_VARIANTS_PER_DOC)`, gedeckelt auf
die Collection-Größe) und dedupliziert *vor* dem Kappen auf `n` — der
Vektorpfad liefert damit wieder n verschiedene Dokumente wie vor den
Varianten. `TOP_K_CANDIDATES` selbst blieb unverändert, um die Messung
nicht nachträglich zu tunen. Nachgemessen: Tuning unverändert **91 %
(30/33)**, Holdout jetzt **93 % (14/15)** — ein Fall mehr als vorher
(87 %, 13/15) und mehr als die Tuning-Zahl. Ehrlich: ein einzelner
zusätzlicher Treffer auf 15 Fragen ist ein kleines Sample und kein Beweis
für einen großen Effekt, aber es ist der erste tatsächlich positive Befund
der Varianten — vorher hat der Bug jeden echten Gewinn verdeckt.

Zwei weitere Kandidaten wurden im selben Zug gemessen und beide verworfen.
Der **Seitentitel zusätzlich im `page_chunk`-Embedding** (statt nur der
Überschrift) sollte generischen Überschriften wie „Ablauf" den fehlenden
Kontext geben. Gemessen: Tuning 91 % → 88 %, Held-out 93 % → 87 %. Die
Ursache steht im Indexer-Log: der Titel ist bei allen Chunks derselben
Seite identisch, macht sie einander also ähnlicher statt unterscheidbarer.
Der Near-Duplicate-Merge (Schwelle 0.95) entfernte daraufhin **10 Chunks
zusätzlich** — darunter `info-mychrono24-0003` und `info-conditions-0004`,
die prompt als neue Misses auftauchten. Ein Feld, das keine Information
zwischen Geschwister-Chunks trägt, verdrängt die, die es tut.

Die **FAQ-Kategorie im Rerank-Text** (der Cross-Encoder sieht
`"Kategorie — Frage"` statt nur `"Frage"`) war exakt neutral: identische
Trefferquote auf beiden Sets, identische Miss-Listen, kein einziger Fall
kippt in irgendeine Richtung. Das Themenlabel trägt für den Reranker
nichts bei, was Frage und Antwort nicht schon hergeben. Die Kategorie
steht weiter im Corpus, wird aber nicht mehr als Chroma-Metadatum
indexiert — Daten ohne Abnehmer wären nur eine Einladung, sie für mehr zu
halten, als sie gemessen sind.

Die 3 verbleibenden Misses sind diagnostizierte harte Fälle — das
Retrieval findet jeweils die richtige Themenfamilie, aber das falsche
Familienmitglied: generische Zieldokumente („Was ist Certified?", „Was
kostet der Kommissionsverkauf?") verlieren gegen spezifischere
Geschwister, die mehr Wortlaut der Frage tragen. Keine geschönten
Fragen, keine kaputte Konfidenzschwelle.

### Held-out-Validierung

Alle Zahlen oben stammen vom selben 33-Fragen-Set, das auch für jede
Tuning-Entscheidung genutzt wurde — das Risiko, unbewusst auf dieses Set hin
zu optimieren, ist real. Als Gegenprobe gibt es `eval/questions_holdout.json`:
15 neue, nie fürs Tuning verwendete Fragen zu Dokumenten, die im
Tuning-Set nicht als Ziel vorkommen (11 FAQ-, 4 Seiten-Chunk-Ziele, 2
englisch), einmalig gegen die finale Konfiguration gemessen:
**93 % (14/15)**. Das liegt sogar leicht über der Tuning-Zahl (91 %) —
kein Anzeichen für schweres Eval-Set-Overfitting, weil ein System, das nur
auf die 33 Tuning-Fragen zugeschnitten wäre, auf neuen Fragen deutlich
stärker einbrechen würde. Die später ergänzte Synonym-Expansion wurde
damals auf dem Held-out gegengeprüft: exakt gleiche Treffer (13/15), kein
Fall kippt — die aktuelle Zahl (14/15) stammt aus dem Überfetch-Fix des
Vektorpfads für Query-Varianten, siehe oben.

### Konfidenz-Gate für themenfremde Fragen

Hit-Rate misst Recall, nicht Abstention — ob der Bot bei Fragen ohne
Chrono24-Bezug ehrlich leer zurückgibt statt zu halluzinieren, ist oben
eine eigene Behauptung. Sie stützt sich auf **drei unabhängige Schichten**,
und die Zahlen in diesem Abschnitt gehören nur zur ersten:

1. **Retrieval-Konfidenz-Gate** (`app/retrieval.py`): kein Treffer, kein
   LLM-Call. Billig, aber grob — gemessen: **50 % der Off-Topic-Fragen**.
2. **System-Prompt**: „Steht die Antwort nicht im Kontext, sage ehrlich…".
   Der [LLM-as-Judge-Lauf](#antwortqualität-llm-as-judge) weist je nach Lauf
   1–2 korrekte Verweigerungen aus.
3. **Laufzeit-Faithcheck**: jeder Antwortsatz wird gegen die zitierten
   Quellen geprüft, siehe
   [Laufzeit-Faithfulness-Check](#laufzeit-faithfulness-check-deterministisch).

Für Schicht 1 gibt es `eval/questions_offtopic.json` (14 Fragen, gemischt
eindeutig themenfremd und absichtlich nah am Domänenvokabular —
Wertschätzung einer konkreten Uhr, Konkurrenzvergleich, Kapitalanlage) und
`abstention_rate()` in `eval/run_eval.py`, das den Anteil korrekt leerer
Antworten misst.

**Erste Messung: 0 % (0/14).** Auch „Wie backe ich einen Hefezopf?" bekam
einen Treffer. Das alte Gate verlangte, dass *beide* Signale unter ihrer
Schwelle liegen (`sim < 0.35 und bm25 < 4.0`) — und die Cosine-Similarity
des multilingualen MiniLM sinkt für kurze deutsche Fragesätze praktisch
nie so tief: schon reine Fragesatz-Struktur („Wie … ich …?") erzeugt hohe
Ähnlichkeit, unabhängig vom Thema. Der Hefezopf erreichte 0.742 — mehr als
acht echte Fachfragen.

Deshalb die Signale einzeln vermessen, on-topic (48 Tuning- und
Holdout-Fragen) gegen off-topic (14):

| Signal | on-topic min | off-topic max | trennt allein? |
|---|---|---|---|
| Cosine-Sim | 0.607 | 0.773 | nein |
| BM25 (bester Treffer) | 6.36 | 20.96 | nein |
| Reranker (bester Score) | -5.6 | -0.19 (Median -5.2) | nein |

Kein Signal trennt allein, aber jedes hat einen Bereich, in dem nur
Off-Topic-Fragen liegen. Das Gate ist jetzt **ODER-verknüpft**: reicht
*ein* Signal eindeutig nicht (`sim < 0.40` oder `bm25 < 5.5` oder
`rerank < -6.0`), antwortet der Bot leer. Die Schwellen liegen jeweils unter
dem on-topic-Minimum. Gegen dieselben Daten gerechnet:

| Regel | Abstention | on-topic verloren |
|---|---|---|
| alt: `sim < 0.35` **und** `bm25 < 4.0` | 0/14 | 0 |
| `sim < 0.40` **oder** `bm25 < 5.5` | 5/14 | 0 |
| zusätzlich **oder** `rerank < -6.0` (aktiv) | **7/14 (50 %)** | **0** |
| … mit `rerank < -5.0` stattdessen | 9/14 | 1 (Versand-Frage) |

Hit-Rate bleibt unverändert (91 % / 93 %). Die 7 Durchrutscher sind genau
die gewollt domänennahen Fragen („Wie viel ist meine Omega wert?",
„eBay-Rückgabe im Vergleich zu Chrono24?", Versicherungsvergleich) — die
landen auf thematisch benachbarten FAQs, kein Retrieval-Signal kann sie vom
Korpus trennen. Dort müssen Schicht 2 und 3 greifen.

Ehrlich dazu: 62 Fragen sind ein kleines Sample, und die Margen sind dünn
(BM25-Schwelle 5.5 gegen on-topic-Minimum 6.36). `MIN_ABSTENTION_RATE`
steht deshalb bei 35 % statt 50 %, mit Puffer für Einzelfrage-Rauschen
(eine Frage = 7 Prozentpunkte). Eine Frage, die im Schwellenbereich kippt,
macht CI nicht rot, eine Rückkehr zum alten Verhalten schon.

### Antwortqualität (LLM-as-Judge)

Hit-Rate misst nur, ob der richtige Kontext gefunden wird — nicht, ob die
Antwort ihn auch korrekt nutzt. Dafür läuft `eval/judge.py` die komplette
Live-Pipeline (Query-Rewrite → Retrieval → Antwort-Streaming) für alle 33
Testfragen durch und lässt einen zweiten Haiku-Call die Antwort anhand des
tatsächlich gesehenen Kontexts bewerten: `faithful` (sind alle
Tatsachenaussagen belegt?) und `answered` (voll/teilweise/nein/verweigert).

Der erste Lauf ergab 100 % (33/33) und stand so eine Weile im README. Diese
Zahl ist nicht reproduzierbar, und das ist die eigentliche Lehre dieses
Abschnitts. Sechs Läufe über dieselben 33 Fragen:

| # | Judge | Bot | Faithful-Rate |
|---|---|---|---|
| 1 (21.08., lokal) | temp 1.0 | temp 1.0 | 100 % (33/33) |
| 2 (23.08., CI) | temp 1.0 | temp 1.0 | 94 % (31/33) |
| 3 (23.08., CI) | temp 1.0 | temp 1.0 | **88 % (29/33)** |
| 4 (23.08., lokal) | **temp 0** | temp 1.0 | 94 % |
| 5 (23.08., lokal) | temp 0 | **temp 0** | 91 % |
| 6 (23.08., lokal) | temp 0 | temp 0 | 94 % |
| 7 (23.08., CI) | temp 0 | temp 0 | 97 % (32/33) |

**Zwischen Lauf 2 und 3 lag kein einziger Commit an der Pipeline** — nur
Markdown und eine CI-Datei. Die 6 Punkte Unterschied sind reines
Sampling-Rauschen. Ursache: keiner der drei Claude-Calls (Query-Rewrite,
Antwort, Judge) setzte `temperature`, liefen also auf dem API-Default 1.0
— ein Messinstrument, das selbst würfelt.

Alle drei stehen jetzt auf `temperature=0`. Das hilft, löst es aber nicht:
Läufe 5 bis 7 sind bei identischem Code und identischer Temperatur 91 %,
94 % und 97 %. Temperatur 0 ist gierige Dekodierung, keine
Determinismus-Garantie der API. Bei 33 Fragen entspricht eine einzige Frage
3 Prozentpunkten — das Sample ist für eine Zahl mit zwei Stellen zu klein.
Wer hier eine Bestmarke herausgreift, sucht sich seinen Lauf aus.

Konsequenz: `MIN_FAITHFUL_RATE` steht bei **82 %** (27/33), zwei Fragen
unter dem gemessenen Boden von 29/33. Eine Schwelle bei 90 % war auf diesem
Sample nachweislich flaky — ein Lauf von sechs fiel darunter, ohne dass
sich am Code etwas geändert hatte, und rote CI ohne Regression ist
schlimmer als gar kein Gate, weil man aufhört hinzusehen. Das Gate fängt
damit einen echten Einbruch, nicht das Rauschen.

Zur Einordnung der Verteilung (Lauf 6): voll 25, teilweise 5, verweigert 3,
nein 0. Rohdaten des ersten Laufs in `eval/judge_results.json`, die
späteren stehen in den CI-Logs des `quality-gate`-Jobs.

Nicht jeder Treffer ist Rauschen. Aus Lauf 2 stammt ein echter inhaltlicher
Fehler: Auf die Frage nach der französischen Meldepflicht dreht die
Antwort eine Schwelle um — sie schreibt, Verkäufe würden „nur gemeldet, wenn
mindestens 20 Verkäufe oder mindestens 3.000 EUR" erreicht sind, während der
Kontext das Gegenteil sagt (unterhalb dieser Schwelle wird *nicht* gemeldet).
Der Retrieval-Treffer war korrekt, die Formulierung kippt die Logik. Genau
dagegen hilft weder eine bessere Hit-Rate noch das Konfidenz-Gate — es ist
der Fall, für den der Faithcheck und der Judge da sind. Ein milderer Treffer
taucht über mehrere Läufe hinweg auf: eine Versandkosten-Antwort, die den
Auslandsfall aus einer zitierten Quelle unterschlägt.

Ehrlicher Hinweis zur Methodik: Der Judge ist derselbe Modelltyp
(`claude-haiku-4-5`) wie der Chatbot selbst — gleiche Modellfamilie, damit
besteht eine milde Bias-Gefahr (der Judge könnte Fehler des Bots
systematisch übersehen, die ein anderes Modell auffangen würde). Für ein
belastbareres Signal wäre ein stärkeres oder anderes Modell als Judge
vorzuziehen; hier ist es eine bewusste Kostenentscheidung fürs Demo-Projekt.

### Gegenprobe mit Ragas

Gleiche Messung, zweites Werkzeug: [Ragas](https://github.com/explodinggradients/ragas)
ist das verbreitetste Open-Source-Framework für RAG-Evaluation. Seine
Faithfulness-Metrik misst feiner als der eigene Judge — sie zerlegt jede
Antwort in Einzelaussagen und liefert pro Antwort den **Anteil** der durch
den Kontext gedeckten Aussagen (0–1), statt eines Ja/Nein pro Antwort.
Einmalig über alle 33 Testfragen gelaufen (frische Pipeline-Antworten nach
der Synonym-Expansion), mit zwei unterschiedlich starken Judge-Modellen:

| Judge-Modell | Faithfulness (Mittel) | Antworten mit 1.0 | Antworten ≥ 0.8 |
|---|---|---|---|
| Claude Haiku 4.5 | 0.96 | 76 % | 94 % |
| Claude Sonnet 4.5 (stärker) | 0.93 | 70 % | 88 % |

Einfach gelesen: Von 100 Einzelaussagen einer Antwort sind im Schnitt 93–96
direkt im gezeigten Kontext belegt. Das widerspricht der 100-%-Zahl des
eigenen Judges nicht — der urteilt binär pro ganzer Antwort und toleriert
Übergangs- und Zusammenfassungssätze, die Ragas auf Aussagen-Ebene als
„nicht direkt gedeckt" zählt. Erwartbar und beruhigend zugleich: der
stärkere Judge ist strenger (0.93 statt 0.96) und findet mehr Grenzfälle —
ein Hinweis, dass die eigene 100-%-Zahl die freundlichste Lesart ist, nicht
die einzige. Rohdaten: `eval/ragas_results_haiku.json` und
`eval/ragas_results_sonnet.json`; Ragas lief bewusst in einem separaten
venv und ist keine Projekt-Abhängigkeit.

## Laufzeit-Faithfulness-Check (deterministisch)

Der LLM-Judge misst offline; zur Laufzeit prüfte lange nichts, ob eine
Antwort wirklich durch die zitierten Quellen gedeckt ist — die Zitierpflicht
stand nur im System-Prompt. Das schließt `app/faithcheck.py`: nach jedem
Antwort-Stream wird die Antwort in Sätze zerlegt und jeder Satz per
Token-Overlap (Schwelle 0.5) gegen die zitierten `[n]`-Quellen geprüft —
deterministisch, ohne zusätzlichen LLM-Call, unbestechlich. Das Ergebnis
geht als eigenes SSE-Event ans Frontend und erscheint dort als
aufklappbares Panel „Aussagen-Prüfung" mit Ampel pro Satz: ✅ wörtlich
belegt, 🟡 sinngemäß oder ohne Zitat, 🔴 nicht gedeckt oder ungültiges
Zitat. Die Antwort wird nicht blockiert, nur transparent gemacht.

![Chat-Antwort mit Quellenangaben, gefundenen Hilfeseiten und Aussagen-Prüfung](docs/img/chat-antwort.png)

Bewusste Grenze: Token-Overlap ist ein grobes Maß — inhaltlich korrekte
Paraphrasen erscheinen gelb. Das ist Absicht: der Validator misst, statt zu
vertrauen. Derselbe Validator (gleiche Tokenisierung, gleiche Schwelle)
läuft auch im Schwesterprojekt „Handover Brief Generator" — dort
blockierend mit Retry-Logik, hier anzeigend.

## Übergabe an Menschen (Handover-Briefing)

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

## Die Übergabe-Demo

Der Tab „Übergabe-Demo" führt die Produktstory ohne Vorwissen vor: drei
gestellte Sackgassen-Szenarien („Uhr nicht angekommen", „Zollfrage aus der
Schweiz", „Widersprüchliche Angaben"), durchklickbar in Akten. Links der
Nachrichten läuft pro Betreuer eine farbige Zeitachsen-Lane (Bot blau,
Tier-1 grün, Tier-2 bernstein) — am ◆-Meilenstein endet die alte Lane und
die Übergabe erzeugt **live** ein echtes Briefing über `POST /api/handover`.
Die Validierung ist dabei sichtbar inszeniert: jede Briefing-Zeile erscheint
erst als „wird geprüft", dann klappt das Ampel-Ergebnis mit Beleg-Zitat ein.
Nach erfolgreichem Briefing löst ein gestellter Schlussakt den Fall auf
(„✓ Fall gelöst"). Szenario 3 zeigt zusätzlich eine interne Eskalation
Tier-1 → Tier-2. Ein Kino-Modus („▶ Automatisch abspielen") spielt den
kompletten Fall inklusive Briefing selbstständig ab.

Die Chat-Verläufe der Szenarien sind gestellt und als solche markiert; die
Briefings darin sind es nicht — sie werden bei jedem Klick live erzeugt und
validiert.

## Scraping-Ethik

`robots.txt` von chrono24.de wurde vor dem Scrape-Lauf geprüft: für
`User-agent: *` ist `/info/` nicht disallowed — kein Eintrag der
Disallow-Liste erfasst `/info/*`. `Crawl-delay: 0.1` verlangt mindestens
0,1 s zwischen Requests; der Scraper hält sich an 1 Request/Sekunde, also
deutlich höflicher als gefordert. Der Lauf war ein einmaliger lokaler
Vorgang (21 Seiten, mit echtem Chrome statt headless Chromium, weil
headless auf eine Cloudflare-Challenge lief) — der deployte Server scrapt
nie, er liest ausschließlich den versionierten Index.

## Lokal starten

```bash
python -m venv .venv
.venv/Scripts/pip install -r requirements-dev.txt
# .env anlegen (siehe .env.example) und ANTHROPIC_API_KEY eintragen
.venv/Scripts/uvicorn app.main:app --port 8000
```

Danach `http://localhost:8000` im Browser öffnen.

Chroma verändert Index-Dateien unter `data/index/` schon beim bloßen Öffnen
(z. B. durch lokales Starten oder Testläufe) — solche unstaged Änderungen
mit `git restore data/index/` verwerfen, bevor committet wird.

## Pipeline neu bauen

Nur nötig, wenn sich die Chrono24-Hilfeseiten geändert haben oder der Index
neu erzeugt werden soll:

```bash
python -m pipeline.scrape && python -m pipeline.parse && python -m pipeline.index
```

Bei einem chromadb-Upgrade (Version in `requirements.txt`) muss der Index
danach neu gebaut werden — das committete Format ist an die gepinnte Version
gekoppelt.

## Tests

```bash
pytest tests/
```

Retrieval-Qualität messen:

```bash
python -m eval.run_eval
```

### CI Eval Gate

Zwei automatisierte Qualitäts-Regressionstests in `.github/workflows/ci.yml`,
zusätzlich zu ruff/pytest/Docker-Build:

- **`eval-gate`** (jeder PR und jeder Push auf main, keine API-Kosten): lädt
  den committeten Index und prüft Hit-Rate@5 gegen Tuning- und
  Holdout-Fragen sowie die Abstention-Rate gegen themenfremde Fragen
  (`eval/questions_offtopic.json`). Unter der jeweiligen Mindestschwelle
  (`eval/run_eval.py::TUNING_MIN_HIT_RATE` / `HOLDOUT_MIN_HIT_RATE` /
  `MIN_ABSTENTION_RATE`) schlägt der Job fehl. `MIN_ABSTENTION_RATE` steht
  bei 35 % (gemessen 50 %) — siehe [Konfidenz-Gate für themenfremde Fragen](#konfidenz-gate-für-themenfremde-fragen)
  für die Herleitung und den Puffer.
- **`quality-gate`** (nur bei Push auf main, kostet Haiku-API-Calls): lässt
  `eval/judge.py --gate` über alle Tuning-Fragen laufen und prüft die
  Faithful-Rate gegen `eval/judge.py::MIN_FAITHFUL_RATE`. Braucht das
  Repo-Secret `ANTHROPIC_API_KEY`. Ein Push, der nur Markdown oder `docs/`
  anfasst, überspringt den Lauf — ~66 Haiku-Calls für eine Änderung, die
  die Antwortqualität nicht berühren kann, sind rausgeworfenes Geld. Lässt
  sich der Commit-Bereich nicht auflösen (erster Push, Force-Push), läuft
  der Judge trotzdem: lieber einmal zu viel messen als eine Regression
  verpassen.

Beide Skripte laufen auch lokal manuell:

```bash
python -m eval.run_eval --gate
python -m eval.judge --gate
```

## Guards

Öffentliche Demo, deshalb zwei Schutzmechanismen gegen Missbrauch/Kosten:

- **Rate-Limit:** 10 Anfragen/Minute und 50 Anfragen/Tag pro IP.
- **Tagesbudget:** 200.000 Tokens/Tag global, danach liefert `/api/chat`
  `429` bis zum nächsten Tag.

Das Rate-Limit greift pro Client-IP (`get_remote_address`). Hinter einem
Reverse Proxy wie auf Render sieht uvicorn ohne Weiteres nur die
Proxy-IP — damit teilen sich dann alle Besucher:innen einen einzigen
Rate-Limit-Eimer. Deshalb läuft uvicorn dort mit
`--forwarded-allow-ips="*"` (siehe Dockerfile-`CMD`), damit es die
`X-Forwarded-For`-Adresse des Proxys übernimmt. Lokal ohne Proxy ist das
wirkungslos, weil Clients den Server dort direkt erreichen. Trade-off: der
`X-Forwarded-For`-Header ist grundsätzlich spoofbar, das Rate-Limit ist also
kein hartes Sicherheitsnetz — das globale Tagesbudget bleibt der eigentliche
Kostendeckel, unabhängig von der IP.

## Deployment (Render)

Läuft als Docker-Runtime (siehe `Dockerfile`) auf Render.

- **RAM-Bedarf:** Embedding-Modell plus Cross-Encoder-Reranker brauchen
  zusammen deutlich mehr als die 512 MB des Render-Free-Tiers — für die
  Demo ist ein Tier mit ≥ 1 GB nötig oder alternativ Hugging Face Spaces
  (Docker).

- **Env-Var:** `ANTHROPIC_API_KEY` muss gesetzt sein — ohne ihn startet der
  Service gar nicht erst (fail-fast beim Boot statt kaputter Antworten zur
  Laufzeit).
- **Health-Check-Pfad:** `/api/health`.
- **Kaltstart:** auf dem Render-Free-Tier ca. 30 s, weil der Container nach
  Inaktivität einschläft.
- **Proxy:** `--forwarded-allow-ips="*"` ist im Dockerfile-`CMD` gesetzt,
  siehe Begründung im Guards-Abschnitt oben.
- **Tagesbudget ist nicht persistent:** der Token-Zähler liegt in SQLite auf
  dem ephemeren Dateisystem des Containers und resettet bei jedem Neustart
  oder Redeploy — das Tagesbudget ist also kein verlässlicher Kostendeckel
  über einen Neustart hinweg, sondern nur innerhalb einer laufenden
  Instanz.
