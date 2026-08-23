# Chrono24-FAQ-Chatbot

Ein RAG-Chatbot, der Fragen zu Kauf, Verkauf, Käuferschutz und Versand auf
Basis der öffentlichen Chrono24-Hilfeseiten beantwortet — mit Quellenangaben
und sichtbaren Retrieval-Treffern statt einer Blackbox-Antwort. Was er nicht
belegen kann, übergibt er an Menschen: als automatisch erzeugtes Briefing,
dessen Aussagen ein deterministischer Validator gegen den Chatverlauf prüft.

**Demo-Link:** folgt (Deploy steht noch aus).

![Übergabe-Demo: Szenario läuft in die Sackgasse, der Bot übergibt mit live geprüftem Briefing, der Support löst den Fall](docs/img/demo-briefing.png)

**Auf einen Blick:**

- **Hybrid-Retrieval:** BM25 (SQLite FTS5) + Vektorsuche (sqlite-vec),
  RRF-Fusion, Cross-Encoder-Reranker, Synonym-Expansion, LLM-generierte
  Query-Varianten je FAQ — 91 % Hit-Rate@5,
  held-out validiert (100 %)
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
indexer              → data/index/hybrid.db (SQLite: FTS5 + sqlite-vec,
                                              nicht versioniert, Build-Artefakt)

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
| verworfen: `TOP_K_CANDIDATES` 15/20/25/30/40 (erneut, nach den Varianten) | 91 % bis k=25, dann 88 → 85 %; Abstention 50 % → 43 % |
| verworfen: Reranker nur als *ein* Signal (RRF über Rerank- und Fusionsrang) | 91 % (30/33), neutral bis −3 pp |
| verworfen: stärkerer Reranker `bge-reranker-v2-m3` (568M) | 0 von 3 Misses gelöst, Latenz 0,7 s → 6,7–10,9 s |
| verworfen: Embedder `multilingual-e5-base` statt MiniLM (Merge-Schwelle fair kalibriert) | 91 % / 100 %, **exakt dieselben drei Misses** |
| verworfen: Doc2Query — generierte Fragen auch für `page_chunks` (568 Stück) | 91 % / 100 %, unverändert |
| verworfen: FAQ-Kategorie in den **Embedding**-Text (statt nur Rerank-Text) | 91 % / 100 %, dieselben drei Misses |
| verworfen: Kategorie bei allen Dokumenten im Embedding | 91 % / 100 %, dieselben drei Misses |
| verworfen: `RRF_K` 30/10/5/2/1 und Pfad-Gewichte bis 1:2 | Status quo (60, 1:1) ist Optimum; `w_bm=1.5` hebt Tuning auf 100 %, senkt Held-out auf 87 % |
| verworfen: Rollen-Malus über die FAQ-**Kategorie**, weich, nach dem Ranking | 91 % bei Malus 0,7/0,5; 88 % ab 0,3 — **kein harter Test der Rollen-Idee**, siehe unten |
| neutral: `audience`-Feld als harter Pre-Filter, echter Ausschluss vor der RRF-Fusion | 91 % / 100 % — **exakt unverändert, beide Rollen-Misses bleiben Misses**, siehe unten |
| Engine-Wechsel: Chroma + Hand-BM25 (`rank_bm25`) → SQLite (`sqlite-vec` + FTS5), siehe unten | 91 % / 100 % (**exakt dieselben drei Misses**), Abstention 50 % → 57 % (8/14) |
| Reranker-Finetune auf Käufer/Verkäufer-Hard-Negatives (1016 Beispiele, 2 Epochen), siehe unten | 91 % / 100 % — **dieselben drei Misses**, Abstention 57 % → 100 % (14/14) |

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

Die 3 verbleibenden Misses sind einzeln diagnostiziert, und die Diagnose ist
unbequemer als „harte Fälle": **es sind Fehlurteile des Cross-Encoders, nicht
Lücken im Retrieval.** Setzt man `TOP_K_CANDIDATES` auf 25, liegen alle drei
Zieldokumente in der Kandidatenmenge — der Reranker stuft sie nur klar negativ:

| Frage | Gold-Dokument | Rerank-Score | Platz 1 stattdessen |
|---|---|---|---|
| „Muss ich als privater **Verkäufer** etwas bezahlen …?" | faq-0098 „Was kostet der Verkauf?" (nennt die 6,5 % Provision) | −5,65 | faq-0019 „Was muss ich bei Privatverkäufern beachten?" (+6,52) — **Käufersicht** |
| „Worauf muss ich als **Verkäufer** achten, bevor ich … verschicke?" | info-escrow-0007 („prüfen Sie den Zahlungseingang, bevor Sie versenden") | −4,69 | faq-0027 „Der Verkäufer bietet nur Vorkasse an. Ist das vertrauenswürdig?" (+0,74) — **Käufersicht** |
| „Was genau ist das Certified-Programm?" | faq-0033 „Was ist Certified?" | −0,05 | faq-0048 (+4,56), ein spezifischeres Geschwister |

Zwei der drei teilen einen benennbaren Fehlermodus: **Rollenverwechslung
Käufer ↔ Verkäufer.** Beide Fragen sind aus Verkäufersicht gestellt, beide
Gold-Dokumente enthalten die Antwort wörtlich, und der Reranker zieht
stattdessen Dokumente hoch, die dasselbe Vokabular tragen („Privatverkäufer",
„Vorkasse", „Überweisung"), aber die andere Rolle adressieren. Das
mehrsprachige MS-MARCO-Modell kennt diese Unterscheidung schlicht nicht.

**Fünf Auswege wurden gemessen, alle fünf verworfen** (Zahlen in der
Ablationstabelle). Der naheliegendste zuerst, weil er auch der war, den
dieser Abschnitt zwischenzeitlich empfahl:

- **Stärkerer Reranker** (`BAAI/bge-reranker-v2-m3`, 568M statt 118M
  Parameter): löst **null von drei** Misses — faq-0033 rückt von Platz 8 auf
  6, bleibt Miss; die anderen beiden sind bei `TOP_K_CANDIDATES=10` gar nicht
  erst Kandidat. Dazu **6,7–10,9 s Rerank-Latenz** pro Anfrage auf CPU gegen
  0,7–1,0 s beim aktuellen Modell. Für eine Live-Demo untragbar, und der
  Gegenwert ist null.
- **Anderer Embedder** (`intfloat/multilingual-e5-base`, Wegwerf-Index):
  Tuning 91 % → 88 %, Held-out 100 % → 93 %, dieselben drei Misses plus ein
  neuer. Die höheren Cosine-Werte lassen zusätzlich den
  Near-Duplicate-Merge (0.95) Zieldokumente wegräumen — derselbe Effekt wie
  beim Seitentitel-Experiment weiter oben.
- **Mehr Kandidaten**, **Reranker als nur ein Signal**, **RRF-Parameter und
  Pfad-Gewichte** — siehe Tabelle, alle neutral oder schlechter.

Das ändert die Diagnose: Es ist kein Modell-Problem, das ein größeres Modell
löst. **Zwei der drei Misses sind ein Kandidaten-Problem** (das Ziel steht im
Vektor-Ranking auf Platz 76 bzw. 130, nur BM25 findet es auf Platz 8 bzw. 6,
und die RRF-Fusion bevorzugt Dokumente, die in *beiden* Listen stehen), **einer
ist ein Reranker-Problem** (faq-0033 ist Kandidat und wird auf Platz 8
gestuft). Die Kandidaten-Trefferquote@10 liegt bei 94 % Tuning / 100 %
Held-out — sie ist bereits höher als die Endzahl.

#### Warum der Vektorpfad hier versagt

Das Embedding-Modell `paraphrase-multilingual-MiniLM-L12-v2` ist ein
**Paraphrase-Modell**: es misst, wie ähnlich sich zwei Sätze sind — nicht, ob
ein Dokument eine Frage beantwortet. Für Retrieval ist das die falsche
Modellklasse, und man sieht es an den Zahlen. Für
„Muss ich als privater Verkäufer etwas bezahlen, wenn ich meine Uhr verkaufe?":

| Ähnlichkeit | Dokument |
|---|---|
| **0.870** | faq-0121 „Wie versende ich als **Privatverkäufer meine Uhr** sicher?" — teilt die Nomen, beantwortet die Frage nicht |
| **0.495** | faq-0098, dessen Variante „Wie viel muss ich zahlen, wenn ich etwas auf Chrono24 verkaufe?" fast die Frage selbst ist |

Das Modell belohnt Nomen-Überlappung, nicht Intention. Beim
`page_chunk`-Fall kommt Asymmetrie dazu: eine kurze Frage gegen einen langen
Fließtext erreicht **0.371**, während thematisch falsche Kurzfragen 0.77
erreichen.

**Zwei gezielte Gegenmittel gemessen, beide ohne Wirkung:**

- **Doc2Query** — für alle 132 `page_chunks` Fragen generieren lassen, die
  der Chunk beantwortet (568 Stück), und mitembedden. Eine handformulierte
  Frage hatte im Vorversuch 0.371 → 0.692 erreicht, das sah nach dem Fix aus.
  Real gemessen: Rang 130 → 93, Similarity 0.371 → **0.475**, Hit-Rate
  **unverändert**. Der Grund steht im Detail: die generierte Frage lautet
  „Worauf sollte ich prüfen, bevor ich einen **Artikel** … versende?" — sie
  sagt „Artikel", die Nutzerfrage sagt „Uhr". Wer auf Nomen-Überlappung
  angewiesen ist, verliert an einem einzigen Wort.
- **Moderneres Embedding** (`multilingual-e5-base`, asymmetrisch trainiert,
  mit `query:`/`passage:`-Präfixen, Merge-Schwelle auf gleiche
  Duplikat-Anzahl kalibriert): 91 % / 100 % — **exakt dieselben drei Misses**.

Dass ein komplett anderes Modell an denselben drei Fällen scheitert, ist das
stärkste Argument dafür, dass hier kein Modell von der Stange mehr hilft.

**Diese Idee wurde inzwischen hart getestet — und hat die beiden Rollen-Misses
nicht gelöst.** Der oben verworfene Rollen-Malus benutzte die FAQ-Kategorie
als Rollen-Ersatz und zog Punkte *nach* dem Ranking ab. Beides war zu schwach
für einen ehrlichen Test: 132 der 318 Dokumente tragen überhaupt keine
Kategorie — darunter ausgerechnet info-escrow-0007, einer der beiden
Rollen-Fälle. Der Malus konnte dieses Dokument gar nicht erreichen.

Als härterer, anderer Mechanismus (Design in
`docs/superpowers/specs/2026-08-23-corpus-storage-rethink-design.md`): ein
eigenes `audience`-Feld (`kaeufer`/`verkaeufer`/`neutral`) auf *allen* 318
Dokumenten, per Wortstamm-Heuristik (`classify_audience()` in
`app/textproc.py`, gleiches Muster wie `looks_german()`) plus einer kleinen
Hand-Korrekturliste (`pipeline/populate_audience.py`, Prinzip wie
`QUERY_SYNONYMS`) für Fälle, in denen die Heuristik die Gegenpartei öfter
zählt als die eigentliche Rolle — z. B. alle 9 `info-escrow-*`-Chunks (die
Seite hat einen Käufer- und einen separaten Verkäufer-Abschnitt) und, mit
Blick auf die Tabelle oben, ausgerechnet die beiden falschen Konkurrenz-
Dokumente selbst: faq-0019 und faq-0027 nennen "Verkäufer"/"Privatverkäufer"
öfter als "kaufen", obwohl beide klar aus Käufersicht geschrieben sind, und
wurden ohne Korrektur fälschlich als `verkaeufer` eingestuft — der Filter
hätte sie sonst gar nicht als Konkurrenten erkannt. `Retriever.retrieve()`
schneidet damit beide Ranglisten (BM25 und Vektor) *vor* der RRF-Fusion auf
Dokumente der passenden Rolle zu — echter Ausschluss, kein Score-Abzug. Ein
eigener Testfall pinnt den Mechanismus (`tests/test_retrieval.py::
test_retrieve_with_audience_excludes_wrong_role`): ein Verkäufer-Dokument
verschwindet komplett, sobald `audience="kaeufer"` gesetzt ist.

Gemessen mit aktivem Filter (`eval.run_eval.hit_rate_at_k_with_audience`,
Query-Rolle per `classify_audience()`): **Tuning 91 % (30/33), Holdout 100 %
(15/15) — exakt dieselben Zahlen wie ohne Filter, und dieselben drei Misses**,
faq-0098 und info-escrow-0007 eingeschlossen. Der Filter tut nachweislich,
was er soll — faq-0019 und faq-0027 verschwinden aus den Kandidatenlisten
beider Anfragen —, aber das reicht nicht:

- Für „Muss ich als privater Verkäufer etwas bezahlen …?" bleibt faq-0098
  auch nach dem Filter *Kandidat* (letzter Platz der RRF-Fusion, nur über den
  BM25-Pfad), verliert aber gegen andere, korrekt als `verkaeufer`
  eingestufte Dokumente (faq-0121, faq-0124, faq-0102 u. a.) — kein
  Rollenproblem mehr, sondern dieselbe Reranker-Schwäche, die oben schon für
  faq-0033 diagnostiziert ist.
- Für „Worauf muss ich als Verkäufer achten, bevor ich … verschicke?" schafft
  es info-escrow-0007 nach dem Filter nicht einmal mehr in die Top-10 der
  RRF-Fusion: es liegt exakt gleichauf mit einem anderen Kandidaten auf dem
  letzten Platz und verliert den Tiebreak an die Einfüge-Reihenfolge des
  Vektor-Pfads (der zuerst verarbeitet wird). Eine Handvoll Punkte Marge, kein
  strukturelles Versagen des Filters — aber eben auch kein Treffer.

Der harte Filter behebt also nicht das ursprünglich diagnostizierte
Kandidaten-Problem (README oben: „das Ziel steht im Vektor-Ranking auf Platz
76 bzw. 130"), weil `TOP_K_CANDIDATES=10` beide Zieldokumente vektorseitig
ohnehin nie erreicht — der Filter kann nur entfernen, was in den bereits auf
n Plätze gekappten Ranglisten steht, nicht tiefer liegende Kandidaten
nachrücken lassen. Ehrliches Fazit: der Mechanismus ist sauber implementiert
und per Unit-Test bewiesen wirksam, löst aber auf dem tatsächlichen Eval-Set
keinen der beiden Fälle, für die er gebaut wurde. Der Code bleibt additiv im
Repo (`audience`-Parameter mit Default `None`, ohne Wirkung auf bestehende
Aufrufer), weil er keine Regression verursacht — nur eben auch keinen Gewinn.

Was darüber hinaus bliebe, ist Domänen-Finetuning auf
Chrono24-Frage/Antwort-Paaren — das wurde inzwischen ohne auf echten Traffic
zu warten versucht, siehe nächster Abschnitt.

#### Reranker-Finetune auf Käufer/Verkäufer-Hard-Negatives

Statt auf echten Traffic zu warten: 1016 synthetische Trainingsbeispiele aus
den bestehenden FAQ-Kategorien gebaut (Käufer-Kategorien wie „Uhren kaufen …"
gegen Verkäufer-Kategorien wie „Privat/Gewerblich Uhren verkaufen …", je
Positiv-Paar 2 Hard-Negatives aus der jeweils anderen Rolle mit hoher
Wortüberlappung plus ein generisches Zufalls-Negativ). Die bestehenden
LLM-Query-Varianten (`data/variants.json`, ohnehin schon bezahlt) liefern
pro FAQ bis zu 5 zusätzliche Formulierungen statt nur der einen Originalfrage
— das allein versechsfacht die Trainingsmenge gegenüber einem ersten,
kleineren Testlauf (162 Beispiele, 1 Epoche, siehe unten). Alle 48
Eval-Zieldokumente (Tuning + Holdout) sind aus dem Training komplett
ausgeschlossen, weder als Positiv- noch als Negativ-Beispiel — die Messung
bleibt unabhängig vom Training. Train/Val-Split auf Dokument-Ebene (80/20),
damit keine Variante derselben FAQ in beiden Töpfen landet.
`cross-encoder/mmarco-mMiniLMv2-L12-H384-v1` (Basismodell) 2 Epochen
`BinaryCrossEntropyLoss` trainiert, bestes Checkpoint nach Val-Accuracy
(95,3 %, F1 0,90) übernommen.

**Ergebnis, gemessen gegen den finalen Indexstand (SQLite-Engine + aktiver
Audience-Filter):** Tuning 91 % (30/33), Holdout 100 % (15/15) — **exakt
dieselben drei Misses wie ohne Finetune**, faq-0098 und info-escrow-0007
eingeschlossen. Auch mit gezieltem Training auf genau dieses Rollenmuster
bleibt der Reranker bei diesen zwei Fällen bei seinem Fehlurteil: score-mäßig
verlieren die Zieldokumente sogar deutlicher gegen ihre Konkurrenten als beim
Basismodell. Naheliegende Erklärung: beide Zieldokumente waren bewusst vom
Training ausgeschlossen (Eval-Integrität), das Modell musste das
Rollen-Konzept auf neue, ungesehene Dokumente generalisieren — und hat das
bei genau diesen zwei besonders vokabular-überlappenden Fällen nicht
geschafft. Ein erster Testlauf (162 Beispiele, 1 Epoche, ohne Split) hatte
unter dem alten Chroma-Indexstand einen scheinbaren Nebengewinn gezeigt (ein
vierter, bis dahin ungelöster Miss verschwand) — dieser Effekt reproduziert
sich auf dem aktuellen, per `audience`-Feld und Dedupe-Rebuild veränderten
Corpus-Stand nicht und war damit vermutlich an den damaligen Indexstand
gebunden, kein robuster Fix.

Echter, reproduzierbarer Gewinn liegt woanders: die Rerank-Score-Verteilung
ist nach dem Finetune deutlich sauberer getrennt. On-topic-Minimum 3,01 vs.
Off-topic-Maximum 2,38 (48 on-topic-, 14 Off-Topic-Fragen) — eine klare Lücke
allein auf Basis des Rerank-Signals, verglichen mit −5,6 vs. −0,19 beim
Basismodell, das dafür noch zwei weitere Signale (Cosine-Sim, BM25) brauchte.
`RERANK_THRESHOLD` entsprechend neu kalibriert (2,9, knapp unter dem
on-topic-Minimum). Damit steigt die Abstention-Rate auf **100 % (14/14)** —
gegenüber 50–57 % vorher der stärkste Einzeleffekt in dieser gesamten
Optimierungsreihe, allerdings auf einem kleinen Sample (95-%-Intervall
[78 %, 100 %]) mit entsprechendem Vorbehalt.

**Deployment-Konsequenz:** das Modell (470 MB, `cross-encoder/
mmarco-mMiniLMv2-L12-H384-v1`-Architektur mit trainierten Gewichten) liegt in
einem privaten Hugging-Face-Hub-Repo (`VoidFloat/chrono24-faq-reranker`) —
zu groß für einen normalen Git-Commit (GitHub blockt Dateien > 100 MB ohne
LFS), und ein öffentliches Repo hätte das trainierte Gewicht ohne Not offen
gelegt. Der Server braucht deshalb `HF_TOKEN` als zusätzliche Env-Var (siehe
Deployment-Abschnitt), sonst schlägt das Laden des Rerankers beim Boot fehl.

#### Engine-Konsolidierung: Chroma + Hand-BM25 → SQLite (FTS5 + sqlite-vec)

Schritt 2 desselben Designs (`docs/superpowers/specs/
2026-08-23-corpus-storage-rethink-design.md`) tauscht die Speicher-Engine,
nicht das Retrieval-Verhalten: Chroma (Vektor-Index) und ein von Hand
gepflegter `rank_bm25.BM25Okapi`-Index (`data/index/bm25.pkl`) werden durch
eine einzige SQLite-Datei (`data/index/hybrid.db`) ersetzt — eine
`vec0`-Virtualtabelle (`sqlite-vec`) für die Vektorsuche, eine
FTS5-Virtualtabelle für BM25 nativ. Zwei synchronisierte Doc-ID-Räume werden
zu einem Schema; der harte `audience`-Filter aus Schritt 1 wandert von einem
Python-seitigen Nachfilter zu einem SQL-`WHERE` auf einer `PARTITION KEY`-
Spalte, direkt in beiden Teil-Queries.

Zwei Stellschrauben mussten dabei neu vermessen werden, weil sich die
zugrundeliegende Engine ändert, nicht weil sich das Modell oder der Corpus
ändern:

- **`SIM_THRESHOLD` (0.40) blieb unverändert.** `vec0` mit
  `distance_metric=cosine` liefert für normalisierte Vektoren exakt
  `1 − Cosine-Similarity` — gemessen identisch zum alten Chroma-Wert (on-topic
  Similarity-Minimum 0.607 vor und nach dem Wechsel, auf die dritte
  Nachkommastelle gleich).
- **`BM25_THRESHOLD` musste neu kalibriert werden.** FTS5s eingebautes
  `bm25()` hat eine andere Skala und ein anderes Vorzeichen als
  `rank_bm25.BM25Okapi.get_scores()` (kleinerer/negativerer Wert =
  relevanter). Der Retriever negiert den FTS5-Score bei der Abfrage
  (`best_bm25 = -bm25(...)`), damit „höher = relevanter" überall im Code
  gültig bleibt — die Zahl selbst ist trotzdem neu, weil FTS5s
  BM25-Implementierung nicht exakt dieselben Werte wie `rank_bm25` produziert.
  Gemessen auf demselben 48-Fragen-Set (Tuning + Holdout) wie oben: on-topic
  BM25-Minimum 6.29 (vorher 6.36 — nahezu identisch trotz anderer
  Bibliothek). Schwelle bei 5.5 belassen (zufällig derselbe Zahlenwert wie vor
  dem Wechsel, weil beide Bibliotheken dieselbe Okapi-BM25-Formel in
  ähnlicher Größenordnung berechnen).

Gemessen mit dem committeten Reranker-Modell
(`cross-encoder/mmarco-mMiniLMv2-L12-H384-v1`) gegen einen Wegwerf-Index
außerhalb des Repos, vor/nach dem Wechsel:

| Metrik | Vorher (Chroma + `rank_bm25`) | Nachher (SQLite: FTS5 + `sqlite-vec`) |
|---|---|---|
| Tuning-Hit-Rate@5 | 91 % (30/33) | 91 % (30/33) — **exakt dieselben drei Misses** |
| Holdout-Hit-Rate@5 | 100 % (15/15) | 100 % (15/15) |
| Abstention-Rate | 50 % (7/14) | 57 % (8/14) |

Die Abstention-Rate bewegt sich von 7/14 auf 8/14 — ein zusätzlicher
False-Hit weniger. Bei 14 Fragen liegt eine einzelne Frage bei 7
Prozentpunkten; das 95-%-Wilson-Intervall für beide Werte überlappt
deutlich (7/14: [27 %, 73 %]; 8/14: [33 %, 79 %]), die Differenz ist also
nicht von Rauschen zu unterscheiden. Ehrlich berichtet als „kein Nachteil,
vermutlich kein echter Vorteil" statt als Verbesserung verkauft.

Strukturell hat der Wechsel den dokumentierten Schmerzpunkt „Chroma mutiert
committete Indexdateien beim bloßen Öffnen" beseitigt — nicht durch
Disziplin (`git restore data/index/` vor jedem Commit), sondern durch
Wegfall der Prämisse: `data/index/` ist seit diesem Wechsel nicht mehr
versioniert (siehe `.gitignore`), sondern wird bei Bedarf lokal bzw. im
Docker-Build und in CI neu erzeugt (`python -m pipeline.index`, kostenlos —
das Embedding-Modell läuft lokal).

### Held-out-Validierung

Alle Zahlen oben stammen vom selben 33-Fragen-Set, das auch für jede
Tuning-Entscheidung genutzt wurde — das Risiko, unbewusst auf dieses Set hin
zu optimieren, ist real. Als Gegenprobe gibt es `eval/questions_holdout.json`:
15 neue, nie fürs Tuning verwendete Fragen zu Dokumenten, die im
Tuning-Set nicht als Ziel vorkommen (11 FAQ-, 4 Seiten-Chunk-Ziele, 2
englisch), einmalig gegen die finale Konfiguration gemessen:
**100 % (15/15)**. Das liegt über der Tuning-Zahl (91 %) — kein Anzeichen
für schweres Eval-Set-Overfitting, weil ein System, das nur auf die 33
Tuning-Fragen zugeschnitten wäre, auf neuen Fragen deutlich stärker
einbrechen würde.

Die Entwicklung dieser Zahl gehört dazu: 87 % (13/15) beim ersten Lauf,
93 % nach dem Überfetch-Fix des Vektorpfads, und 100 % erst, seit die Eval
denselben Pfad misst wie der Live-Bot. **Die letzten 7 Punkte sind kein
Fortschritt am System, sondern eine korrigierte Messung.** Der Live-Bot
schickt nicht-deutsche Fragen erst durch `rewrite_query`, bevor sie das
Retrieval sehen — BM25 arbeitet auf einem deutschen Korpus, eine englische
Query bekommt dort einen Score von exakt 0. Die Eval tat das nicht und
zählte „Where do I actually have to pay the customs duties…?" als Miss,
obwohl der Bot das Dokument in Produktion findet. Sie maß einen Pfad, den
es nirgends gibt.

Damit der `eval-gate`-Job weiterhin ohne API-Kosten läuft, sind die fünf
Umformulierungen einmalig offline erzeugt und liegen als Feld `rewritten`
in den Fragen-JSONs — dasselbe Muster wie `data/variants.json`. Die
Konsequenz steht offen dabei: es sind eingefrorene Umformulierungen, kein
Live-Call. Ändert sich der Rewrite-Prompt, muss das Feld neu erzeugt
werden, sonst misst die Eval wieder etwas anderes als Produktion.

### Was diese Zahlen aushalten (Konfidenzintervalle)

Alle Trefferquoten oben stammen aus kleinen Stichproben, und das begrenzt,
was man aus ihnen lesen darf. 95-%-Wilson-Intervalle:

| Zahl | Wert | 95-%-Intervall | Breite |
|---|---|---|---|
| Tuning-Hit-Rate@5 | 91 % (30/33) | [76 %, 97 %] | 20 pp |
| Held-out-Hit-Rate@5 | 100 % (15/15) | [80 %, 100 %] | 20 pp |
| Abstention-Rate | 50 % (7/14) | [27 %, 73 %] | 46 pp |

Im Klartext: **„100 % auf dem Held-out-Set" heißt statistisch „mindestens
80 %"** — 15 Fragen können nicht mehr belegen. Und die Abstention-Rate ist
mit 46 Punkten Breite kaum mehr als eine Richtungsangabe. Der Unterschied
zwischen 88 % und 94 % auf dem Tuning-Set, über den in diesem README
mehrfach diskutiert wird, liegt vollständig innerhalb des Rauschens.

Das ist auch der Grund, warum die Ablationstabelle so viele Einträge mit
„neutral" trägt: bei 33 Fragen ist eine Änderung erst ab etwa 4 Fragen
Unterschied überhaupt vom Zufall unterscheidbar. Alles darunter ist nicht
messbar, egal wie plausibel die Idee klingt.

Was das kosten würde zu beheben — bei gleicher Trefferquote von 91 %:

| Fragen | Intervall | Breite |
|---|---|---|
| 33 (heute) | [76 %, 97 %] | 20 pp |
| 100 | [84 %, 95 %] | 11 pp |
| 150 | [85 %, 94 %] | 9 pp |

Damit das nicht im Kleingedruckten verschwindet, druckt jeder Eval-Lauf das
Intervall mit — und markiert selbst, wenn die eigene Zahl zu dünn belegt ist
(`eval/stats.py`, Schwelle 20 Punkte Breite):

```text
Tuning-Hit-Rate@5:  91% (30/33, 95%-KI [76%, 97%])  <- Stichprobe zu klein: 20% Punkte breit
Holdout-Hit-Rate@5: 100% (15/15, 95%-KI [80%, 100%])  <- Stichprobe zu klein: 20% Punkte breit
Abstention-Rate:    50% (7/14, 95%-KI [27%, 73%])  <- Stichprobe zu klein: 46% Punkte breit
```

Alle drei Zahlen dieses Projekts stehen aktuell unter dieser Warnung. Das ist
gewollt sichtbar: eine Kennzahl, die ihre eigene Unsicherheit verschweigt, ist
in einem Projekt über Belegbarkeit die falsche Kennzahl.

Der ehrlichste offene Punkt dieses Projekts ist damit nicht die Hit-Rate,
sondern die Stichprobengröße — Plan dafür in
`HANDOVER-eval-set-groesse.md`.

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

Hit-Rate bleibt unverändert (damals 91 % / 93 %). Die 7 Durchrutscher sind genau
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

Ein Einzellauf trägt also nicht. Deshalb fünf Läufe am Stück, pro Frage
ausgezählt — das trennt echte Fehler von Judge-Rauschen sauber:

| | |
|---|---|
| Faithful-Rate je Lauf | 97 / 94 / 85 / 88 / 97 % |
| Mittel über 5 Läufe | **92,1 %** |
| Fragen nie beanstandet | 27 von 33 |
| Fragen in 4 von 5 Läufen beanstandet | 2 |
| Fragen in genau 1 von 5 Läufen beanstandet | 3 |

Die drei „1 von 5"-Fälle sind Rauschen — sie werden in den anderen vier
Läufen als „voll" und treu bewertet. Die beiden hartnäckigen Fälle sind
echt, und einer davon ist aufschlussreich: **„What exactly is the Certified
program on Chrono24?" wird 4× als nicht treu und 4× als „nein beantwortet"
gewertet — das ist einer der drei Retrieval-Misses von oben.** Der Bot kann
nicht belegen, was das Retrieval ihm nie gezeigt hat. Dasselbe gilt
abgeschwächt für die Überweisungs-Frage (2 von 5, davon 3× „verweigert").

Retrieval-Fehler und Faithfulness sind hier also keine getrennten Metriken:
zwei der drei reproduzierbaren Faithfulness-Treffer sind direkte Folge der
drei Retrieval-Misses. Wer die Reranker-Fehlurteile oben behebt, hebt beide
Zahlen gleichzeitig.

Rohdaten des ersten Laufs in `eval/judge_results.json`, die späteren stehen
in den CI-Logs des `quality-gate`-Jobs.

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
python -m pipeline.index   # baut data/index/hybrid.db aus data/corpus.json
.venv/Scripts/uvicorn app.main:app --port 8000
```

Danach `http://localhost:8000` im Browser öffnen.

`data/index/` ist nicht versioniert (siehe `.gitignore`) — der erste Start
(und jeder Testlauf) braucht deshalb einmal `python -m pipeline.index`, sonst
findet `Retriever.__init__` keine `hybrid.db`. Frühere Versionen committeten
den Index; Chroma veränderte die Datei dabei schon beim bloßen Öffnen, was
einen `git restore data/index/`-Tanz vor jedem Commit erzwang. Mit dem
Wechsel auf SQLite (FTS5 + sqlite-vec) und dem Wegfall des Commits entfällt
das Problem durch Wegfall der Prämisse, nicht durch Disziplin.

## Pipeline neu bauen

Nur nötig, wenn sich die Chrono24-Hilfeseiten geändert haben oder der Index
neu erzeugt werden soll (kostenlos — das Embedding-Modell läuft lokal, kein
API-Call):

```bash
python -m pipeline.scrape && python -m pipeline.parse && python -m pipeline.index
```

Ein laufender `uvicorn`-Prozess hält `data/index/hybrid.db` unter Windows
offen (SQLite-Datei-Lock) — vor einem Reindex den Server stoppen, sonst
schlägt das `unlink()` im Build fehl.

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

- **`eval-gate`** (jeder PR und jeder Push auf main, keine API-Kosten): baut
  den Index lokal (`python -m pipeline.index`, Embedding läuft ohne API-Call)
  und prüft Hit-Rate@5 gegen Tuning- und
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
- **Env-Var `HF_TOKEN`:** seit dem Reranker-Finetune (siehe „Warum
  Hybrid-RAG") nötig, weil `VoidFloat/chrono24-faq-reranker` ein privates
  Hugging-Face-Hub-Repo ist. Ohne Token schlägt das Laden des Rerankers beim
  Boot fehl — ein Fine-grained-Token mit `read`-Recht auf das Repo genügt.
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
