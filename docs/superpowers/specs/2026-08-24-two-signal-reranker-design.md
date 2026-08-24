# Design: Zwei-Signal-LLM-Reranker (Rang + Konfidenz)

**Stand:** 2026-08-24 · Vorgänger: `HANDOVER-llm-reranker.md`

## Problem

Der LLM-Reranker mit reiner Rangfolge löst alle drei bisherigen Dauer-Misses
(Tuning-Hit-Rate@5: 91 % → 100 %), reißt aber die Abstention-Rate von 100 % auf
36 % ein. Grund: Rangfolge ist ein relatives Signal (bester von zehn Kandidaten
bekommt immer einen hohen Rang, auch wenn keiner davon relevant ist) und trägt
keine absolute Konfidenz. Das bestehende `RERANK_THRESHOLD`-Gate setzt aber
„hoher Score = sicher relevant" voraus.

Sortier-Signal und Konfidenz-Signal sind zwei verschiedene Dinge. Diese Spec
entwirft einen Prompt, der beides liefert, plus die Kalibrierungs- und
Testmethodik dafür.

**Scope:** Nur Design + Testplan + Kalibrierungsmethodik (Handover-Schritte
1–3). Die Integrationsentscheidung (Ersatz des Cross-Encoders vs. Kaskade
dahinter, Tagesbudget-Neukalkulation) ist explizit **nicht** Teil dieser Spec
— die folgt erst nach den hier beschriebenen Messungen.

## Architektur / Datenfluss

Ein LLM-Call ersetzt den bisherigen reinen Rang-Call. Voraussetzung bleibt der
Kandidaten-Union-Fix: Kandidatenmenge ist die Vereinigung der Top-10 aus
Vektor- und BM25-Ranking (`_dedupe_ranking(vector_ranking + bm25_ranking)`),
nicht der RRF-Top-n-Cut aus `app/retrieval.py`. Ohne diesen Fix sieht kein
Reranker die beiden alten Miss-Kandidaten (faq-0098, info-escrow-0007).

Output-Format:

```json
{"ranking": [i, j, k, ...], "top1_confidence": 0-10}
```

Ein Call, kein zweiter Roundtrip — keine zusätzliche Latenz gegenüber der
reinen Rang-Variante (Ø 1,1–1,3 s zusätzliche Latenz bleibt wie gemessen).

`top1_confidence` bezieht sich **nur auf den nach Rangfolge bestplatzierten
Kandidaten**, nicht auf die gesamte Kandidatenmenge. Grund: unabhängige
0–10-Scores pro Kandidat (frühere Messung, verworfen) erzeugten bei
Themenclustern gehäufte Höchst-Scores und einen neuen Miss (DAC7/faq-0162).
Rangfolge übernimmt die Sortierarbeit, Konfidenz bewertet nur noch das
bereits sortierte Ergebnis — das umgeht den Fehlermodus strukturell.

Skala: unteres Ende fest verankert („0 = beantwortet die Frage überhaupt
nicht, auch nicht der beste Kandidat"), oberes Ende ebenfalls fest verankert
(„10 = beantwortet die Frage vollständig und eindeutig"), damit die Skala
zwischen Calls nicht driftet. Exakter Prompt-Wortlaut wird bei der
Implementierung ausformuliert, nicht in dieser Spec.

## Kalibrierungs-Methodik

Gleiches Verfahren wie beim Cross-Encoder-Finetune: on-topic-Minimum vs.
off-topic-Maximum über das bestehende Eval-Set (33 Tuning + 15 Holdout + 14
Abstention) messen, `RERANK_THRESHOLD` auf den neuen `top1_confidence`-Wert
legen — nicht mehr auf den Rang-Score. Messung mit `rerank_threshold=-999`,
damit die alte, für dieses Signal falsch kalibrierte Schwelle die
Rohmessung nicht verfälscht.

## Testplan

Zwei Stufen, wie im Vorexperiment (erst 4 Calls, dann 66):

1. **Stufe 1, billig:** die 4 bekannten Problemfälle (faq-0098,
   info-escrow-0007, faq-0033, faq-0162) plus eine kleine Off-Topic-
   Stichprobe aus dem Abstention-Set. Prüft, ob `top1_confidence` bei
   echten Treffern hoch und bei Off-Topic-Anfragen niedrig ausfällt, bevor
   der teure volle Lauf folgt.
2. **Stufe 2, voller Eval:** Tuning-Hit-Rate@5, Holdout-Hit-Rate@5,
   Abstention-Rate mit dem neuen Design messen, gegen die Ausgangstabelle
   vergleichen:

   | Metrik | Finetune-Cross-Encoder | Rang-only (verworfen) | Zwei-Signal (Ziel) |
   |---|---|---|---|
   | Tuning-Hit-Rate@5 | 91 % (30/33) | 100 % (33/33) | ≥ 100 % |
   | Holdout-Hit-Rate@5 | 100 % (15/15) | 100 % (15/15) | 100 % |
   | Abstention-Rate | 100 % (14/14) | 36 % (5/14) | möglichst nahe 100 % |

## Fehlerbehandlung

Robustheitsfix aus dem Vorexperiment übernehmen: exakte Kandidatenzahl im
System-Prompt nennen, Beispiel-Array mitgeben, `max_tokens=400` (Haiku
lieferte bei knapperem Budget teils nur Teil-Rangfolgen).

Zusätzlich, neu gegenüber dem Rang-only-Experiment: Parsing-Fallback, falls
`top1_confidence` fehlt oder außerhalb 0–10 liegt. In dem Fall konservativ
behandeln (wie Konfidenz 0 / abstain), nicht crashen — ein Malformed-Value
darf nicht zu einem falsch-positiven Treffer führen.

## Reversibilität

Weiterhin `Retriever.retrieve()` monkeypatchen statt `app/retrieval.py` fest
zu ändern, solange nur gemessen und nicht integriert wird. Pattern aus dem
Vorexperiment: private Helfer `_vector_candidates`/`_bm25_candidates` der
aktuellen SQLite-Engine wiederverwenden, nur Kandidaten-Auswahl (Union statt
Top-n-Cut) und Reranker-Callable ersetzen. Hält das Experiment rückbaubar
und das Repo sauber, bis die Integrationsentscheidung (außerhalb dieser
Spec) gefallen ist.

## Kosten-Hinweis (unverändert aus Handover)

Echte, laufende API-Kosten pro Anfrage — anders als der bestehende
Rewrite-Call, der bei deutschen Fragen ohne Chatverlauf entfällt. Bei
Live-Traffic würde ein LLM-Reranker-Call pro Anfrage das Tagesbudget
(`daily_token_budget`, aktuell 200.000) deutlich schneller aufbrauchen.
Bleibt Teil der späteren Integrationsentscheidung, nicht dieser Spec.
