import json

import pytest

from app.retrieval import RetrievedDoc, Retriever, rrf_fuse
from pipeline.index import build_index

DOCS = [
    {"id": "faq-0001", "type": "faq", "question": "Wie funktioniert der Käuferschutz?",
     "answer": "Der Käuferschutz sichert deine Zahlung ab.", "category": "Kaufen",
     "url": "https://www.chrono24.de/info/faqs.htm"},
    {"id": "faq-0002", "type": "faq", "question": "Wie verkaufe ich eine Uhr?",
     "answer": "Über ein Verkäuferkonto.", "category": "Verkaufen",
     "url": "https://www.chrono24.de/info/faqs.htm"},
    {"id": "info-shipping-0001", "type": "page_chunk", "title": "Versand",
     "heading": "Versicherter Versand", "text": "Uhren werden versichert verschickt.",
     "url": "https://www.chrono24.de/info/shipping.htm"},
]

DOC_VECS = {"Käuferschutz": [1.0, 0.0, 0.0], "verkaufe": [0.0, 1.0, 0.0],
            "Versand": [0.0, 0.0, 1.0]}


def encode_one(text):
    for key, vec in DOC_VECS.items():
        if key in text:
            return vec
    return [-1.0, 0.0, 0.0]


def neutral_reranker(query, texts):
    return [1.0 for _ in texts]


def make_retriever(tmp_path, reranker, rerank_threshold=-6.0, use_llm_reranker=False):
    corpus_path = tmp_path / "corpus.json"
    corpus_path.write_text(json.dumps({"scraped_at": "2026-08-20", "documents": DOCS}),
                           encoding="utf-8")
    index_dir = tmp_path / "index"
    build_index(corpus_path, index_dir, encoder=lambda texts: [encode_one(t) for t in texts])
    # BM25-Absolutwerte skalieren mit der Korpusgroesse: im 3-Dokument-Korpus
    # liegt ein echter Treffer bei ~1.8 (gemessen), deutlich unter der
    # Produktionsschwelle (auf 313 Dokumente gemessen), die hier alles
    # abweisen wuerde.
    # rerank_threshold fest auf -6.0 statt der Produktions-Konstante: Mock-
    # Reranker in diesen Tests geben feste, willkuerliche Scores (1.0, 2.0,
    # -5.0, -9.0) zurueck -- die muessen unabhaengig davon bleiben, welches
    # echte Modell/welche Schwelle gerade in app.retrieval.RERANK_THRESHOLD
    # konfiguriert ist (die Schwelle ist an ein konkretes Modell gekoppelt,
    # siehe README, Reranker-Finetune).
    return Retriever(index_dir, corpus_path, encoder=encode_one, reranker=reranker,
                     bm25_threshold=1.0, rerank_threshold=rerank_threshold,
                     use_llm_reranker=use_llm_reranker)


@pytest.fixture()
def retriever(tmp_path):
    return make_retriever(tmp_path, reranker=neutral_reranker)


def test_rrf_fuse_rewards_docs_in_both_rankings():
    scores = rrf_fuse([["a", "b"], ["b", "c"]], k=60)
    assert scores["b"] > scores["a"]
    assert scores["b"] > scores["c"]
    assert scores["b"] == pytest.approx(1 / 62 + 1 / 61)


def test_retrieve_finds_matching_faq(retriever):
    docs = retriever.retrieve("Wie funktioniert der Käuferschutz?")
    assert docs
    assert isinstance(docs[0], RetrievedDoc)
    assert docs[0].id == "faq-0001"
    assert docs[0].title == "Wie funktioniert der Käuferschutz?"
    assert "sichert deine Zahlung" in docs[0].text


def test_retrieve_returns_empty_for_offtopic(retriever):
    docs = retriever.retrieve("Gedicht über Katzen bitte")
    assert docs == []


def test_reranker_reorders_candidates(tmp_path):
    def prefer_selling(query, texts):
        return [2.0 if "Verkäuferkonto" in t else 1.0 for t in texts]

    retriever = make_retriever(tmp_path, reranker=prefer_selling)
    docs = retriever.retrieve("Wie funktioniert der Käuferschutz?")
    assert docs[0].id == "faq-0002"
    assert docs[0].rerank_score == 2.0
    assert docs[1].rerank_score == 1.0


def test_reranker_false_keeps_rrf_order(tmp_path):
    retriever = make_retriever(tmp_path, reranker=False)
    docs = retriever.retrieve("Wie funktioniert der Käuferschutz?")
    assert docs[0].id == "faq-0001"
    assert docs[0].rerank_score is None


def test_gate_fires_before_reranker(tmp_path):
    def exploding_reranker(query, texts):
        raise AssertionError("Reranker darf bei Off-Topic nicht laufen")

    retriever = make_retriever(tmp_path, reranker=exploding_reranker)
    assert retriever.retrieve("Gedicht über Katzen bitte") == []


def test_variant_hit_resolves_to_canonical_doc(tmp_path):
    """Query matcht nur die generierte Variante, nicht die Original-Frage direkt --
    der Vektor-Teil liefert die Varianten-ID zurueck, der Retriever muss sie auf
    faq-0001 zurueckmappen."""
    docs = [
        {"id": "faq-0001", "type": "faq", "question": "Wie funktioniert der Käuferschutz?",
         "answer": "Er sichert Zahlungen ab.", "category": "Kaufen",
         "url": "https://www.chrono24.de/info/faqs.htm"},
        {"id": "faq-0002", "type": "faq", "question": "Wie verkaufe ich eine Uhr?",
         "answer": "Über ein Verkäuferkonto.", "category": "Verkaufen",
         "url": "https://www.chrono24.de/info/faqs.htm"},
    ]
    corpus_path = tmp_path / "corpus.json"
    corpus_path.write_text(json.dumps({"scraped_at": "2026-08-20", "documents": docs}),
                           encoding="utf-8")
    variants_path = tmp_path / "variants.json"
    variants_path.write_text(json.dumps({"faq-0001": ["Was deckt der Kaeuferschutz ab?"]}),
                             encoding="utf-8")

    # Nur die Variante bekommt den Such-Vektor -- die Original-Frage liegt bewusst
    # weit weg, ein Treffer ist also nur ueber die Variante moeglich.
    vecs = {
        "Wie funktioniert der Käuferschutz?": [0.0, 0.0, 1.0],
        "Wie verkaufe ich eine Uhr?": [0.0, 1.0, 0.0],
        "Was deckt der Kaeuferschutz ab?": [1.0, 0.0, 0.0],
    }

    def encode(text):
        return vecs.get(text, [1.0, 0.0, 0.0])

    index_dir = tmp_path / "index"
    build_index(corpus_path, index_dir, encoder=lambda texts: [encode(t) for t in texts],
                variants_path=variants_path)
    # BM25-Gate aus: in Mini-Korpora sind BM25-Rohwerte nicht aussagekraeftig,
    # hier zaehlt nur der Vektorpfad.
    retriever = Retriever(index_dir, corpus_path, encoder=encode, reranker=False,
                          bm25_threshold=float("-inf"))

    docs_out = retriever.retrieve("Was deckt der Kaeuferschutz ab?", top_k=5)
    assert docs_out[0].id == "faq-0001"
    assert [d.id for d in docs_out].count("faq-0001") == 1


def test_vector_path_yields_n_distinct_canonical_docs_after_dedupe(tmp_path):
    """Pinnt die Ueberfetch-Invariante: der Vektorpfad muss n *verschiedene*
    kanonische Dokumente liefern, auch wenn mehrere Varianten derselben FAQ
    die naechsten Nachbarn der Query sind. Vor dem Fix (n_results=n VOR der
    canonical_id-Dedupe) kollabiert das hier auf 1 Dokument, weil die 6
    Eintraege (1 Original + 5 Varianten) von FAQ 0 die kompletten n=5
    Rohplaetze fuer sich beanspruchen -- FAQ 1..4 kommen gar nicht erst zum
    Zug."""
    num_faqs = 5
    variants_per_faq = 5
    docs = [
        {"id": f"faq-100{c}", "type": "faq", "question": f"FAQ Frage {c}",
         "answer": f"Antwort {c}.", "category": "Kaufen",
         "url": "https://www.chrono24.de/info/faqs.htm"}
        for c in range(num_faqs)
    ]
    variants = {
        f"faq-100{c}": [f"FAQ Variante {c}-{v}" for v in range(1, variants_per_faq + 1)]
        for c in range(num_faqs)
    }

    # Jede FAQ bildet ein eng zusammenliegendes Cluster (Cosine-Similarity
    # 0.995-1.0 fuer FAQ 0, 0.895-0.9 fuer FAQ 1, ...), die Cluster selbst
    # liegen mit 0.1 Abstand klar getrennt -- keine Ties, deterministische
    # Rohreihenfolge.
    vecmap: dict[str, tuple[float, float]] = {}
    for c in range(num_faqs):
        for v in range(variants_per_faq + 1):
            s = 1.0 - c * 0.1 - v * 0.001
            text = f"FAQ Frage {c}" if v == 0 else f"FAQ Variante {c}-{v}"
            vecmap[text] = (s, (1 - s ** 2) ** 0.5)
    query_vec = (1.0, 0.0)

    corpus_path = tmp_path / "corpus.json"
    corpus_path.write_text(json.dumps({"scraped_at": "2026-08-20", "documents": docs}),
                           encoding="utf-8")
    variants_path = tmp_path / "variants.json"
    variants_path.write_text(json.dumps(variants), encoding="utf-8")
    index_dir = tmp_path / "index"
    build_index(corpus_path, index_dir,
                encoder=lambda texts: [list(vecmap[t]) for t in texts],
                variants_path=variants_path)

    retriever = Retriever(index_dir, corpus_path,
                          encoder=lambda text: list(vecmap.get(text, query_vec)),
                          reranker=False, bm25_threshold=float("-inf"))

    # Fragetext kommt in keinem FAQ-Text vor -> BM25 traegt nichts bei, die
    # Rueckgabe-Reihenfolge stammt damit ausschliesslich aus dem Vektorpfad.
    docs_out = retriever.retrieve("Xylophon Quietscheentchen Zauberstab", top_k=num_faqs)
    ids = [d.id for d in docs_out]
    assert len(set(ids)) == num_faqs


def test_gate_abstains_when_only_bm25_is_low(tmp_path):
    """Off-Topic-Fragen in Fragesatzform erreichen beim multilingualen MiniLM hohe
    Cosine-Similarity ("Wie backe ich einen Hefezopf?": 0.742) -- das Sim-Signal
    allein kann sie nicht erkennen. BM25 sieht aber, dass kein Wort im Korpus
    vorkommt. Ein einzelnes eindeutig schwaches Signal muss reichen (ODER-Gate),
    sonst ist das Gate wirkungslos."""
    def hefezopf_looks_like_faq(text):
        # Embedding-Rauschen nachgestellt: Off-Topic-Frage landet auf faq-0001.
        return [1.0, 0.0, 0.0] if "Hefezopf" in text else encode_one(text)

    corpus_path = tmp_path / "corpus.json"
    corpus_path.write_text(json.dumps({"scraped_at": "2026-08-20", "documents": DOCS}),
                           encoding="utf-8")
    index_dir = tmp_path / "index"
    build_index(corpus_path, index_dir, encoder=lambda texts: [encode_one(t) for t in texts])
    retriever = Retriever(index_dir, corpus_path, encoder=hefezopf_looks_like_faq,
                          reranker=neutral_reranker, bm25_threshold=1.0,
                          rerank_threshold=-6.0)

    assert retriever.retrieve("Wie backe ich einen Hefezopf?") == []
    # Gegenprobe: eine echte Frage mit demselben Vektor bleibt durch.
    assert retriever.retrieve("Wie funktioniert der Käuferschutz?")


def test_gate_abstains_when_reranker_rejects_every_candidate(tmp_path):
    """Nah am Domänenvokabular (Sim und BM25 beide hoch) bleibt der Cross-Encoder
    das letzte Signal: liegt selbst sein bester Score unter RERANK_THRESHOLD,
    passt kein Kandidat und der Bot antwortet leer statt zu raten."""
    def rejects_all(query, texts):
        return [-9.0 for _ in texts]

    retriever = make_retriever(tmp_path, reranker=rejects_all)
    assert retriever.retrieve("Wie funktioniert der Käuferschutz?") == []


def test_retrieve_with_audience_excludes_wrong_role(tmp_path):
    """Harter Pre-Filter (Schritt 1, corpus-storage-rethink-design.md): ein
    Verkaeufer-Dokument darf bei einer als 'kaeufer' klassifizierten Anfrage
    gar nicht erst in die Kandidatenmenge -- echter Ausschluss vor der
    RRF-Fusion (jetzt als SQL-WHERE auf der audience-Partition, Schritt 2),
    kein Score-Abzug wie beim verworfenen Rollen-Malus."""
    docs = [
        {"id": "faq-buyer", "type": "faq", "question": "Wie funktioniert der Schutz beim Kauf?",
         "answer": "Der Käuferschutz sichert deine Zahlung ab.", "category": "Kaufen",
         "audience": "kaeufer", "url": "https://www.chrono24.de/info/faqs.htm"},
        {"id": "faq-seller", "type": "faq",
         "question": "Wie funktioniert der Schutz beim Verkauf?",
         "answer": "Der Verkäuferschutz sichert deine Auszahlung ab.", "category": "Verkaufen",
         "audience": "verkaeufer", "url": "https://www.chrono24.de/info/faqs.htm"},
    ]
    corpus_path = tmp_path / "corpus.json"
    corpus_path.write_text(json.dumps({"scraped_at": "2026-08-20", "documents": docs}),
                           encoding="utf-8")
    index_dir = tmp_path / "index"

    def encode(text):
        return [1.0, 0.0, 0.0]  # identisches Embedding -- Vektor-Pfad findet beide gleich gut

    build_index(corpus_path, index_dir, encoder=lambda texts: [encode(t) for t in texts])
    retriever = Retriever(index_dir, corpus_path, encoder=encode, reranker=False,
                          bm25_threshold=float("-inf"))

    # Ohne audience-Filter finden beide Rollen den gleichen Kandidatenpool.
    docs_out = retriever.retrieve("Wie funktioniert der Schutz?")
    assert {d.id for d in docs_out} == {"faq-buyer", "faq-seller"}

    # Mit hartem Filter verschwindet das Verkaeufer-Dokument komplett.
    docs_out = retriever.retrieve("Wie funktioniert der Schutz?", audience="kaeufer")
    ids = [d.id for d in docs_out]
    assert "faq-buyer" in ids
    assert "faq-seller" not in ids


def test_gate_keeps_candidates_when_reranker_is_merely_unsure(tmp_path):
    """Leicht negative Rerank-Scores sind bei echten Treffern normal (on-topic
    Minimum gemessen -5.6) -- nur eindeutig unter der Schwelle wird verworfen."""
    def unsure(query, texts):
        return [-5.0 for _ in texts]

    retriever = make_retriever(tmp_path, reranker=unsure)
    assert retriever.retrieve("Wie funktioniert der Käuferschutz?")


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
    def encode_no_ties(text):
        if "Käuferschutz" in text:
            return [1.0, 0.0, 0.0]
        if "verkaufe" in text:
            return [0.8, 0.6, 0.0]
        if "Versand" in text:
            return [0.6, 0.0, 0.8]
        return [-1.0, 0.0, 0.0]

    corpus_path = tmp_path / "corpus.json"
    corpus_path.write_text(json.dumps({"scraped_at": "2026-08-24", "documents": DOCS}),
                           encoding="utf-8")
    index_dir = tmp_path / "index"
    build_index(corpus_path, index_dir, encoder=lambda texts: [encode_no_ties(t) for t in texts])
    retriever = Retriever(index_dir, corpus_path, encoder=encode_no_ties, reranker=False,
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
