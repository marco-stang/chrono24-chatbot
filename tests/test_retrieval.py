import json

import chromadb
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


def make_retriever(tmp_path, reranker):
    corpus_path = tmp_path / "corpus.json"
    corpus_path.write_text(json.dumps({"scraped_at": "2026-08-20", "documents": DOCS}),
                           encoding="utf-8")
    index_dir = tmp_path / "index"
    build_index(corpus_path, index_dir, encoder=lambda texts: [encode_one(t) for t in texts])
    # BM25-Absolutwerte skalieren mit der Korpusgroesse: im 3-Dokument-Korpus
    # liegt ein echter Treffer bei ~1.9, die Produktionsschwelle (5.5, auf
    # 313 Dokumente gemessen) wuerde hier alles abweisen.
    return Retriever(index_dir, corpus_path, encoder=encode_one, reranker=reranker,
                     bm25_threshold=1.0)


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
    Chroma liefert die Varianten-ID zurueck, der Retriever muss sie auf faq-0001
    zurueckmappen."""
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
    # BM25-Gate aus: in Mini-Korpora wird der IDF negativ, hier zaehlt nur der Vektorpfad.
    retriever = Retriever(index_dir, corpus_path, encoder=encode, reranker=False,
                          bm25_threshold=float("-inf"))

    docs_out = retriever.retrieve("Was deckt der Kaeuferschutz ab?", top_k=5)
    assert docs_out[0].id == "faq-0001"
    assert [d.id for d in docs_out].count("faq-0001") == 1


def test_retrieve_falls_back_to_doc_id_when_metadata_missing(tmp_path):
    """Ältere Indexstände (vor Metadaten-Einführung) tragen keine Metadaten;
    Chroma liefert dann None. Der Retriever muss auf die Dokument-ID selbst
    zurückfallen und darf nicht crashen."""
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

    vecs = {
        "Wie funktioniert der Käuferschutz?": [1.0, 0.0, 0.0],
        "Wie verkaufe ich eine Uhr?": [0.0, 1.0, 0.0],
    }

    def encode(text):
        return vecs.get(text, [1.0, 0.0, 0.0])

    index_dir = tmp_path / "index"

    # Baue Index manuell ohne Metadaten, um alten Zustand zu simulieren
    import pickle

    from rank_bm25 import BM25Okapi

    from app.textproc import tokenize

    client = chromadb.PersistentClient(path=str(index_dir / "chroma"))
    try:
        client.delete_collection("docs")
    except chromadb.errors.NotFoundError:
        pass
    coll = client.create_collection("docs", metadata={"hnsw:space": "cosine"})

    ids = [d["id"] for d in docs]
    texts = [d["question"] if d["type"] == "faq" else f"{d['heading']}\n{d['text']}"
             for d in docs]
    embeddings = [encode(t) for t in texts]
    # Wichtig: add() ohne metadatas, um None zu erzeugen
    coll.add(ids=ids, embeddings=embeddings)

    bm25_texts = [d["question"] + "\n" + d["answer"] if d["type"] == "faq"
                  else d.get("heading", "") + "\n" + d.get("text", "")
                  for d in docs]
    bm25 = BM25Okapi([tokenize(t) for t in bm25_texts])
    with open(index_dir / "bm25.pkl", "wb") as f:
        pickle.dump({"doc_ids": ids, "bm25": bm25}, f)

    retriever = Retriever(index_dir, corpus_path, encoder=encode, reranker=False,
                          bm25_threshold=float("-inf"))

    # Sollte nicht crashen und das richtige Dokument zurückgeben
    docs_out = retriever.retrieve("Wie funktioniert der Käuferschutz?", top_k=5)
    assert docs_out
    assert docs_out[0].id == "faq-0001"


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
                          reranker=neutral_reranker, bm25_threshold=1.0)

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
    RRF-Fusion, kein Score-Abzug wie beim verworfenen Rollen-Malus."""
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
