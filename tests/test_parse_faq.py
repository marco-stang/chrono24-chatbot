from pathlib import Path

import pytest

from pipeline.parse import MAX_CHUNK_WORDS, CorpusValidationError, parse_faq_page

FIXTURE = Path("tests/fixtures/faq_sample.html").read_text(encoding="utf-8")
URL = "https://www.chrono24.de/info/faqs.htm"


def test_parse_faq_extracts_qa_pairs():
    docs = parse_faq_page(FIXTURE, URL)
    assert len(docs) >= 2
    first = docs[0]
    assert first["type"] == "faq"
    assert first["id"] == "faq-0001"
    assert first["url"].startswith(URL)
    assert first["question"].strip()
    assert first["answer"].strip()
    assert first["category"].strip()


def test_parse_faq_url_carries_question_anchor():
    docs = parse_faq_page(FIXTURE, URL)
    assert docs[0]["url"] == f"{URL}#acc-22"
    assert docs[1]["url"] == f"{URL}#acc-57"


def test_parse_faq_ids_are_unique_and_sequential():
    docs = parse_faq_page(FIXTURE, URL)
    ids = [d["id"] for d in docs]
    assert ids == [f"faq-{i:04d}" for i in range(1, len(ids) + 1)]


def _faq_html(question: str, answer_html: str) -> str:
    return (
        '<html><body><div class="js-faq-chapter"><h2>Kaufen</h2>'
        '<div class="js-accordion-item" id="acc-1">'
        f'<span class="js-accordion-title">{question}</span>'
        f'<div class="js-accordion-body">{answer_html}</div>'
        "</div></div></body></html>"
    )


def test_parse_faq_never_splits_a_long_answer():
    """Ein Chunk = ein Frage-Antwort-Paar, egal wie lang die Antwort ist.

    Diese Grenze ist im FAQ-Pfad die natürliche und darf nie zerschnitten
    werden: die Vektorsuche embeddet gezielt nur die Frage
    (`pipeline/index.py::doc_embed_text`), und ein Antwortfragment ohne seine
    Frage verliert damit seinen Anker. Auch Reranker und LLM-Kontext bekommen
    Frage und Antwort immer als Paar (`app/retrieval.py::_to_doc`,
    `app/llm.py::build_context`).

    Fällt dieser Test, hat jemand `_split_long_text` oder Ähnliches in den
    FAQ-Pfad eingebaut — das ist der Fehler, nicht der Test. Seiten-Chunks
    dürfen und müssen gesplittet werden, FAQ-Antworten nicht.
    """
    # Deutlich über MAX_CHUNK_WORDS, mit Absätzen und Aufzählung — also genau
    # die Form, bei der Splitting verlockend aussieht.
    paragraphs = "".join(
        f"<p>{' '.join(f'Wort{i}-{j}' for j in range(300))}</p>" for i in range(3)
    )
    bullets = "<ul>" + "".join(f"<li>Punkt {i}</li>" for i in range(10)) + "</ul>"
    docs = parse_faq_page(_faq_html("Wie läuft der Kauf ab?", paragraphs + bullets), URL)

    assert len(docs) == 1, "FAQ-Antwort wurde zerschnitten — siehe Docstring"
    assert len(docs[0]["answer"].split()) > MAX_CHUNK_WORDS
    assert docs[0]["question"] == "Wie läuft der Kauf ab?"
    assert "Wort0-0" in docs[0]["answer"]
    assert "Punkt 9" in docs[0]["answer"]


def test_parse_faq_raises_on_zero_items():
    html = (
        '<html><body><div class="js-faq-chapter"><h2>Kategorie</h2>'
        "</div></body></html>"
    )
    with pytest.raises(CorpusValidationError, match="Keine FAQ-Einträge"):
        parse_faq_page(html, URL)
