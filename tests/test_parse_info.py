from pathlib import Path

from pipeline.parse import MAX_CHUNK_WORDS, parse_info_page

FIXTURE = Path("tests/fixtures/info_sample.html").read_text(encoding="utf-8")
URL = "https://www.chrono24.de/info/buyer-protection.htm"


def test_parse_info_page_builds_chunks():
    docs = parse_info_page(FIXTURE, URL)
    assert len(docs) >= 2
    for doc in docs:
        assert doc["type"] == "page_chunk"
        assert doc["title"].strip()
        assert doc["heading"].strip()
        assert doc["text"].strip()
        assert doc["url"] == URL


def test_long_sections_are_split():
    docs = parse_info_page(FIXTURE, URL)
    for doc in docs:
        assert len(doc["text"].split()) <= MAX_CHUNK_WORDS


def test_footer_headings_outside_main_are_excluded():
    """Regression test for #main-content-Eingrenzung: die Footer-Navigation
    (eigene h2-Tags außerhalb von #main-content, z. B. "Chrono24 Newsletter")
    darf nicht als Inhalts-Abschnitt landen."""
    docs = parse_info_page(FIXTURE, URL)
    headings = {doc["heading"] for doc in docs}
    texts = " ".join(doc["text"] for doc in docs)
    assert "Chrono24 Newsletter" not in headings
    assert "Jetzt kostenlos anmelden" not in texts


def test_comments_are_not_included_in_chunk_text():
    html = (
        "<html><head><title>Testseite</title></head><body>"
        '<main id="main-content">'
        "<h2>Head</h2><!-- secret --><p>real</p>"
        "</main></body></html>"
    )
    docs = parse_info_page(html, URL)
    assert len(docs) == 1
    assert "secret" not in docs[0]["text"]
    assert docs[0]["text"] == "real"


def test_single_overlong_paragraph_without_newlines_is_hard_split():
    long_paragraph = " ".join(f"wort{i}" for i in range(700))
    html = (
        "<html><head><title>Testseite</title></head><body>"
        '<main id="main-content">'
        f"<h2>Head</h2><p>{long_paragraph}</p>"
        "</main></body></html>"
    )
    docs = parse_info_page(html, URL)
    assert len(docs) == 2
    for doc in docs:
        assert len(doc["text"].split()) <= MAX_CHUNK_WORDS
    total_words = sum(len(doc["text"].split()) for doc in docs)
    assert total_words == 700
