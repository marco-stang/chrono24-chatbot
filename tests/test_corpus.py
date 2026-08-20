import pytest

from pipeline.parse import MIN_CHUNK_DOCS, MIN_FAQ_DOCS, CorpusValidationError, validate_corpus


def make_corpus(docs):
    return {"scraped_at": "2026-08-20", "documents": docs}


def make_faq(i):
    return {"id": f"faq-{i:04d}", "type": "faq", "question": f"Frage {i}?",
            "answer": f"Antwort {i}.", "category": "Kaufen",
            "url": "https://www.chrono24.de/info/faqs.htm"}


def make_chunk(i):
    return {"id": f"chunk-{i:04d}", "type": "page_chunk", "title": "Käuferschutz",
            "heading": f"Abschnitt {i}", "text": f"Text {i}.",
            "url": "https://www.chrono24.de/info/buyer-protection.htm"}


def make_valid_docs():
    """Kleinste Dokumentmenge, die beide Typ-Minima erfüllt."""
    return ([make_faq(i) for i in range(1, MIN_FAQ_DOCS + 1)]
            + [make_chunk(i) for i in range(1, MIN_CHUNK_DOCS + 1)])


def test_validate_accepts_valid_corpus():
    validate_corpus(make_corpus(make_valid_docs()))


def test_validate_rejects_too_few_faq_docs():
    with pytest.raises(CorpusValidationError, match="FAQ-Dokumente"):
        validate_corpus(make_corpus([make_faq(1)]))


def test_validate_rejects_too_few_chunk_docs():
    docs = [make_faq(i) for i in range(1, MIN_FAQ_DOCS + 1)]
    with pytest.raises(CorpusValidationError, match="Seiten-Chunk-Dokumente"):
        validate_corpus(make_corpus(docs))


def test_validate_rejects_missing_field():
    docs = make_valid_docs()
    docs[0]["answer"] = ""
    with pytest.raises(CorpusValidationError, match="answer"):
        validate_corpus(make_corpus(docs))


def test_validate_rejects_duplicate_ids():
    docs = make_valid_docs()
    docs[1]["id"] = docs[0]["id"]
    with pytest.raises(CorpusValidationError, match="doppelte"):
        validate_corpus(make_corpus(docs))


def test_validate_rejects_unknown_type():
    # Ein Dokument extra, damit das Umbiegen auf "blogpost" nicht selbst das
    # FAQ-Minimum unterschreitet und einen anderen Fehler auslöst.
    docs = ([make_faq(i) for i in range(1, MIN_FAQ_DOCS + 2)]
            + [make_chunk(i) for i in range(1, MIN_CHUNK_DOCS + 1)])
    docs[0]["type"] = "blogpost"
    with pytest.raises(CorpusValidationError, match="Typ"):
        validate_corpus(make_corpus(docs))


def test_build_corpus_warns_on_empty_page(tmp_path, monkeypatch, capsys):
    import pipeline.parse as parse_module
    from pipeline.scrape import START_URL, url_to_filename

    monkeypatch.setattr(parse_module, "parse_faq_page", lambda html, url: [make_faq(1)])
    monkeypatch.setattr(parse_module, "parse_info_page", lambda html, url: [])

    (tmp_path / url_to_filename(START_URL)).write_text("<html></html>", encoding="utf-8")
    empty_name = "info__leere-seite.htm.html"
    (tmp_path / empty_name).write_text("<html></html>", encoding="utf-8")

    corpus = parse_module.build_corpus(tmp_path)

    assert len(corpus["documents"]) == 1
    captured = capsys.readouterr()
    assert "WARNUNG" in captured.out
    assert empty_name in captured.out
