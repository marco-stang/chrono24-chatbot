import pytest

from pipeline.parse import CorpusValidationError, validate_corpus


def make_corpus(docs):
    return {"scraped_at": "2026-08-20", "documents": docs}


def make_faq(i):
    return {"id": f"faq-{i:04d}", "type": "faq", "question": f"Frage {i}?",
            "answer": f"Antwort {i}.", "category": "Kaufen",
            "url": "https://www.chrono24.de/info/faqs.htm"}


def test_validate_accepts_valid_corpus():
    validate_corpus(make_corpus([make_faq(i) for i in range(1, 31)]))


def test_validate_rejects_too_few_docs():
    with pytest.raises(CorpusValidationError, match="Dokumente"):
        validate_corpus(make_corpus([make_faq(1)]))


def test_validate_rejects_missing_field():
    docs = [make_faq(i) for i in range(1, 31)]
    docs[0]["answer"] = ""
    with pytest.raises(CorpusValidationError, match="answer"):
        validate_corpus(make_corpus(docs))


def test_validate_rejects_duplicate_ids():
    docs = [make_faq(i) for i in range(1, 31)]
    docs[1]["id"] = docs[0]["id"]
    with pytest.raises(CorpusValidationError, match="doppelte"):
        validate_corpus(make_corpus(docs))


def test_validate_rejects_unknown_type():
    docs = [make_faq(i) for i in range(1, 31)]
    docs[0]["type"] = "blogpost"
    with pytest.raises(CorpusValidationError, match="Typ"):
        validate_corpus(make_corpus(docs))
