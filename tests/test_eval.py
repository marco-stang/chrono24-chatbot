from pathlib import Path

from app.retrieval import RetrievedDoc
from eval.run_eval import QUESTIONS_PATH, _questions_path, abstention_rate, hit_rate_at_k


class StubRetriever:
    def __init__(self, mapping):
        self.mapping = mapping

    async def retrieve(self, query, top_k=5, audience=None, client=None):
        docs = [RetrievedDoc(id=i, type="faq", title=f"Titel {i}", url="u", text="x", score=0.1)
                for i in self.mapping.get(query, [])]
        return docs, 0


async def test_hit_rate_counts_hits_in_top_k():
    retriever = StubRetriever({
        "F1": ["faq-0001", "faq-0002"],
        "F2": ["faq-0009"],
    })
    questions = [{"question": "F1", "expected_doc_id": "faq-0002"},
                 {"question": "F2", "expected_doc_id": "faq-0003"}]
    rate, misses = await hit_rate_at_k(retriever, questions, k=5)
    assert rate == 0.5
    assert misses[0]["question"] == "F2"
    assert misses[0]["got"] == ["faq-0009"]


async def test_abstention_rate_counts_questions_with_no_hits():
    retriever = StubRetriever({
        "off1": [],
        "off2": ["faq-0005"],
    })
    questions = [{"question": "off1"}, {"question": "off2"}]
    rate, false_hits = await abstention_rate(retriever, questions)
    assert rate == 0.5
    assert false_hits[0]["question"] == "off2"
    assert false_hits[0]["got_id"] == "faq-0005"
    assert false_hits[0]["got_title"] == "Titel faq-0005"


async def test_abstention_rate_is_one_when_all_questions_abstain():
    retriever = StubRetriever({"off1": [], "off2": []})
    questions = [{"question": "off1"}, {"question": "off2"}]
    rate, false_hits = await abstention_rate(retriever, questions)
    assert rate == 1.0
    assert false_hits == []


def test_questions_path_defaults_to_tuning_set():
    assert _questions_path([]) == QUESTIONS_PATH


def test_questions_path_uses_custom_flag():
    assert _questions_path(["--questions", "eval/questions_holdout.json"]) == \
        Path("eval/questions_holdout.json")


async def test_hit_rate_uses_offline_rewrite_when_present():
    """Der Live-Bot schickt nicht-deutsche Fragen durch rewrite_query, bevor sie
    das Retrieval sehen (BM25 arbeitet auf deutschem Korpus). Die Eval tat das
    nicht und mass damit einen Pfad, den es in Produktion nicht gibt --
    'Where do I pay the customs duties?' galt als Miss, obwohl der Bot das
    Dokument findet. Die Umformulierung liegt offline erzeugt im Fragen-JSON,
    damit der CI-Job weiter ohne API-Kosten laeuft."""
    retriever = StubRetriever({
        "Wo bezahle ich Zollgebühren?": ["faq-0025"],
        "Where do I pay the customs duties?": ["faq-0999"],
    })
    questions = [{
        "question": "Where do I pay the customs duties?",
        "rewritten": "Wo bezahle ich Zollgebühren?",
        "expected_doc_id": "faq-0025",
    }]
    rate, misses = await hit_rate_at_k(retriever, questions, k=5)
    assert rate == 1.0
    assert misses == []


async def test_hit_rate_falls_back_to_question_without_rewrite():
    """Deutsche Fragen tragen kein rewritten-Feld -- der Live-Bot laesst sie
    ebenfalls unangetastet durch."""
    retriever = StubRetriever({"Was kostet der Verkauf?": ["faq-0098"]})
    questions = [{"question": "Was kostet der Verkauf?", "expected_doc_id": "faq-0098"}]
    rate, _ = await hit_rate_at_k(retriever, questions, k=5)
    assert rate == 1.0
