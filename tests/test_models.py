from ragchat.common.models import Chunk, Answer
from ragchat.metrics.models import QuestionMetrics


def test_chunk_defaults():
    c = Chunk(text="t", title="T", source_url="u")
    assert c.doc_id == "" and c.score is None


def test_answer_holds_metrics():
    m = QuestionMetrics(sub_queries=1, chunks_used=0, embed_tokens=1, chat_tokens=2, total_tokens=3)
    a = Answer(text="hi", sources=[], found=False, metrics=m)
    assert a.found is False and a.metrics.total_tokens == 3
