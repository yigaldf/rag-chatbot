from ragchat.slack.formatting import strip_mention, format_answer, format_stats
from ragchat.common.models import Answer
from ragchat.metrics.models import QuestionMetrics, AggregateStats


def _answer(text, sources, found):
    m = QuestionMetrics(sub_queries=2, chunks_used=3, embed_tokens=10, chat_tokens=100, total_tokens=110)
    return Answer(text=text, sources=sources, found=found, metrics=m)


def test_strip_mention_removes_leading_bot_id():
    assert strip_mention("<@U123> what is EV/EBIT?", "U123") == "what is EV/EBIT?"
    assert strip_mention("<@U123>   spaced  ", "U123") == "spaced"
    # mention in the middle is also removed; whitespace is collapsed
    assert strip_mention("hey <@U123> hi", "U123") == "hey hi"


def test_format_answer_found_includes_sources_and_metrics():
    out = format_answer(_answer("The answer.", ["Doc A — u"], True))
    assert "The answer." in out
    assert "Doc A — u" in out
    assert "Sources" in out
    assert "110 tokens" in out


def test_format_answer_idk_has_no_sources_section():
    out = format_answer(_answer("I don't know.", [], False))
    assert "I don't know." in out
    assert "Sources" not in out


def test_format_stats():
    agg = AggregateStats(num_questions=5, total_sub_queries=10, avg_sub_queries=2.0,
                         total_chunks=20, avg_chunks=4.0, total_embed_tokens=100,
                         total_chat_tokens=900, total_tokens=1000, avg_tokens=200.0)
    out = format_stats(agg)
    assert "5" in out and "2.0" in out and "200.0" in out
