from ragchat.metrics.registry import MetricsRegistry
from ragchat.metrics.models import QuestionMetrics


def _m(sq, ch, e, c):
    return QuestionMetrics(sub_queries=sq, chunks_used=ch, embed_tokens=e, chat_tokens=c, total_tokens=e + c)


def test_record_persists_and_aggregates(tmp_path):
    path = tmp_path / "metrics.jsonl"
    reg = MetricsRegistry(path)
    reg.record(_m(3, 10, 40, 5000))
    reg.record(_m(1, 0, 10, 160))
    s = reg.stats()
    assert s.num_questions == 2
    assert s.total_sub_queries == 4 and s.avg_sub_queries == 2.0
    assert s.total_tokens == 5210 and s.avg_tokens == 2605.0
    # reload from disk in a fresh registry
    reg2 = MetricsRegistry(path)
    assert reg2.stats().num_questions == 2


def test_corrupt_line_is_skipped(tmp_path):
    path = tmp_path / "metrics.jsonl"
    path.write_text('{"bad json\n' + _m(1, 1, 1, 1).model_dump_json() + "\n")
    reg = MetricsRegistry(path)
    assert reg.stats().num_questions == 1
