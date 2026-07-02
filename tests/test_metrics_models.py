from ragchat.metrics.models import TokenMeter, QuestionMetrics, AggregateStats


class _Usage:
    def __init__(self, p, c):
        self.prompt_tokens = p
        self.completion_tokens = c


def test_token_meter_accumulates_and_builds_metrics():
    m = TokenMeter()
    m.add_embed(10)
    m.add_chat(_Usage(100, 20))
    m.add_chat(_Usage(5, 3))
    metrics = m.to_metrics(sub_queries=3, chunks_used=7)
    assert metrics.embed_tokens == 10
    assert metrics.chat_tokens == 128  # 100+20+5+3
    assert metrics.total_tokens == 138
    assert metrics.sub_queries == 3 and metrics.chunks_used == 7


def test_aggregate_stats_from_metrics_list():
    ms = [
        QuestionMetrics(sub_queries=3, chunks_used=10, embed_tokens=40, chat_tokens=5000, total_tokens=5040),
        QuestionMetrics(sub_queries=1, chunks_used=0, embed_tokens=10, chat_tokens=160, total_tokens=170),
    ]
    agg = AggregateStats.from_metrics(ms)
    assert agg.num_questions == 2
    assert agg.total_sub_queries == 4 and agg.avg_sub_queries == 2.0
    assert agg.total_tokens == 5210 and agg.avg_tokens == 2605.0


def test_aggregate_stats_empty_is_zero_safe():
    agg = AggregateStats.from_metrics([])
    assert agg.num_questions == 0 and agg.avg_tokens == 0.0
