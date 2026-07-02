import ragchat.cli as cli
from ragchat.common.models import Answer
from ragchat.metrics.models import QuestionMetrics, AggregateStats


def test_cli_ask_prints_answer_and_records(monkeypatch, capsys):
    m = QuestionMetrics(sub_queries=2, chunks_used=3, embed_tokens=10, chat_tokens=100, total_tokens=110)
    monkeypatch.setattr(cli, "answer", lambda q: Answer(text="A", sources=["S — u"], found=True, metrics=m))
    recorded = {}

    class _Reg:
        def record(self, mm):
            recorded["m"] = mm

        def stats(self):
            return AggregateStats.from_metrics([])

    monkeypatch.setattr(cli, "get_registry", lambda: _Reg())
    cli.main(["ask", "what is x?"])
    out = capsys.readouterr().out
    assert "A" in out and "S — u" in out and "110 tokens" in out
    assert recorded["m"] is m


def test_cli_stats_prints_aggregate(monkeypatch, capsys):
    agg = AggregateStats(num_questions=5, total_sub_queries=10, avg_sub_queries=2.0,
                         total_chunks=20, avg_chunks=4.0, total_embed_tokens=100,
                         total_chat_tokens=900, total_tokens=1000, avg_tokens=200.0)

    class _Reg:
        def stats(self):
            return agg

    monkeypatch.setattr(cli, "get_registry", lambda: _Reg())
    cli.main(["stats"])
    out = capsys.readouterr().out
    assert "5" in out and "200.0" in out
