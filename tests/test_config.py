from ragchat.common.config import Settings


def test_defaults_and_paths(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    s = Settings(_env_file=None, data_dir=str(tmp_path))
    assert s.embed_model == "text-embedding-3-small"
    assert s.chat_model == "gpt-4o-mini"
    assert s.router_model == "gpt-4o-mini"
    assert s.vector_size == 1536
    assert s.top_k == 5
    assert s.score_threshold == 0.30
    assert s.chunk_words == 250
    assert s.collection == "finance_kb"
    assert str(s.corpus_dir).endswith("corpus")
    assert str(s.qdrant_dir).endswith("qdrant")
    assert str(s.metrics_path).endswith("metrics/metrics.jsonl")
    assert s.openai_api_key == "sk-test"
