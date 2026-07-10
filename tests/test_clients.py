import pytest

import ragchat.common.clients as clients


def test_qdrant_client_is_singleton(tmp_path, monkeypatch):
    monkeypatch.setattr(clients.settings, "data_dir", str(tmp_path))
    clients.reset_clients()
    q1 = clients.get_qdrant()
    q2 = clients.get_qdrant()
    assert q1 is q2
    clients.reset_clients()


def test_close_qdrant_is_idempotent(tmp_path, monkeypatch):
    monkeypatch.setattr(clients.settings, "data_dir", str(tmp_path))
    clients.reset_clients()
    clients.get_qdrant()
    clients._close_qdrant()
    assert clients._qdrant is None
    clients._close_qdrant()  # second call is a no-op, must not raise
    assert clients._qdrant is None


def test_get_qdrant_uses_url_when_set(monkeypatch):
    monkeypatch.setattr(clients.settings, "qdrant_url", "http://qdrant:6333")
    clients.reset_clients()
    captured = {}

    class FakeQC:
        def __init__(self, **kw):
            captured.update(kw)

    monkeypatch.setattr(clients, "QdrantClient", FakeQC)
    clients.get_qdrant()
    clients.reset_clients()
    assert captured == {"url": "http://qdrant:6333"}


def test_get_qdrant_uses_path_when_url_empty(monkeypatch, tmp_path):
    monkeypatch.setattr(clients.settings, "qdrant_url", "")
    monkeypatch.setattr(clients.settings, "data_dir", str(tmp_path))
    clients.reset_clients()
    captured = {}

    class FakeQC:
        def __init__(self, **kw):
            captured.update(kw)

    monkeypatch.setattr(clients, "QdrantClient", FakeQC)
    clients.get_qdrant()
    clients.reset_clients()
    assert "path" in captured and "url" not in captured


def test_wait_for_qdrant_is_noop_in_embedded_mode(monkeypatch):
    monkeypatch.setattr(clients.settings, "qdrant_url", "")
    called = {"n": 0}
    monkeypatch.setattr(clients, "get_qdrant", lambda: called.__setitem__("n", 1))
    clients.wait_for_qdrant()
    assert called["n"] == 0


def test_wait_for_qdrant_retries_until_ready(monkeypatch):
    monkeypatch.setattr(clients.settings, "qdrant_url", "http://qdrant:6333")
    calls = {"n": 0}

    class FakeQC:
        def get_collections(self):
            calls["n"] += 1
            if calls["n"] < 3:
                raise ConnectionError("not up yet")
            return []

    monkeypatch.setattr(clients, "get_qdrant", lambda: FakeQC())
    monkeypatch.setattr(clients, "_close_qdrant", lambda: None)
    clients.wait_for_qdrant(attempts=5, delay=0.0, sleep=lambda _s: None)
    assert calls["n"] == 3


def test_wait_for_qdrant_raises_after_exhausting_attempts(monkeypatch):
    monkeypatch.setattr(clients.settings, "qdrant_url", "http://qdrant:6333")

    class FakeQC:
        def get_collections(self):
            raise ConnectionError("never up")

    monkeypatch.setattr(clients, "get_qdrant", lambda: FakeQC())
    monkeypatch.setattr(clients, "_close_qdrant", lambda: None)
    with pytest.raises(RuntimeError, match="not reachable"):
        clients.wait_for_qdrant(attempts=2, delay=0.0, sleep=lambda _s: None)
