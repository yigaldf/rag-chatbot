import ragchat.common.clients as clients


def test_qdrant_client_is_singleton(tmp_path, monkeypatch):
    monkeypatch.setattr(clients.settings, "data_dir", str(tmp_path))
    clients.reset_clients()
    q1 = clients.get_qdrant()
    q2 = clients.get_qdrant()
    assert q1 is q2
    clients.reset_clients()
