import ragchat.common.clients as clients
from ragchat.ingestion import store


def test_point_id_is_deterministic():
    assert store.point_id("a.md", 0) == store.point_id("a.md", 0)
    assert store.point_id("a.md", 0) != store.point_id("a.md", 1)


def test_manifest_roundtrip(tmp_path, monkeypatch):
    monkeypatch.setattr(clients.settings, "data_dir", str(tmp_path))
    clients.reset_clients()
    assert store.load_manifest() == {}
    store.save_manifest({"a.md": "hash1"})
    assert store.load_manifest() == {"a.md": "hash1"}


def test_ensure_collection_creates_once(tmp_path, monkeypatch):
    monkeypatch.setattr(clients.settings, "data_dir", str(tmp_path))
    clients.reset_clients()
    store.ensure_collection()
    q = clients.get_qdrant()
    assert q.collection_exists(clients.settings.collection)
    clients.reset_clients()
