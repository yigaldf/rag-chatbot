import ragchat.common.clients as clients
from ragchat.ingestion import ingest as ingest_mod


def _write(corpus, name, title, body):
    corpus.mkdir(parents=True, exist_ok=True)
    (corpus / name).write_text(f'---\ntitle: "{title}"\nsource_url: "http://x"\n---\n{body}')


def test_ingest_cold_warm_edit_force(tmp_path, monkeypatch):
    monkeypatch.setattr(clients.settings, "data_dir", str(tmp_path))
    clients.reset_clients()
    corpus = tmp_path / "corpus"
    _write(corpus, "a.md", "A", "alpha beta gamma")
    _write(corpus, "b.md", "B", "delta epsilon")

    calls = {"n": 0}

    def fake_embed(texts, meter=None):
        calls["n"] += len(texts)
        return [[0.0] * clients.settings.vector_size for _ in texts]

    monkeypatch.setattr(ingest_mod, "embed", fake_embed)

    r1 = ingest_mod.ingest()
    assert r1["embedded_docs"] == 2 and r1["skipped_docs"] == 0 and r1["total_points"] == 2

    before = calls["n"]
    r2 = ingest_mod.ingest()
    assert r2["embedded_docs"] == 0 and r2["skipped_docs"] == 2
    assert calls["n"] == before  # warm run: no embeds

    _write(corpus, "a.md", "A", "alpha beta gamma delta echo")
    r3 = ingest_mod.ingest()
    assert r3["embedded_docs"] == 1 and r3["skipped_docs"] == 1 and r3["total_points"] == 2

    r4 = ingest_mod.ingest(force=True)
    assert r4["embedded_docs"] == 2 and r4["total_points"] == 2
    clients.reset_clients()
