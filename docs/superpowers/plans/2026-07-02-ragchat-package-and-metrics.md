# ragchat Package + Metrics + CLI Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Extract the notebook's RAG pipeline into a flow-organized `ragchat` package with per-question + aggregate metrics and a CLI (`ingest`/`ask`/`stats`).

**Architecture:** A `src/ragchat/` package split by flow (common, ingestion, retrieval, answering, metrics) plus a CLI. Pydantic models throughout; config via pydantic-settings. Per-question token counting uses a per-call `TokenMeter` (no module global). Aggregate metrics live in a thread-safe `MetricsRegistry` persisted as JSONL. This is a straight port of already-verified notebook code into modules, so behavior is known.

**Tech Stack:** Python 3.12, uv, OpenAI SDK (2.x), qdrant-client (local/on-disk), pydantic v2 + pydantic-settings, pytest.

---

## File Structure

```
src/ragchat/
  __init__.py              # public API
  common/config.py         # Settings (pydantic-settings)
  common/clients.py        # lazy OpenAI + persistent Qdrant singletons
  common/models.py         # Chunk, Answer
  ingestion/chunking.py    # parse_frontmatter, chunk_text, chunk_document
  ingestion/store.py       # ensure_collection, manifest load/save, point-id, delete-by-doc
  ingestion/ingest.py      # ingest(force=False)
  retrieval/embeddings.py  # embed(texts, meter)
  retrieval/planner.py     # QueryPlan, plan_queries(question, meter)
  retrieval/retriever.py   # retrieve(queries, meter)
  answering/generator.py   # answer(question) -> Answer
  metrics/models.py        # TokenMeter, QuestionMetrics, AggregateStats
  metrics/registry.py      # MetricsRegistry, get_registry()
  cli.py                   # python -m ragchat.cli {ingest|ask|stats}
tests/                     # one test module per source module
pyproject.toml             # add pydantic-settings; package config
```

Tests run with a stubbed embedder + temp Qdrant dir where possible (no network). Only optional smoke tests hit OpenAI.

---

### Task 0: Package scaffolding + dependency

**Files:**
- Create: `src/ragchat/__init__.py` (empty for now), and empty `__init__.py` in `common/`, `ingestion/`, `retrieval/`, `answering/`, `metrics/`
- Create: `tests/__init__.py`
- Modify: `pyproject.toml`

- [ ] **Step 1: Add pydantic-settings + package config**

Run:
```bash
uv add pydantic-settings
```

Then ensure `pyproject.toml` declares the src layout. Add (or confirm) under `[tool.hatch.build.targets.wheel]` / or use setuptools; if the project has no build backend for packages, add to `pyproject.toml`:
```toml
[tool.pytest.ini_options]
pythonpath = ["src"]
testpaths = ["tests"]
```
This lets `import ragchat` resolve from `src/` during tests without installing.

- [ ] **Step 2: Create the package directories with empty __init__.py**

```bash
mkdir -p src/ragchat/common src/ragchat/ingestion src/ragchat/retrieval src/ragchat/answering src/ragchat/metrics tests
touch src/ragchat/__init__.py src/ragchat/common/__init__.py src/ragchat/ingestion/__init__.py \
      src/ragchat/retrieval/__init__.py src/ragchat/answering/__init__.py src/ragchat/metrics/__init__.py \
      tests/__init__.py
```

- [ ] **Step 3: Verify import path works**

Run: `uv run python -c "import sys; sys.path.insert(0,'src'); import ragchat; print('ok')"`
Expected: prints `ok`

- [ ] **Step 4: Commit**

```bash
git add pyproject.toml uv.lock src/ragchat tests
git commit -m "chore: scaffold ragchat package + pydantic-settings"
```

---

### Task 1: common/config.py — Settings

**Files:**
- Create: `src/ragchat/common/config.py`
- Test: `tests/test_config.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_config.py
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_config.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'ragchat.common.config'`

- [ ] **Step 3: Write minimal implementation**

```python
# src/ragchat/common/config.py
from pathlib import Path
from pydantic import Field, computed_field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # Models
    embed_model: str = "text-embedding-3-small"
    chat_model: str = "gpt-4o-mini"
    router_model: str = "gpt-4o-mini"
    vector_size: int = 1536

    # Tunables
    top_k: int = 5
    score_threshold: float = 0.30
    chunk_words: int = 250
    collection: str = "finance_kb"

    # Data root. Defaults to <repo>/data resolved from CWD, or /app/data in the container.
    data_dir: str = Field(default="")

    # Secrets (from env)
    openai_api_key: str = ""
    slack_bot_token: str = ""
    slack_app_token: str = ""

    idk_message: str = "I don't know — I couldn't find that in my knowledge base."
    system_prompt: str = (
        "You are a strategy-investment expert. Answer using ONLY the information in "
        "the context below — never outside knowledge. You may synthesize, compare, "
        "and summarize across the provided sources to answer. For a comparison, "
        "cover each item the context supports; if the context covers some items but "
        "not others, answer for those it does cover and explicitly say which it "
        'lacks. Only if the context is essentially irrelevant to the question, reply '
        'exactly "I don\'t know." Cite the source title for each claim.'
    )

    def _resolve_data_dir(self) -> Path:
        if self.data_dir:
            return Path(self.data_dir)
        # repo layout: prefer ./data, fall back to ../data (e.g. running from notebooks/)
        for cand in (Path("data"), Path("../data")):
            if (cand / "corpus").exists():
                return cand
        return Path("data")

    @computed_field
    @property
    def corpus_dir(self) -> Path:
        return self._resolve_data_dir() / "corpus"

    @computed_field
    @property
    def qdrant_dir(self) -> Path:
        return self._resolve_data_dir() / "qdrant"

    @computed_field
    @property
    def metrics_path(self) -> Path:
        return self._resolve_data_dir() / "metrics" / "metrics.jsonl"


settings = Settings()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_config.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/ragchat/common/config.py tests/test_config.py
git commit -m "feat: ragchat Settings (pydantic-settings)"
```

---

### Task 2: metrics/models.py — TokenMeter, QuestionMetrics, AggregateStats

Built before `common/models.py` because `Answer` depends on `QuestionMetrics` (metrics is a leaf; it imports nothing from `common`).

**Files:**
- Create: `src/ragchat/metrics/models.py`
- Test: `tests/test_metrics_models.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_metrics_models.py
from ragchat.metrics.models import TokenMeter, QuestionMetrics, AggregateStats

class _Usage:
    def __init__(self, p, c): self.prompt_tokens = p; self.completion_tokens = c

def test_token_meter_accumulates_and_builds_metrics():
    m = TokenMeter()
    m.add_embed(10)
    m.add_chat(_Usage(100, 20))
    m.add_chat(_Usage(5, 3))
    metrics = m.to_metrics(sub_queries=3, chunks_used=7)
    assert metrics.embed_tokens == 10
    assert metrics.chat_tokens == 128           # 100+20+5+3
    assert metrics.total_tokens == 138
    assert metrics.sub_queries == 3 and metrics.chunks_used == 7

def test_aggregate_stats_from_metrics_list():
    ms = [
        QuestionMetrics(sub_queries=3, chunks_used=10, embed_tokens=40, chat_tokens=5000, total_tokens=5040),
        QuestionMetrics(sub_queries=1, chunks_used=0,  embed_tokens=10, chat_tokens=160,  total_tokens=170),
    ]
    agg = AggregateStats.from_metrics(ms)
    assert agg.num_questions == 2
    assert agg.total_sub_queries == 4 and agg.avg_sub_queries == 2.0
    assert agg.total_tokens == 5210 and agg.avg_tokens == 2605.0

def test_aggregate_stats_empty_is_zero_safe():
    agg = AggregateStats.from_metrics([])
    assert agg.num_questions == 0 and agg.avg_tokens == 0.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_metrics_models.py -v`
Expected: FAIL (`No module named 'ragchat.metrics.models'`)

- [ ] **Step 3: Write minimal implementation**

```python
# src/ragchat/metrics/models.py
from pydantic import BaseModel


class QuestionMetrics(BaseModel):
    sub_queries: int
    chunks_used: int
    embed_tokens: int
    chat_tokens: int
    total_tokens: int


class TokenMeter(BaseModel):
    """Per-question token accumulator (thread-safe by virtue of being per-call)."""
    embed: int = 0
    chat_in: int = 0
    chat_out: int = 0

    def add_embed(self, total_tokens: int) -> None:
        self.embed += total_tokens

    def add_chat(self, usage) -> None:
        self.chat_in += usage.prompt_tokens
        self.chat_out += usage.completion_tokens

    def to_metrics(self, sub_queries: int, chunks_used: int) -> QuestionMetrics:
        chat = self.chat_in + self.chat_out
        return QuestionMetrics(
            sub_queries=sub_queries, chunks_used=chunks_used,
            embed_tokens=self.embed, chat_tokens=chat, total_tokens=self.embed + chat,
        )


class AggregateStats(BaseModel):
    num_questions: int
    total_sub_queries: int
    avg_sub_queries: float
    total_chunks: int
    avg_chunks: float
    total_embed_tokens: int
    total_chat_tokens: int
    total_tokens: int
    avg_tokens: float

    @classmethod
    def from_metrics(cls, items: list[QuestionMetrics]) -> "AggregateStats":
        n = len(items)
        if n == 0:
            return cls(num_questions=0, total_sub_queries=0, avg_sub_queries=0.0,
                       total_chunks=0, avg_chunks=0.0, total_embed_tokens=0,
                       total_chat_tokens=0, total_tokens=0, avg_tokens=0.0)
        tsq = sum(m.sub_queries for m in items)
        tch = sum(m.chunks_used for m in items)
        tem = sum(m.embed_tokens for m in items)
        tca = sum(m.chat_tokens for m in items)
        tto = sum(m.total_tokens for m in items)
        return cls(
            num_questions=n,
            total_sub_queries=tsq, avg_sub_queries=tsq / n,
            total_chunks=tch, avg_chunks=tch / n,
            total_embed_tokens=tem, total_chat_tokens=tca,
            total_tokens=tto, avg_tokens=tto / n,
        )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_metrics_models.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/ragchat/metrics/models.py tests/test_metrics_models.py
git commit -m "feat: TokenMeter, QuestionMetrics, AggregateStats"
```

---

### Task 3: common/models.py — Chunk, Answer

`Answer` composes `QuestionMetrics` from Task 2, so metrics/models must exist first.

**Files:**
- Create: `src/ragchat/common/models.py`
- Test: `tests/test_models.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_models.py
from ragchat.common.models import Chunk, Answer
from ragchat.metrics.models import QuestionMetrics

def test_chunk_defaults():
    c = Chunk(text="t", title="T", source_url="u")
    assert c.doc_id == "" and c.score is None

def test_answer_holds_metrics():
    m = QuestionMetrics(sub_queries=1, chunks_used=0, embed_tokens=1, chat_tokens=2, total_tokens=3)
    a = Answer(text="hi", sources=[], found=False, metrics=m)
    assert a.found is False and a.metrics.total_tokens == 3
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_models.py -v`
Expected: FAIL (`No module named 'ragchat.common.models'`)

- [ ] **Step 3: Write minimal implementation**

```python
# src/ragchat/common/models.py
from pydantic import BaseModel
from ragchat.metrics.models import QuestionMetrics


class Chunk(BaseModel):
    text: str
    title: str
    source_url: str
    doc_id: str = ""
    score: float | None = None


class Answer(BaseModel):
    text: str
    sources: list[str]
    found: bool
    metrics: QuestionMetrics
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_models.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/ragchat/common/models.py tests/test_models.py
git commit -m "feat: Chunk + Answer models"
```

---

### Task 4: common/clients.py — lazy client singletons

**Files:**
- Create: `src/ragchat/common/clients.py`
- Test: `tests/test_clients.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_clients.py
import ragchat.common.clients as clients

def test_qdrant_client_is_singleton(tmp_path, monkeypatch):
    monkeypatch.setattr(clients.settings, "data_dir", str(tmp_path))
    clients.reset_clients()
    q1 = clients.get_qdrant()
    q2 = clients.get_qdrant()
    assert q1 is q2
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_clients.py -v`
Expected: FAIL (`No module named 'ragchat.common.clients'`)

- [ ] **Step 3: Write minimal implementation**

```python
# src/ragchat/common/clients.py
from openai import OpenAI
from qdrant_client import QdrantClient
from ragchat.common.config import settings

_openai: OpenAI | None = None
_qdrant: QdrantClient | None = None


def get_openai() -> OpenAI:
    global _openai
    if _openai is None:
        _openai = OpenAI(api_key=settings.openai_api_key or None)
    return _openai


def get_qdrant() -> QdrantClient:
    global _qdrant
    if _qdrant is None:
        settings.qdrant_dir.mkdir(parents=True, exist_ok=True)
        _qdrant = QdrantClient(path=str(settings.qdrant_dir))
    return _qdrant


def reset_clients() -> None:
    """Test helper: drop cached clients (releases the Qdrant dir lock)."""
    global _openai, _qdrant
    if _qdrant is not None:
        try:
            _qdrant.close()
        except Exception:
            pass
    _openai = None
    _qdrant = None
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_clients.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/ragchat/common/clients.py tests/test_clients.py
git commit -m "feat: lazy OpenAI + persistent Qdrant client singletons"
```

---

### Task 5: ingestion/chunking.py

**Files:**
- Create: `src/ragchat/ingestion/chunking.py`
- Test: `tests/test_chunking.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_chunking.py
from ragchat.ingestion.chunking import parse_frontmatter, chunk_text, chunk_document

def test_parse_frontmatter():
    raw = '---\ntitle: "T"\nsource_url: "http://x"\n---\nhello world'
    meta, body = parse_frontmatter(raw)
    assert meta["title"] == "T" and meta["source_url"] == "http://x"
    assert body == "hello world"

def test_parse_frontmatter_none():
    meta, body = parse_frontmatter("just text")
    assert meta == {} and body == "just text"

def test_chunk_text_word_bounded():
    assert chunk_text("a b c d e", 2) == ["a b", "c d", "e"]
    assert chunk_text("", 2) == []

def test_chunk_document_tags_doc_and_meta():
    raw = '---\ntitle: "T"\nsource_url: "u"\n---\nalpha beta gamma'
    chunks = chunk_document(raw, size=2, doc_id="d.md")
    assert len(chunks) == 2
    assert chunks[0].title == "T" and chunks[0].source_url == "u" and chunks[0].doc_id == "d.md"
    assert chunks[0].text == "alpha beta"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_chunking.py -v`
Expected: FAIL (`No module named 'ragchat.ingestion.chunking'`)

- [ ] **Step 3: Write minimal implementation**

```python
# src/ragchat/ingestion/chunking.py
from ragchat.common.models import Chunk


def parse_frontmatter(raw: str):
    """Return (meta dict, body) from a markdown doc with --- frontmatter."""
    if raw.startswith("---"):
        end = raw.index("\n---", 3)
        fm_block = raw[3:end].strip()
        body = raw[end + 4:].strip()
        meta = {}
        for line in fm_block.splitlines():
            if ":" in line:
                k, v = line.split(":", 1)
                meta[k.strip()] = v.strip().strip('"')
        return meta, body
    return {}, raw.strip()


def chunk_text(body: str, size: int):
    words = body.split()
    if not words:
        return []
    return [" ".join(words[i:i + size]) for i in range(0, len(words), size)]


def chunk_document(raw: str, size: int, doc_id: str = ""):
    meta, body = parse_frontmatter(raw)
    title = meta.get("title", "")
    url = meta.get("source_url", "")
    return [Chunk(text=t, title=title, source_url=url, doc_id=doc_id)
            for t in chunk_text(body, size)]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_chunking.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/ragchat/ingestion/chunking.py tests/test_chunking.py
git commit -m "feat: chunking (frontmatter + word-bounded)"
```

---

### Task 6: ingestion/store.py — collection, manifest, helpers

**Files:**
- Create: `src/ragchat/ingestion/store.py`
- Test: `tests/test_store.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_store.py
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_store.py -v`
Expected: FAIL (`No module named 'ragchat.ingestion.store'`)

- [ ] **Step 3: Write minimal implementation**

```python
# src/ragchat/ingestion/store.py
import json
import uuid
from qdrant_client.models import (
    Distance, VectorParams, Filter, FieldCondition, MatchValue,
)
from ragchat.common.clients import get_qdrant
from ragchat.common.config import settings

NAMESPACE = uuid.UUID("d3b07384-d9a0-4c9b-8e2a-00000000feed")


def point_id(doc_id: str, idx: int) -> str:
    return str(uuid.uuid5(NAMESPACE, f"{doc_id}:{idx}"))


def load_manifest() -> dict:
    p = settings.qdrant_dir / "manifest.json"   # manifest lives beside the qdrant store
    if p.exists():
        try:
            return json.loads(p.read_text())
        except json.JSONDecodeError:
            return {}
    return {}


def save_manifest(m: dict) -> None:
    settings.qdrant_dir.mkdir(parents=True, exist_ok=True)
    (settings.qdrant_dir / "manifest.json").write_text(json.dumps(m, indent=2))


def ensure_collection() -> None:
    q = get_qdrant()
    if not q.collection_exists(settings.collection):
        q.create_collection(
            collection_name=settings.collection,
            vectors_config=VectorParams(size=settings.vector_size, distance=Distance.COSINE),
        )


def delete_doc_points(doc_id: str) -> None:
    get_qdrant().delete(
        collection_name=settings.collection,
        points_selector=Filter(must=[FieldCondition(key="doc_id", match=MatchValue(value=doc_id))]),
    )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_store.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/ragchat/ingestion/store.py tests/test_store.py
git commit -m "feat: store — collection, manifest, point-id, delete-by-doc"
```

---

### Task 7: ingestion/ingest.py — incremental ingest

**Files:**
- Create: `src/ragchat/ingestion/ingest.py`
- Test: `tests/test_ingest.py`

- [ ] **Step 1: Write the failing test** (stubbed embedder, temp corpus + qdrant)

```python
# tests/test_ingest.py
import ragchat.common.clients as clients
from ragchat.ingestion import ingest as ingest_mod

def _write(corpus, name, title, body):
    (corpus).mkdir(parents=True, exist_ok=True)
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_ingest.py -v`
Expected: FAIL (`No module named 'ragchat.ingestion.ingest'`)

- [ ] **Step 3: Write minimal implementation**

```python
# src/ragchat/ingestion/ingest.py
import hashlib
from qdrant_client.models import PointStruct
from ragchat.common.clients import get_qdrant
from ragchat.common.config import settings
from ragchat.ingestion.chunking import chunk_document
from ragchat.ingestion.store import (
    ensure_collection, load_manifest, save_manifest, point_id, delete_doc_points,
)
from ragchat.retrieval.embeddings import embed  # thin: embed(texts, meter=None)


def ingest(force: bool = False) -> dict:
    q = get_qdrant()
    if force and q.collection_exists(settings.collection):
        q.delete_collection(settings.collection)
    ensure_collection()
    manifest = {} if force else load_manifest()

    embedded_docs, skipped_docs, added = 0, 0, 0
    for path in sorted(settings.corpus_dir.glob("*.md")):
        doc_id = path.name
        raw = path.read_text(encoding="utf-8")
        h = hashlib.sha256(raw.encode("utf-8")).hexdigest()

        if manifest.get(doc_id) == h:
            skipped_docs += 1
            continue
        if doc_id in manifest:
            delete_doc_points(doc_id)

        chunks = chunk_document(raw, settings.chunk_words, doc_id)
        if chunks:
            vectors = embed([c.text for c in chunks])
            points = [
                PointStruct(id=point_id(doc_id, i), vector=vec,
                            payload={"text": c.text, "title": c.title,
                                     "source_url": c.source_url, "doc_id": doc_id})
                for i, (c, vec) in enumerate(zip(chunks, vectors))
            ]
            q.upsert(collection_name=settings.collection, points=points)
            added += len(points)

        manifest[doc_id] = h
        embedded_docs += 1

    save_manifest(manifest)
    total = q.count(collection_name=settings.collection).count
    return {"embedded_docs": embedded_docs, "skipped_docs": skipped_docs,
            "total_points": total, "added_points": added}
```

Note: `embed` here is called without a meter (default `None`) — ingest doesn't track per-question tokens. Task 8 defines `embed(texts, meter=None)`.

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_ingest.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/ragchat/ingestion/ingest.py tests/test_ingest.py
git commit -m "feat: incremental ingest (sha256 skip + per-doc replace + force)"
```

---

### Task 8: retrieval/embeddings.py

**Files:**
- Create: `src/ragchat/retrieval/embeddings.py`
- Test: `tests/test_embeddings.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_embeddings.py
import ragchat.retrieval.embeddings as emb
from ragchat.metrics.models import TokenMeter

class _Resp:
    class _D:
        def __init__(self, v): self.embedding = v
    class _U:
        total_tokens = 7
    def __init__(self): self.data = [self._D([1.0, 2.0])]; self.usage = self._U()

def test_embed_records_tokens(monkeypatch):
    class _Client:
        class embeddings:
            @staticmethod
            def create(model, input): return _Resp()
    monkeypatch.setattr(emb, "get_openai", lambda: _Client())
    meter = TokenMeter()
    vecs = emb.embed(["hello"], meter)
    assert vecs == [[1.0, 2.0]]
    assert meter.embed == 7
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_embeddings.py -v`
Expected: FAIL (`No module named 'ragchat.retrieval.embeddings'`)

- [ ] **Step 3: Write minimal implementation**

```python
# src/ragchat/retrieval/embeddings.py
from ragchat.common.clients import get_openai
from ragchat.common.config import settings


def embed(texts, meter=None):
    """Return embedding vectors for a list of strings; record tokens on meter if given."""
    resp = get_openai().embeddings.create(model=settings.embed_model, input=texts)
    if meter is not None:
        meter.add_embed(resp.usage.total_tokens)
    return [d.embedding for d in resp.data]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_embeddings.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/ragchat/retrieval/embeddings.py tests/test_embeddings.py
git commit -m "feat: embed() with optional TokenMeter"
```

---

### Task 9: retrieval/planner.py

**Files:**
- Create: `src/ragchat/retrieval/planner.py`
- Test: `tests/test_planner.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_planner.py
import ragchat.retrieval.planner as planner
from ragchat.retrieval.planner import QueryPlan
from ragchat.metrics.models import TokenMeter

class _U: prompt_tokens = 50; completion_tokens = 12
class _Msg:
    def __init__(self, plan): self.parsed = plan
class _Choice:
    def __init__(self, plan): self.message = _Msg(plan)
class _Resp:
    def __init__(self, plan): self.choices = [_Choice(plan)]; self.usage = _U()

def test_plan_queries_returns_parsed_and_records_tokens(monkeypatch):
    class _Client:
        class beta:
            class chat:
                class completions:
                    @staticmethod
                    def parse(model, temperature, response_format, messages):
                        return _Resp(QueryPlan(queries=["q1", "q2"]))
    monkeypatch.setattr(planner, "get_openai", lambda: _Client())
    meter = TokenMeter()
    qs = planner.plan_queries("compare a and b", meter)
    assert qs == ["q1", "q2"]
    assert meter.chat_in == 50 and meter.chat_out == 12

def test_plan_queries_falls_back_on_error(monkeypatch):
    class _Client:
        class beta:
            class chat:
                class completions:
                    @staticmethod
                    def parse(**kw): raise RuntimeError("boom")
    monkeypatch.setattr(planner, "get_openai", lambda: _Client())
    qs = planner.plan_queries("what is x?", TokenMeter())
    assert qs == ["what is x?"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_planner.py -v`
Expected: FAIL (`No module named 'ragchat.retrieval.planner'`)

- [ ] **Step 3: Write minimal implementation**

```python
# src/ragchat/retrieval/planner.py
from pydantic import BaseModel
from ragchat.common.clients import get_openai
from ragchat.common.config import settings


class QueryPlan(BaseModel):
    queries: list[str]


_SYSTEM = (
    "You plan retrieval for a document Q&A system.\n"
    "Rules:\n"
    "- Single-topic question: return exactly ONE query (the cleaned question).\n"
    "- Comparison / multi-item question: return exactly ONE focused query per distinct "
    "item, and fold the aspect asked (e.g. pros and cons) INTO each item's query.\n"
    "- Do NOT add generic/overview queries, the original umbrella question, or duplicates.\n"
    "- Never invent items not present in the question. At most {max_q} queries."
)


def plan_queries(question: str, meter=None, max_q: int = 6):
    try:
        resp = get_openai().beta.chat.completions.parse(
            model=settings.router_model, temperature=0,
            response_format=QueryPlan,
            messages=[
                {"role": "system", "content": _SYSTEM.format(max_q=max_q)},
                {"role": "user", "content": question},
            ],
        )
        if meter is not None:
            meter.add_chat(resp.usage)
        qs = [q.strip() for q in resp.choices[0].message.parsed.queries if q and q.strip()]
        return qs[:max_q] or [question]
    except Exception:
        return [question]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_planner.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/ragchat/retrieval/planner.py tests/test_planner.py
git commit -m "feat: plan_queries (structured QueryPlan + fallback)"
```

---

### Task 10: retrieval/retriever.py

**Files:**
- Create: `src/ragchat/retrieval/retriever.py`
- Test: `tests/test_retriever.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_retriever.py
import ragchat.retrieval.retriever as retr
from ragchat.metrics.models import TokenMeter

class _Hit:
    def __init__(self, id, score, doc_id, text):
        self.id = id; self.score = score
        self.payload = {"text": text, "title": doc_id, "source_url": "u", "doc_id": doc_id}
class _Pts:
    def __init__(self, pts): self.points = pts

def test_retrieve_unions_dedups_and_caps_per_doc(monkeypatch):
    monkeypatch.setattr(retr, "embed", lambda queries, meter=None: [[0.0]] * len(queries))
    # doc A returns 4 chunks (should be capped to per_doc=3); doc B returns 1
    hits = [_Hit(f"a{i}", 0.9 - i*0.01, "A", f"a{i}") for i in range(4)] + [_Hit("b0", 0.5, "B", "b0")]
    class _Q:
        def query_points(self, collection_name, query, limit): return _Pts(hits)
    monkeypatch.setattr(retr, "get_qdrant", lambda: _Q())
    monkeypatch.setattr(retr.settings, "score_threshold", 0.3)
    out = retr.retrieve(["q"], TokenMeter(), per_query_k=10, per_doc=3, max_chunks=12)
    doc_ids = [c.doc_id for c in out]
    assert doc_ids.count("A") == 3      # capped
    assert doc_ids.count("B") == 1
    assert out == sorted(out, key=lambda c: c.score, reverse=True)

def test_retrieve_drops_below_threshold(monkeypatch):
    monkeypatch.setattr(retr, "embed", lambda queries, meter=None: [[0.0]])
    hits = [_Hit("x", 0.1, "X", "x")]
    class _Q:
        def query_points(self, **kw): return _Pts(hits)
    monkeypatch.setattr(retr, "get_qdrant", lambda: _Q())
    monkeypatch.setattr(retr.settings, "score_threshold", 0.3)
    assert retr.retrieve(["q"], TokenMeter()) == []
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_retriever.py -v`
Expected: FAIL (`No module named 'ragchat.retrieval.retriever'`)

- [ ] **Step 3: Write minimal implementation**

```python
# src/ragchat/retrieval/retriever.py
from ragchat.common.clients import get_qdrant
from ragchat.common.config import settings
from ragchat.common.models import Chunk
from ragchat.retrieval.embeddings import embed


def retrieve(queries, meter=None, per_query_k=4, per_doc=3, max_chunks=12, threshold=None):
    threshold = settings.score_threshold if threshold is None else threshold
    best, per = {}, {}
    for qvec in embed(queries, meter):
        for h in get_qdrant().query_points(
            collection_name=settings.collection, query=qvec, limit=per_query_k
        ).points:
            if h.score < threshold:
                continue
            if h.id in best:
                best[h.id].score = max(best[h.id].score, h.score)
                continue
            did = h.payload.get("doc_id", "")
            if per.get(did, 0) >= per_doc:
                continue
            per[did] = per.get(did, 0) + 1
            p = h.payload
            best[h.id] = Chunk(text=p["text"], title=p["title"], source_url=p["source_url"],
                               doc_id=did, score=h.score)
    return sorted(best.values(), key=lambda c: c.score, reverse=True)[:max_chunks]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_retriever.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/ragchat/retrieval/retriever.py tests/test_retriever.py
git commit -m "feat: retrieve() — union + dedup + per-doc cap"
```

---

### Task 11: answering/generator.py — answer()

**Files:**
- Create: `src/ragchat/answering/generator.py`
- Test: `tests/test_generator.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_generator.py
import ragchat.answering.generator as gen
from ragchat.common.models import Chunk

class _U: prompt_tokens = 200; completion_tokens = 40
class _Msg:
    def __init__(self, c): self.content = c
class _Choice:
    def __init__(self, c): self.message = _Msg(c)
class _Chat:
    def __init__(self, c): self.choices = [_Choice(c)]; self.usage = _U()

def test_answer_grounded(monkeypatch):
    monkeypatch.setattr(gen, "plan_queries", lambda q, meter: ["q1", "q2", "q3"])
    monkeypatch.setattr(gen, "retrieve", lambda queries, meter: [
        Chunk(text="t", title="EV/EBIT", source_url="u", doc_id="02.md", score=0.5)])
    class _Client:
        class chat:
            class completions:
                @staticmethod
                def create(model, messages): return _Chat("grounded answer")
    monkeypatch.setattr(gen, "get_openai", lambda: _Client())
    a = gen.answer("compare things")
    assert a.found is True and a.text == "grounded answer"
    assert a.sources == ["EV/EBIT — u"]
    assert a.metrics.sub_queries == 3 and a.metrics.chunks_used == 1
    assert a.metrics.chat_tokens == 240   # 200 + 40

def test_answer_idk_when_no_chunks(monkeypatch):
    monkeypatch.setattr(gen, "plan_queries", lambda q, meter: ["q1"])
    monkeypatch.setattr(gen, "retrieve", lambda queries, meter: [])
    called = {"chat": False}
    class _Client:
        class chat:
            class completions:
                @staticmethod
                def create(**kw): called["chat"] = True
    monkeypatch.setattr(gen, "get_openai", lambda: _Client())
    a = gen.answer("weather today?")
    assert a.found is False and a.metrics.chunks_used == 0
    assert called["chat"] is False        # gate short-circuits, LLM not called
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_generator.py -v`
Expected: FAIL (`No module named 'ragchat.answering.generator'`)

- [ ] **Step 3: Write minimal implementation**

```python
# src/ragchat/answering/generator.py
from ragchat.common.clients import get_openai
from ragchat.common.config import settings
from ragchat.common.models import Answer
from ragchat.metrics.models import TokenMeter
from ragchat.retrieval.planner import plan_queries
from ragchat.retrieval.retriever import retrieve


def _build_user_prompt(question, chunks):
    context = "\n\n".join(f"[Source: {c.title} | {c.source_url}]\n{c.text}" for c in chunks)
    return f"Context:\n{context}\n\nQuestion: {question}"


def answer(question: str) -> Answer:
    meter = TokenMeter()
    queries = plan_queries(question, meter)
    chunks = retrieve(queries, meter)

    if not chunks:
        metrics = meter.to_metrics(sub_queries=len(queries), chunks_used=0)
        return Answer(text=settings.idk_message, sources=[], found=False, metrics=metrics)

    resp = get_openai().chat.completions.create(
        model=settings.chat_model,
        messages=[
            {"role": "system", "content": settings.system_prompt},
            {"role": "user", "content": _build_user_prompt(question, chunks)},
        ],
    )
    meter.add_chat(resp.usage)
    sources = list(dict.fromkeys(f"{c.title} — {c.source_url}" for c in chunks))
    metrics = meter.to_metrics(sub_queries=len(queries), chunks_used=len(chunks))
    return Answer(text=resp.choices[0].message.content, sources=sources, found=True, metrics=metrics)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_generator.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/ragchat/answering/generator.py tests/test_generator.py
git commit -m "feat: answer() — planner + retrieve + grounded generation + metrics"
```

---

### Task 12: metrics/registry.py — MetricsRegistry

**Files:**
- Create: `src/ragchat/metrics/registry.py`
- Test: `tests/test_registry.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_registry.py
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_registry.py -v`
Expected: FAIL (`No module named 'ragchat.metrics.registry'`)

- [ ] **Step 3: Write minimal implementation**

```python
# src/ragchat/metrics/registry.py
import json
import threading
from pathlib import Path
from ragchat.metrics.models import QuestionMetrics, AggregateStats
from ragchat.common.config import settings


class MetricsRegistry:
    def __init__(self, path: Path):
        self._path = Path(path)
        self._lock = threading.Lock()
        self._items: list[QuestionMetrics] = self._load()

    def _load(self) -> list[QuestionMetrics]:
        items = []
        if self._path.exists():
            for line in self._path.read_text().splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    items.append(QuestionMetrics.model_validate_json(line))
                except Exception:
                    continue  # skip corrupt lines
        return items

    def record(self, m: QuestionMetrics) -> None:
        with self._lock:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            with self._path.open("a", encoding="utf-8") as f:
                f.write(m.model_dump_json() + "\n")
            self._items.append(m)

    def stats(self) -> AggregateStats:
        with self._lock:
            return AggregateStats.from_metrics(list(self._items))


_registry: MetricsRegistry | None = None


def get_registry() -> MetricsRegistry:
    global _registry
    if _registry is None:
        _registry = MetricsRegistry(settings.metrics_path)
    return _registry
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_registry.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/ragchat/metrics/registry.py tests/test_registry.py
git commit -m "feat: MetricsRegistry (JSONL persistence + aggregate stats)"
```

---

### Task 13: Public API + CLI

**Files:**
- Modify: `src/ragchat/__init__.py`
- Create: `src/ragchat/cli.py`
- Test: `tests/test_cli.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_cli.py
import ragchat.cli as cli
from ragchat.common.models import Answer
from ragchat.metrics.models import QuestionMetrics, AggregateStats

def test_cli_ask_prints_answer_and_records(monkeypatch, capsys):
    m = QuestionMetrics(sub_queries=2, chunks_used=3, embed_tokens=10, chat_tokens=100, total_tokens=110)
    monkeypatch.setattr(cli, "answer", lambda q: Answer(text="A", sources=["S — u"], found=True, metrics=m))
    recorded = {}
    class _Reg:
        def record(self, mm): recorded["m"] = mm
        def stats(self): return AggregateStats.from_metrics([])
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
        def stats(self): return agg
    monkeypatch.setattr(cli, "get_registry", lambda: _Reg())
    cli.main(["stats"])
    out = capsys.readouterr().out
    assert "5" in out and "200.0" in out
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_cli.py -v`
Expected: FAIL (`No module named 'ragchat.cli'`)

- [ ] **Step 3: Write minimal implementation**

```python
# src/ragchat/__init__.py
from ragchat.answering.generator import answer
from ragchat.ingestion.ingest import ingest
from ragchat.metrics.registry import get_registry
from ragchat.common.models import Answer, Chunk
from ragchat.metrics.models import QuestionMetrics, AggregateStats

__all__ = ["answer", "ingest", "get_registry", "Answer", "Chunk",
           "QuestionMetrics", "AggregateStats"]
```

```python
# src/ragchat/cli.py
import argparse
from ragchat.answering.generator import answer
from ragchat.ingestion.ingest import ingest
from ragchat.metrics.registry import get_registry


def _print_answer(r):
    print(r.text)
    if r.found and r.sources:
        print("\nSources:")
        for s in r.sources:
            print("  -", s)
    m = r.metrics
    print(f"\nMetrics: {m.sub_queries} sub-quer{'y' if m.sub_queries == 1 else 'ies'} · "
          f"{m.chunks_used} chunk(s) · {m.total_tokens} tokens "
          f"(embed {m.embed_tokens} + chat {m.chat_tokens})")


def _print_stats(s):
    print("Aggregate metrics")
    print(f"  questions   : {s.num_questions}")
    print(f"  sub-queries : total {s.total_sub_queries}, avg {s.avg_sub_queries:.2f}")
    print(f"  chunks      : total {s.total_chunks}, avg {s.avg_chunks:.2f}")
    print(f"  tokens      : total {s.total_tokens}, avg {s.avg_tokens:.1f} "
          f"(embed {s.total_embed_tokens} + chat {s.total_chat_tokens})")


def main(argv=None):
    p = argparse.ArgumentParser(prog="ragchat")
    sub = p.add_subparsers(dest="cmd", required=True)
    pi = sub.add_parser("ingest"); pi.add_argument("--force", action="store_true")
    pa = sub.add_parser("ask"); pa.add_argument("question")
    sub.add_parser("stats")
    args = p.parse_args(argv)

    if args.cmd == "ingest":
        print(ingest(force=args.force))
    elif args.cmd == "ask":
        r = answer(args.question)
        get_registry().record(r.metrics)
        _print_answer(r)
    elif args.cmd == "stats":
        _print_stats(get_registry().stats())


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_cli.py -v`
Expected: PASS

- [ ] **Step 5: Run the full suite**

Run: `uv run pytest -q`
Expected: all tests PASS

- [ ] **Step 6: Commit**

```bash
git add src/ragchat/__init__.py src/ragchat/cli.py tests/test_cli.py
git commit -m "feat: public API + CLI (ingest/ask/stats)"
```

---

### Task 14: Smoke test against real corpus (optional, needs OPENAI_API_KEY)

**Files:** none (manual verification)

- [ ] **Step 1: Ingest the real corpus**

Run: `uv run python -m ragchat.cli ingest`
Expected: `{'embedded_docs': 7, 'skipped_docs': 0, ...}` on first run (cold); `embedded_docs: 0, skipped_docs: 7` on a second run.

- [ ] **Step 2: Ask an in-corpus question**

Run: `uv run python -m ragchat.cli ask "Compare EV/EBIT and O'Shaughnessy — pros and cons"`
Expected: a cited comparison; Metrics line shows ~3 sub-queries and chunks > 0.

- [ ] **Step 3: Ask an off-topic question**

Run: `uv run python -m ragchat.cli ask "What's the weather in Tel Aviv?"`
Expected: the IDK message; Metrics line shows 0 chunks.

- [ ] **Step 4: View stats**

Run: `uv run python -m ragchat.cli stats`
Expected: `questions: 2` (or however many `ask` runs), with totals/averages populated.

- [ ] **Step 5: Commit any config/doc tweaks discovered during smoke test**

```bash
git add -A && git commit -m "chore: smoke-test fixes for ragchat CLI"
```

---

## Notes for the implementer

- **Import cycle caution:** `common/models.py` imports `metrics/models.py` (for `Answer.metrics`). `metrics/models.py` must NOT import `common/models.py`. Keep the dependency one-directional (metrics is a leaf).
- **Qdrant lock:** local on-disk Qdrant allows one client per directory. `reset_clients()` in tests closes it between temp dirs. Never open two clients on the same path.
- **The notebook stays as-is** for now (still useful for experimentation). A follow-up (out of this plan) can slim it to `from ragchat import answer`.
- **Do not** wire Slack or Docker here — that's Plan 2 (`slack-bot-and-docker`).
