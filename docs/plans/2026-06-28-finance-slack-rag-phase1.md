# Finance Strategy Expert — Slack RAG Bot (Phase 1) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a Slack bot that answers investment-strategy questions strictly from a Qdrant vector knowledge base, replying "I don't know" when the answer isn't in the corpus.

**Architecture:** A deterministic RAG pipeline. `ingest` chunks the markdown corpus, embeds with Voyage, and stores vectors in Qdrant. At query time `retriever` embeds the question, searches Qdrant, and applies a similarity threshold (the grounding gate). `answerer` either returns "I don't know" (no chunks pass) or calls Claude with a strict, context-only prompt. `app.py` wires this to Slack via Socket Mode on `@mention`. Core logic is pure and unit-testable; Qdrant tests use in-memory mode so no Docker/keys are needed for the test suite.

**Tech Stack:** Python, `slack-bolt` (Socket Mode), `qdrant-client`, `voyageai`, `anthropic`, `python-dotenv`, `pytest`.

**Reference:** Design spec at `docs/specs/2026-06-28-finance-slack-rag-phase1-design.md`.

---

## File structure

| File | Responsibility |
|---|---|
| `src/config.py` | Env vars, model IDs, tunables, prompt constants |
| `src/models.py` | `Chunk` and `Answer` dataclasses |
| `src/chunker.py` | Pure: parse frontmatter + split markdown into word-bounded chunks |
| `src/embeddings.py` | Thin Voyage embedding wrapper |
| `src/store.py` | Qdrant client + collection/upsert/search helpers |
| `src/retriever.py` | Embed query → Qdrant search → threshold filter (grounding gate) |
| `src/ingest.py` | Orchestrate corpus → chunks → vectors → Qdrant (CLI) |
| `src/answerer.py` | Grounded prompt build + Claude call, or "I don't know" |
| `app.py` | Slack Socket Mode listener on `app_mention` |
| `tests/` | One test module per `src/` unit |
| `requirements.txt`, `.env.example`, `README.md` | Setup & docs |

**Note on model IDs:** defaults are `CLAUDE_MODEL=claude-sonnet-4-6` and `EMBED_MODEL=voyage-3` (1024-dim). Both are overridable via env. Confirm they are current before the live smoke test (Task 11); the test suite does not depend on them.

---

## Task 1: Project scaffolding & dependencies

**Files:**
- Create: `requirements.txt`, `.env.example`, `src/__init__.py`, `tests/__init__.py`, `pytest.ini`

- [ ] **Step 1: Create `requirements.txt`**

```
anthropic
voyageai
qdrant-client
slack-bolt
python-dotenv
jupyter
pytest
```

- [ ] **Step 2: Create `.env.example`**

```
ANTHROPIC_API_KEY=
VOYAGE_API_KEY=
SLACK_BOT_TOKEN=xoxb-...
SLACK_APP_TOKEN=xapp-...
QDRANT_URL=http://localhost:6333
QDRANT_API_KEY=
# Optional overrides:
# CLAUDE_MODEL=claude-sonnet-4-6
# EMBED_MODEL=voyage-3
```

- [ ] **Step 3: Create empty `src/__init__.py` and `tests/__init__.py`**

Both files are empty.

- [ ] **Step 4: Create `pytest.ini`**

```ini
[pytest]
testpaths = tests
```

- [ ] **Step 5: Create and activate a virtualenv, install deps**

Run:
```bash
python3 -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt
```
Expected: installs complete without error.

- [ ] **Step 6: Commit**

```bash
git add requirements.txt .env.example src/__init__.py tests/__init__.py pytest.ini
git commit -m "chore: project scaffolding and dependencies"
```

---

## Task 2: Config and data models

**Files:**
- Create: `src/config.py`, `src/models.py`
- Test: `tests/test_models.py`

- [ ] **Step 1: Write the failing test**

`tests/test_models.py`:
```python
from src.models import Chunk, Answer


def test_chunk_defaults_score_none():
    c = Chunk(text="hi", title="T", source_url="http://x")
    assert c.score is None
    assert c.text == "hi"


def test_answer_holds_sources_and_found():
    a = Answer(text="ans", sources=["S — http://x"], found=True)
    assert a.found is True
    assert a.sources == ["S — http://x"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_models.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.models'`

- [ ] **Step 3: Create `src/models.py`**

```python
from dataclasses import dataclass


@dataclass
class Chunk:
    text: str
    title: str
    source_url: str
    score: float | None = None


@dataclass
class Answer:
    text: str
    sources: list[str]
    found: bool
```

- [ ] **Step 4: Create `src/config.py`**

```python
import os

from dotenv import load_dotenv

load_dotenv()

ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
VOYAGE_API_KEY = os.getenv("VOYAGE_API_KEY", "")
SLACK_BOT_TOKEN = os.getenv("SLACK_BOT_TOKEN", "")
SLACK_APP_TOKEN = os.getenv("SLACK_APP_TOKEN", "")
QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY") or None

CLAUDE_MODEL = os.getenv("CLAUDE_MODEL", "claude-sonnet-4-6")
EMBED_MODEL = os.getenv("EMBED_MODEL", "voyage-3")
VECTOR_SIZE = 1024          # voyage-3 output dimension
COLLECTION = "finance_kb"
TOP_K = 5
SCORE_THRESHOLD = 0.45      # tune in the notebook (Task 12)
CHUNK_WORDS = 500
CORPUS_DIR = "data/corpus"

IDK_MESSAGE = "I don't know — I couldn't find that in my knowledge base."
SYSTEM_PROMPT = (
    "You are a strategy-investment expert. Answer the question using ONLY the "
    "context provided below. If the context does not contain the answer, reply "
    'exactly "I don\'t know." Do not use any outside knowledge. Cite the source '
    "title for each claim."
)
```

- [ ] **Step 5: Run test to verify it passes**

Run: `pytest tests/test_models.py -v`
Expected: PASS (2 passed)

- [ ] **Step 6: Commit**

```bash
git add src/config.py src/models.py tests/test_models.py
git commit -m "feat: add config constants and Chunk/Answer models"
```

---

## Task 3: Chunker (frontmatter parsing + word-bounded splitting)

**Files:**
- Create: `src/chunker.py`
- Test: `tests/test_chunker.py`

- [ ] **Step 1: Write the failing test**

`tests/test_chunker.py`:
```python
from src.chunker import parse_frontmatter, chunk_text, chunk_document


def test_parse_frontmatter_extracts_meta_and_body():
    raw = '---\ntitle: "My Doc"\nsource_url: http://x.com\n---\n\nHello world body.'
    meta, body = parse_frontmatter(raw)
    assert meta["title"] == "My Doc"
    assert meta["source_url"] == "http://x.com"
    assert body == "Hello world body."


def test_parse_frontmatter_handles_no_frontmatter():
    meta, body = parse_frontmatter("Just text, no frontmatter.")
    assert meta == {}
    assert body == "Just text, no frontmatter."


def test_chunk_text_splits_on_word_size():
    body = " ".join(str(i) for i in range(10))   # 10 words
    chunks = chunk_text(body, size=4)
    assert len(chunks) == 3                       # 4 + 4 + 2
    assert chunks[0] == "0 1 2 3"
    assert chunks[2] == "8 9"


def test_chunk_text_empty_body_returns_empty():
    assert chunk_text("", size=4) == []


def test_chunk_document_attaches_metadata():
    raw = '---\ntitle: "T"\nsource_url: http://u\n---\n\n' + " ".join(["w"] * 7)
    chunks = chunk_document(raw, size=5)
    assert len(chunks) == 2
    assert all(c.title == "T" and c.source_url == "http://u" for c in chunks)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_chunker.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.chunker'`

- [ ] **Step 3: Create `src/chunker.py`**

```python
from src.models import Chunk


def parse_frontmatter(raw: str) -> tuple[dict, str]:
    """Return (frontmatter dict, body) from a markdown doc."""
    if raw.startswith("---"):
        end = raw.index("\n---", 3)
        fm_block = raw[3:end].strip()
        body = raw[end + 4:].strip()
        meta: dict = {}
        for line in fm_block.splitlines():
            if ":" in line:
                key, value = line.split(":", 1)
                meta[key.strip()] = value.strip().strip('"')
        return meta, body
    return {}, raw.strip()


def chunk_text(body: str, size: int) -> list[str]:
    """Split body into chunks of ~size words on word boundaries."""
    words = body.split()
    if not words:
        return []
    return [" ".join(words[i:i + size]) for i in range(0, len(words), size)]


def chunk_document(raw: str, size: int) -> list[Chunk]:
    meta, body = parse_frontmatter(raw)
    title = meta.get("title", "")
    url = meta.get("source_url", "")
    return [Chunk(text=t, title=title, source_url=url) for t in chunk_text(body, size)]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_chunker.py -v`
Expected: PASS (5 passed)

- [ ] **Step 5: Commit**

```bash
git add src/chunker.py tests/test_chunker.py
git commit -m "feat: add markdown chunker with frontmatter parsing"
```

---

## Task 4: Threshold filter (the grounding gate, pure logic)

**Files:**
- Create: `src/retriever.py` (partial — `filter_by_threshold` only)
- Test: `tests/test_threshold.py`

- [ ] **Step 1: Write the failing test**

`tests/test_threshold.py`:
```python
from types import SimpleNamespace

from src.retriever import filter_by_threshold


def _hit(score, text="t"):
    return SimpleNamespace(
        score=score,
        payload={"text": text, "title": "T", "source_url": "http://u"},
    )


def test_keeps_only_hits_at_or_above_threshold():
    hits = [_hit(0.9, "keep"), _hit(0.4, "drop"), _hit(0.5, "edge")]
    chunks = filter_by_threshold(hits, threshold=0.5)
    texts = [c.text for c in chunks]
    assert texts == ["keep", "edge"]


def test_returns_empty_when_all_below_threshold():
    hits = [_hit(0.1), _hit(0.2)]
    assert filter_by_threshold(hits, threshold=0.5) == []


def test_preserves_score_and_metadata():
    chunks = filter_by_threshold([_hit(0.8, "x")], threshold=0.5)
    assert chunks[0].score == 0.8
    assert chunks[0].title == "T"
    assert chunks[0].source_url == "http://u"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_threshold.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.retriever'`

- [ ] **Step 3: Create `src/retriever.py` with `filter_by_threshold` only**

```python
from src.models import Chunk


def filter_by_threshold(hits, threshold: float) -> list[Chunk]:
    """Keep only Qdrant hits scoring >= threshold, as Chunk objects."""
    kept: list[Chunk] = []
    for hit in hits:
        if hit.score >= threshold:
            payload = hit.payload
            kept.append(
                Chunk(
                    text=payload["text"],
                    title=payload["title"],
                    source_url=payload["source_url"],
                    score=hit.score,
                )
            )
    return kept
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_threshold.py -v`
Expected: PASS (3 passed)

- [ ] **Step 5: Commit**

```bash
git add src/retriever.py tests/test_threshold.py
git commit -m "feat: add similarity threshold filter (grounding gate)"
```

---

## Task 5: Qdrant store helpers (tested in-memory)

**Files:**
- Create: `src/store.py`
- Test: `tests/test_store.py`

- [ ] **Step 1: Write the failing test**

`tests/test_store.py` (uses `QdrantClient(":memory:")` — no Docker needed):
```python
from qdrant_client import QdrantClient

from src.models import Chunk
from src import store


def test_recreate_upsert_and_search_roundtrip():
    client = QdrantClient(":memory:")
    chunks = [
        Chunk(text="alpha", title="A", source_url="http://a"),
        Chunk(text="beta", title="B", source_url="http://b"),
    ]
    vectors = [[1.0, 0.0], [0.0, 1.0]]

    store.recreate_collection(client, "kb", vector_size=2)
    store.upsert_chunks(client, "kb", chunks, vectors)

    hits = store.search_points(client, "kb", [1.0, 0.0], top_k=2)
    assert hits[0].payload["text"] == "alpha"     # closest to [1,0]
    assert hits[0].payload["title"] == "A"
    assert hits[0].payload["source_url"] == "http://a"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_store.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.store'`

- [ ] **Step 3: Create `src/store.py`**

```python
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

from src import config
from src.models import Chunk


def get_client() -> QdrantClient:
    return QdrantClient(url=config.QDRANT_URL, api_key=config.QDRANT_API_KEY)


def recreate_collection(client: QdrantClient, collection: str, vector_size: int) -> None:
    client.recreate_collection(
        collection_name=collection,
        vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
    )


def upsert_chunks(client, collection: str, chunks: list[Chunk], vectors) -> None:
    points = [
        PointStruct(
            id=i,
            vector=vec,
            payload={"text": c.text, "title": c.title, "source_url": c.source_url},
        )
        for i, (c, vec) in enumerate(zip(chunks, vectors))
    ]
    client.upsert(collection_name=collection, points=points)


def search_points(client, collection: str, query_vector, top_k: int):
    return client.search(
        collection_name=collection, query_vector=query_vector, limit=top_k
    )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_store.py -v`
Expected: PASS (1 passed)

- [ ] **Step 5: Commit**

```bash
git add src/store.py tests/test_store.py
git commit -m "feat: add Qdrant store helpers (collection/upsert/search)"
```

---

## Task 6: Embeddings wrapper

**Files:**
- Create: `src/embeddings.py`
- Test: `tests/test_embeddings.py`

- [ ] **Step 1: Write the failing test** (mock the Voyage client — no network)

`tests/test_embeddings.py`:
```python
from unittest.mock import MagicMock, patch

from src.embeddings import voyage_embed


def test_voyage_embed_returns_vectors_from_client():
    fake_result = MagicMock()
    fake_result.embeddings = [[0.1, 0.2], [0.3, 0.4]]
    fake_client = MagicMock()
    fake_client.embed.return_value = fake_result

    with patch("src.embeddings.voyageai.Client", return_value=fake_client):
        vectors = voyage_embed(["a", "b"], input_type="document")

    assert vectors == [[0.1, 0.2], [0.3, 0.4]]
    fake_client.embed.assert_called_once()
    # model + input_type are forwarded
    _, kwargs = fake_client.embed.call_args
    assert kwargs["input_type"] == "document"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_embeddings.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.embeddings'`

- [ ] **Step 3: Create `src/embeddings.py`**

```python
import voyageai

from src import config


def voyage_embed(texts: list[str], input_type: str = "document") -> list[list[float]]:
    """Embed texts with Voyage. input_type is 'document' or 'query'."""
    client = voyageai.Client(api_key=config.VOYAGE_API_KEY)
    result = client.embed(texts, model=config.EMBED_MODEL, input_type=input_type)
    return result.embeddings
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_embeddings.py -v`
Expected: PASS (1 passed)

- [ ] **Step 5: Commit**

```bash
git add src/embeddings.py tests/test_embeddings.py
git commit -m "feat: add Voyage embeddings wrapper"
```

---

## Task 7: Retriever `search()` (integration, in-memory + injected embedder)

**Files:**
- Modify: `src/retriever.py` (add `search`)
- Test: `tests/test_retriever_search.py`

- [ ] **Step 1: Write the failing test**

`tests/test_retriever_search.py`:
```python
from qdrant_client import QdrantClient

from src.models import Chunk
from src import store, retriever


def fake_embed(texts, input_type="document"):
    """Deterministic 2-dim embedder: 'ev' -> [1,0], else -> [0,1]."""
    return [[1.0, 0.0] if "ev" in t.lower() else [0.0, 1.0] for t in texts]


def _seed():
    client = QdrantClient(":memory:")
    store.recreate_collection(client, "kb", vector_size=2)
    chunks = [
        Chunk(text="ev/ebit is a valuation metric", title="EV", source_url="http://ev"),
        Chunk(text="unrelated cooking recipe", title="Food", source_url="http://food"),
    ]
    store.upsert_chunks(client, "kb", chunks, fake_embed([c.text for c in chunks]))
    return client


def test_search_returns_relevant_chunk_above_threshold():
    client = _seed()
    results = retriever.search(
        "what is ev/ebit?", client=client, embed_fn=fake_embed,
        collection="kb", top_k=2, threshold=0.5,
    )
    assert results
    assert results[0].title == "EV"


def test_search_returns_empty_when_nothing_passes_threshold():
    client = _seed()
    # Best cosine is 1.0 ([1,0] vs [1,0]); a threshold above 1 forces an empty result.
    results = retriever.search(
        "what is ev/ebit?", client=client, embed_fn=fake_embed,
        collection="kb", top_k=2, threshold=1.01,
    )
    assert results == []
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_retriever_search.py -v`
Expected: FAIL with `AttributeError: module 'src.retriever' has no attribute 'search'`

- [ ] **Step 3: Add `search` to `src/retriever.py`**

Append below `filter_by_threshold`:
```python
from src import config, store
from src.embeddings import voyage_embed


def search(
    question: str,
    client=None,
    embed_fn=voyage_embed,
    collection: str = config.COLLECTION,
    top_k: int = config.TOP_K,
    threshold: float = config.SCORE_THRESHOLD,
) -> list[Chunk]:
    client = client or store.get_client()
    query_vector = embed_fn([question], input_type="query")[0]
    hits = store.search_points(client, collection, query_vector, top_k)
    return filter_by_threshold(hits, threshold)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_retriever_search.py -v`
Expected: PASS (2 passed)

- [ ] **Step 5: Commit**

```bash
git add src/retriever.py tests/test_retriever_search.py
git commit -m "feat: add retriever.search with embed + qdrant + threshold"
```

---

## Task 8: Ingest orchestration (in-memory + injected embedder)

**Files:**
- Create: `src/ingest.py`
- Test: `tests/test_ingest.py`

- [ ] **Step 1: Write the failing test**

`tests/test_ingest.py`:
```python
from qdrant_client import QdrantClient

from src import store, ingest


def fake_embed(texts, input_type="document"):
    return [[1.0, 0.0] for _ in texts]


def test_build_index_loads_corpus_into_qdrant(tmp_path):
    doc = tmp_path / "doc.md"
    doc.write_text(
        '---\ntitle: "T"\nsource_url: http://u\n---\n\n' + " ".join(["word"] * 12),
        encoding="utf-8",
    )
    client = QdrantClient(":memory:")

    n = ingest.build_index(
        corpus_dir=str(tmp_path), client=client, embed_fn=fake_embed,
        collection="kb", vector_size=2, size=5,
    )

    assert n == 3                                  # 12 words / 5 -> 3 chunks
    hits = store.search_points(client, "kb", [1.0, 0.0], top_k=5)
    assert len(hits) == 3
    assert hits[0].payload["title"] == "T"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_ingest.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.ingest'`

- [ ] **Step 3: Create `src/ingest.py`**

```python
import glob
import os

from src import config, store
from src.chunker import chunk_document
from src.embeddings import voyage_embed
from src.models import Chunk


def load_chunks(corpus_dir: str, size: int) -> list[Chunk]:
    chunks: list[Chunk] = []
    for path in sorted(glob.glob(os.path.join(corpus_dir, "*.md"))):
        with open(path, encoding="utf-8") as f:
            chunks.extend(chunk_document(f.read(), size))
    return chunks


def build_index(
    corpus_dir: str = config.CORPUS_DIR,
    client=None,
    embed_fn=voyage_embed,
    collection: str = config.COLLECTION,
    vector_size: int = config.VECTOR_SIZE,
    size: int = config.CHUNK_WORDS,
) -> int:
    client = client or store.get_client()
    chunks = load_chunks(corpus_dir, size)
    vectors = embed_fn([c.text for c in chunks])
    store.recreate_collection(client, collection, vector_size)
    store.upsert_chunks(client, collection, chunks, vectors)
    return len(chunks)


if __name__ == "__main__":
    count = build_index()
    print(f"Indexed {count} chunks into '{config.COLLECTION}'.")
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_ingest.py -v`
Expected: PASS (1 passed)

- [ ] **Step 5: Commit**

```bash
git add src/ingest.py tests/test_ingest.py
git commit -m "feat: add ingest pipeline (corpus -> chunks -> qdrant)"
```

---

## Task 9: Answerer (grounding short-circuit + Claude call)

**Files:**
- Create: `src/answerer.py`
- Test: `tests/test_answerer.py`

- [ ] **Step 1: Write the failing test**

`tests/test_answerer.py`:
```python
from unittest.mock import MagicMock

from src.models import Chunk
from src import config, answerer


def test_no_chunks_returns_idk_without_calling_claude():
    result = answerer.answer("anything", search_fn=lambda q: [], client=MagicMock())
    assert result.found is False
    assert result.text == config.IDK_MESSAGE
    assert result.sources == []


def test_with_chunks_calls_claude_and_returns_answer_with_sources():
    chunks = [Chunk(text="EV/EBIT is...", title="EV Guide", source_url="http://ev")]
    fake_msg = MagicMock()
    fake_msg.content = [MagicMock(text="EV/EBIT measures cheapness.")]
    fake_client = MagicMock()
    fake_client.messages.create.return_value = fake_msg

    result = answerer.answer(
        "what is ev/ebit?", search_fn=lambda q: chunks, client=fake_client
    )

    assert result.found is True
    assert result.text == "EV/EBIT measures cheapness."
    assert result.sources == ["EV Guide — http://ev"]
    fake_client.messages.create.assert_called_once()


def test_build_user_prompt_includes_context_and_question():
    chunks = [Chunk(text="body text", title="T", source_url="http://u")]
    prompt = answerer.build_user_prompt("my question", chunks)
    assert "body text" in prompt
    assert "my question" in prompt
    assert "http://u" in prompt
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_answerer.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.answerer'`

- [ ] **Step 3: Create `src/answerer.py`**

```python
import anthropic

from src import config, retriever
from src.models import Answer, Chunk


def build_user_prompt(question: str, chunks: list[Chunk]) -> str:
    context = "\n\n".join(
        f"[Source: {c.title} | {c.source_url}]\n{c.text}" for c in chunks
    )
    return f"Context:\n{context}\n\nQuestion: {question}"


def answer(question: str, search_fn=retriever.search, client=None) -> Answer:
    chunks = search_fn(question)
    if not chunks:
        return Answer(text=config.IDK_MESSAGE, sources=[], found=False)

    client = client or anthropic.Anthropic(api_key=config.ANTHROPIC_API_KEY)
    message = client.messages.create(
        model=config.CLAUDE_MODEL,
        max_tokens=1024,
        system=config.SYSTEM_PROMPT,
        messages=[{"role": "user", "content": build_user_prompt(question, chunks)}],
    )
    text = message.content[0].text
    sources = list(dict.fromkeys(f"{c.title} — {c.source_url}" for c in chunks))
    return Answer(text=text, sources=sources, found=True)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_answerer.py -v`
Expected: PASS (3 passed)

- [ ] **Step 5: Commit**

```bash
git add src/answerer.py tests/test_answerer.py
git commit -m "feat: add answerer with grounding short-circuit and Claude call"
```

---

## Task 10: Slack app (mention parsing + Socket Mode wiring)

**Files:**
- Create: `app.py`
- Test: `tests/test_app.py`

- [ ] **Step 1: Write the failing test** (unit-test the pure helpers; Slack runtime is manual)

`tests/test_app.py`:
```python
from src.models import Answer
import app


def test_strip_mention_removes_user_token():
    assert app.strip_mention("<@U123> what is ev/ebit?") == "what is ev/ebit?"


def test_format_reply_appends_sources_when_found():
    ans = Answer(text="EV/EBIT is cheapness.", sources=["EV Guide — http://ev"], found=True)
    reply = app.format_reply(ans)
    assert "EV/EBIT is cheapness." in reply
    assert "EV Guide — http://ev" in reply
    assert "Sources" in reply


def test_format_reply_omits_sources_when_not_found():
    ans = Answer(text="I don't know.", sources=[], found=False)
    reply = app.format_reply(ans)
    assert "Sources" not in reply
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_app.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'app'`

- [ ] **Step 3: Create `app.py`**

```python
import re

from slack_bolt import App
from slack_bolt.adapter.socket_mode import SocketModeHandler

from src import config
from src.answerer import answer
from src.models import Answer

app = App(token=config.SLACK_BOT_TOKEN)


def strip_mention(text: str) -> str:
    """Remove the leading <@USERID> mention token."""
    return re.sub(r"<@\w+>", "", text).strip()


def format_reply(result: Answer) -> str:
    reply = result.text
    if result.found and result.sources:
        reply += "\n\n*Sources:*\n" + "\n".join(f"• {s}" for s in result.sources)
    return reply


@app.event("app_mention")
def handle_mention(event, say):
    question = strip_mention(event.get("text", ""))
    say(text=format_reply(answer(question)), thread_ts=event.get("ts"))


if __name__ == "__main__":
    SocketModeHandler(app, config.SLACK_APP_TOKEN).start()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_app.py -v`
Expected: PASS (3 passed)

- [ ] **Step 5: Run the full suite**

Run: `pytest -v`
Expected: all tests pass (Tasks 2–10).

- [ ] **Step 6: Commit**

```bash
git add app.py tests/test_app.py
git commit -m "feat: add Slack Socket Mode app for app_mention"
```

---

## Task 11: README, Slack app setup, and live smoke test

**Files:**
- Create: `README.md`

- [ ] **Step 1: Create `README.md`**

````markdown
# Finance Strategy Expert — Slack RAG Bot (Phase 1)

A Slack bot that answers investment-strategy questions strictly from a curated
vector knowledge base (Qdrant), replying "I don't know" when the corpus lacks the
answer. See `docs/specs/2026-06-28-finance-slack-rag-phase1-design.md`.

## Setup

1. Create and fill `.env` from `.env.example`.
2. `python3 -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt`
3. Start Qdrant locally: `docker run -p 6333:6333 qdrant/qdrant`

## Slack app config

In https://api.slack.com/apps → create app:
- **Socket Mode:** enable → generate an app-level token (`xapp-...`) → `SLACK_APP_TOKEN`.
- **OAuth scopes (Bot):** `app_mentions:read`, `chat:write`. Install → `SLACK_BOT_TOKEN` (`xoxb-...`).
- **Event Subscriptions:** subscribe to bot event `app_mention`.
- Invite the bot to a channel.

## Run

```bash
python -m src.ingest     # build the Qdrant index from data/corpus/
python app.py            # start the Socket Mode listener
```

Then in Slack: `@YourBot what is EV/EBIT?`

## Tests

```bash
pytest -v
```
The suite runs fully offline (Qdrant in-memory, mocked APIs).
````

- [ ] **Step 2: Live smoke test — verify model IDs**

Confirm `CLAUDE_MODEL` and `EMBED_MODEL` in `src/config.py` are current/available for your account; override via `.env` if needed.

- [ ] **Step 3: Live smoke test — ingest**

Run (Qdrant running, `.env` filled):
```bash
python -m src.ingest
```
Expected: prints `Indexed N chunks into 'finance_kb'.` with N > 0.

- [ ] **Step 4: Live smoke test — grounded answer & "I don't know"**

Run:
```bash
python -c "from src.answerer import answer; print(answer('what is EV/EBIT?').text)"
python -c "from src.answerer import answer; r=answer('what is the weather today?'); print(r.found, r.text)"
```
Expected: first prints a cited, on-topic answer; second prints `False` and the "I don't know" message.

- [ ] **Step 5: Live smoke test — Slack end to end**

Run `python app.py`, then in a channel the bot is in: `@YourBot what is the Acquirer's Multiple?`
Expected: a threaded reply with an answer and a Sources list. Ask `@YourBot what's the capital of France?` → expect "I don't know".

- [ ] **Step 6: Commit**

```bash
git add README.md
git commit -m "docs: add README with setup, Slack config, and smoke tests"
```

---

## Notebook (optional, parallel to Task 11)

`notebooks/rag_lab.ipynb` is a tuning aid, not required for the bot to run. Create it
to interactively tune `SCORE_THRESHOLD`, `TOP_K`, and `CHUNK_WORDS`: import
`src.ingest`, `src.retriever`, `src.answerer`, run real questions against a live
Qdrant, inspect returned chunks + scores, and copy the chosen values into
`src/config.py`. Because it imports the same functions the bot uses, tuned settings
apply to the bot with no code changes.

---

## Self-review notes (spec coverage)

- Strict grounding (spec §6): Task 4 (threshold gate) + Task 9 (short-circuit + strict system prompt). ✅
- "I don't know" behavior: Tasks 9 & 11 (unit + live). ✅
- Citations: Task 9 (`sources`) + Task 10 (`format_reply`). ✅
- Qdrant real DB (spec §3): Tasks 5/8 (in-memory for tests), Task 11 (live Docker). ✅
- Voyage embeddings: Task 6, used in Tasks 7/8. ✅
- Claude generation: Task 9. ✅
- Slack Socket Mode @mention (spec §5.4): Task 10. ✅
- Persona / system prompt: Task 2 (`SYSTEM_PROMPT`). ✅
- Out of scope (Phase 2 tool-use): not present in any task. ✅
