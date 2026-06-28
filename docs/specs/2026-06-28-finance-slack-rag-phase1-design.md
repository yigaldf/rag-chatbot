# Finance Strategy Expert — Slack RAG Bot (Phase 1 Design)

**Date:** 2026-06-28
**Status:** Approved design — ready for implementation planning

## 1. Summary

A Slack bot that answers investment-strategy questions **strictly from a curated
vector knowledge base**. Users mention `@finance <question>` in a channel; the bot
retrieves relevant chunks and either answers (with citations) using *only* that
retrieved context, or replies **"I don't know"** when the knowledge base lacks the
answer. The bot's persona is a **strategy-investment expert**.

This document covers **Phase 1** (deterministic RAG pipeline). Phase 2 (agentic
tool-use) is described as a deferred future upgrade in §10.

## 2. Goals & non-goals

### Goals
- Answer finance-strategy questions grounded **only** in the local corpus.
- When the corpus does not contain the answer, reply exactly **"I don't know"** —
  never fabricate or use outside knowledge.
- Cite the source (title / URL) of the material used in each answer.
- Run from a laptop with no public URL / hosting (Socket Mode).
- Keep components small, single-purpose, and independently testable.

### Non-goals (Phase 1)
- Agentic `search_knowledge_base` Claude tool (→ Phase 2).
- Conversation memory / multi-turn follow-ups.
- Automatic re-ingestion on corpus changes.
- Public hosting / HTTP Events API.
- Web UI.

## 3. Key decisions

| Area | Decision | Rationale |
|---|---|---|
| Language | Python | Richest RAG ecosystem |
| Slack transport | `slack_bolt` + **Socket Mode** | No public URL/hosting; easiest local dev |
| Slack trigger | `app_mention` (`@finance ...`) | Matches desired UX |
| Vector DB | **Qdrant** (real DB) | Runs free in local Docker; same code → Qdrant Cloud later |
| Embeddings | **Voyage AI** (`voyage-3` family) | Anthropic's recommended embedding partner; high quality |
| Generation | **Claude** (Sonnet tier default) | Best quality/cost balance; exact model ID verified at build time |
| RAG style | Deterministic pipeline (no tools) | Strict grounding enforced in code, not just prompt |

## 4. Architecture

```
                          Slack workspace
                                |  @finance what is EV/EBIT?
                                v
                          app.py  (Socket Mode listener)
                                |  question text
                                v
                    answerer.answer(question)
                                |
                                v
                    retriever.search(question)
                       | 1. Voyage embed query
                       | 2. Qdrant top-k search
                       | 3. threshold filter
                       v
                +---------------+----------------+
        no chunks pass                    chunks pass
                |                                |
                v                                v
        "I don't know"             Claude (grounded prompt + context)
                |                                |
                |                         answer + citations
                +---------------+----------------+
                                v
                      reply in Slack thread

   (offline, run once)  ingest.py:  data/corpus/*.md -> chunk -> Voyage embed -> Qdrant
```

## 5. Components

Each is small, single-purpose, and independently testable.

### 5.1 `src/ingest.py` — build the knowledge base (run once / on corpus change)
- **Does:** reads `data/corpus/*.md`, splits each doc into ~500-word chunks
  (preserving `title` + `source_url` frontmatter as metadata), embeds each chunk
  with Voyage, upserts vectors into a Qdrant collection.
- **Interface:** `python -m src.ingest` (CLI). Internals: `chunk(doc) -> list[Chunk]`,
  `embed(texts) -> vectors`, `upsert(chunks)`.
- **Depends on:** Voyage API, Qdrant.
- **Idempotent:** re-running rebuilds the collection cleanly.

### 5.2 `src/retriever.py` — find relevant context (the grounding gate)
- **Does:** embeds the question (Voyage), queries Qdrant for top-k, drops anything
  below a similarity threshold.
- **Interface:** `search(question: str) -> list[Chunk]` (empty list = nothing relevant).
- **Depends on:** Voyage API, Qdrant.
- **Key knob:** `SCORE_THRESHOLD` (tuned in the notebook).

### 5.3 `src/answerer.py` — produce the grounded answer
- **Does:** if `search()` returns nothing → returns the "I don't know" message.
  Otherwise builds a prompt with the retrieved chunks and calls Claude with a strict
  system prompt; returns answer text + source list.
- **Interface:** `answer(question: str) -> Answer{ text, sources, found: bool }`.
- **Depends on:** retriever, Claude API.

### 5.4 `app.py` — Slack integration
- **Does:** `slack_bolt` app in Socket Mode; subscribes to `app_mention`; strips the
  mention text, calls `answerer.answer()`, posts the reply in-thread with citations.
- **Interface:** `python app.py` (long-running process).
- **Depends on:** Slack tokens, answerer.

### 5.5 `notebooks/rag_lab.ipynb` — tuning lab
- **Does:** imports `src/` functions to interactively tune chunk size, top-k, the
  score threshold, and the grounded prompt against real questions before freezing
  settings. Imports the *same* functions the bot uses (no code duplication/drift).

## 6. Strict-grounding mechanism

Two independent guards ensure the bot never makes things up:

1. **Code-level retrieval gate (deterministic):** `retriever.search()` returns only
   chunks above `SCORE_THRESHOLD`. Empty → `answerer` short-circuits to
   *"I don't know — I couldn't find that in my knowledge base."* Claude is never called.
2. **Prompt-level constraint (when chunks pass):** system prompt =
   > "You are a strategy-investment expert. Answer the question using ONLY the context
   > below. If the context does not contain the answer, reply exactly 'I don't know.'
   > Do not use any outside knowledge. Cite the source title for each claim."

Guard 1 catches "no relevant docs"; Guard 2 catches "docs retrieved but they don't
actually answer the question."

## 7. Project structure

```
rag-chatbot/
├── data/corpus/            # 6 finance docs (already gathered)
├── src/
│   ├── ingest.py
│   ├── retriever.py
│   ├── answerer.py
│   └── config.py           # env vars, model IDs, threshold, k
├── notebooks/
│   └── rag_lab.ipynb
├── app.py
├── requirements.txt
├── .env.example            # documents required secrets (no real values)
└── README.md               # setup + run instructions
```

## 8. Configuration, dependencies, infrastructure

- **Dependencies (`requirements.txt`):** `anthropic`, `voyageai`, `qdrant-client`,
  `slack-bolt`, `python-dotenv`, `jupyter`.
- **Secrets (`.env`):** `ANTHROPIC_API_KEY`, `VOYAGE_API_KEY`, `SLACK_BOT_TOKEN`,
  `SLACK_APP_TOKEN`, `QDRANT_URL` (local `http://localhost:6333` or Qdrant Cloud +
  `QDRANT_API_KEY`).
- **Tunables (`config.py`):** `CLAUDE_MODEL`, `EMBED_MODEL`, `TOP_K`,
  `SCORE_THRESHOLD`, `CHUNK_SIZE`.
- **Infrastructure:** Qdrant via Docker locally —
  `docker run -p 6333:6333 qdrant/qdrant`.

## 9. Testing approach

- **Unit:** `chunk()` splits correctly and preserves metadata; `search()` returns
  `[]` below threshold.
- **Grounding tests (most important):**
  - In-corpus question ("what is EV/EBIT?") → cited answer.
  - Out-of-corpus question ("what's the weather?") → **"I don't know."**
- **Manual:** tune in the notebook, then end-to-end in a real Slack channel.

## 10. Phase 2 (deferred — documented, not built now)

Expose retrieval as a Claude **tool** (`search_knowledge_base`) so Claude decides when
to search, can reformulate and search again, and can reason across multiple searches
("agentic RAG"). Phase 1's `retriever.search()` becomes the tool's implementation —
an additive upgrade, nothing wasted. Trade-off: grounding shifts from a hard code gate
toward prompt enforcement, so it is deferred until the Phase 1 baseline is solid.

## 11. Build order (Phase 1)

1. `src/ingest.py` → load corpus into Qdrant.
2. `notebooks/rag_lab.ipynb` → tune chunking, top-k, threshold, grounded prompt.
3. Freeze settings into `src/retriever.py` / `src/answerer.py`.
4. `app.py` → wire the proven pipeline to Slack.
