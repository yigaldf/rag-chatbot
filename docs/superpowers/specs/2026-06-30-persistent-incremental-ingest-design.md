# Persistent store + skip-unchanged-by-hash ingest — Design

**Date:** 2026-06-30
**Component:** `notebooks/rag_lab.ipynb` (Phase 1 RAG prototype)
**Goal:** Stop re-embedding the whole corpus on every kernel restart and on every `build_index` call. Persist the vector index to disk and only embed documents whose content has changed.

## Problem

The current notebook uses `QdrantClient(":memory:")` (ephemeral) and `build_index()` unconditionally deletes the collection and re-embeds **all** chunks. Consequences:

- Every kernel restart loses the index and re-embeds the full corpus.
- Adding one new doc (e.g. `07-ebit-ev-strategy.md`) re-embeds all docs, not just the new one.
- Cost and latency scale with the whole corpus on every run.

## Decisions (locked with user)

| Decision | Choice |
|---|---|
| Store | Persistent on-disk Qdrant at `data/qdrant/` |
| Hash granularity | **Doc-level** — sha256 of each doc's raw file content |
| Skip logic | Skip docs whose hash matches the manifest |
| Edited docs | **Per-doc replace** — delete that doc's old points by `doc_id`, then add new chunks |
| New docs | Purely additive |
| Deleted docs | Not garbage-collected (orphans remain until a `force` rebuild) — accepted tradeoff |
| Force rebuild | `ingest(..., force=True)` wipes collection + manifest and rebuilds everything |

## Architecture

### Persistent store
- Replace `QdrantClient(":memory:")` with `QdrantClient(path=QDRANT_DIR)`.
- `QDRANT_DIR` resolved the same robust way as `CORPUS_DIR` (works whether the notebook runs from repo root or `notebooks/`): try `data/qdrant`, fall back to `../data/qdrant`. Create the directory if missing.
- Add `data/qdrant/` to `.gitignore`.
- Local on-disk Qdrant persists points across kernel restarts. A restart loads the existing index with no embedding calls.

### Manifest (source of truth for "what's already ingested")
- File: `<QDRANT_DIR>/manifest.json`, a JSON object mapping `doc_id` → `sha256_hex` of the doc's raw content.
- `doc_id` = the markdown filename (e.g. `"07-ebit-ev-strategy.md"`). Stable and unique within the corpus.
- Loaded at ingest start; rewritten after each doc is processed (or once at the end).

### Stable point IDs
- Point IDs change from the running integer `i` to `uuid5(NAMESPACE, f"{doc_id}:{chunk_index}")`.
- Deterministic per (doc, chunk position), so re-upserting a chunk overwrites rather than duplicating.

### Chunk / payload changes
- `Chunk` dataclass gains a `doc_id` field (defaulted, set during ingest loop where the filename is known).
- Point payload gains `doc_id` so edited/removed docs can be deleted by filter. Payload keeps existing `text`, `title`, `source_url`.

## Data flow: `ingest(force=False)`

```
ensure_collection()                 # create only if missing (do NOT delete every run)
manifest = load_manifest()          # {} if force or file absent
if force: delete_collection(); recreate; manifest = {}

for each doc file (sorted):
    raw   = read file
    h     = sha256(raw)
    did   = filename
    if manifest.get(did) == h:
        skip (no embed)             # unchanged
    else:
        if did in manifest:         # changed doc -> per-doc replace
            delete points where payload.doc_id == did
        chunks = chunk_document(raw, CHUNK_WORDS)  # tagged with did
        vectors = embed([c.text for c in chunks])
        upsert points (id = uuid5(did:idx), payload incl. did)
        manifest[did] = h

save_manifest(manifest)
return summary: {embedded_docs, skipped_docs, total_points}
```

- `ensure_collection()` creates the collection only if it does not already exist, so a persisted collection survives across runs.
- New doc → additive add. Changed doc → delete-by-`doc_id` then add. Unchanged doc → skipped entirely (no OpenAI call).

## Error handling / edge cases

- **Embedding model / vector size / CHUNK_WORDS changed:** persisted vectors become inconsistent with new config. Documented requirement: run `ingest(force=True)` after changing `EMBED_MODEL`, `VECTOR_SIZE`, or `CHUNK_WORDS`. (No automatic detection in Phase 1.)
- **Deleted doc:** its points remain (orphans) until a `force=True` rebuild — explicitly accepted.
- **Corrupt/absent manifest:** treat as empty manifest; docs get re-embedded (safe, just slower). A mismatched collection should be cleared via `force=True`.
- **On-disk Qdrant lock:** local mode allows a single client. The notebook holds one `qdrant` client for the session; documented that two kernels can't open the same path simultaneously.

## Testing / verification

The notebook is the test surface. Acceptance checks to run in-notebook:

1. **Cold start:** fresh `data/qdrant/` → `ingest()` embeds all docs; `report_token_usage()` shows embedding tokens > 0.
2. **Warm restart:** restart kernel → run config + `ingest()` → embedded_docs == 0, skipped == all, embed tokens == 0; `ask(...)` still answers correctly.
3. **Add a doc:** add a new `.md` → `ingest()` embeds exactly 1 doc, skips the rest.
4. **Edit a doc:** change one doc's text → `ingest()` replaces only that doc's points; a query over the edited content returns the new text, and `count` of points for that `doc_id` is not inflated (no duplicates).
5. **Force:** `ingest(force=True)` re-embeds everything.

## Out of scope (YAGNI for Phase 1)

- Chunk-level hashing.
- Garbage collection of fully-deleted docs.
- Automatic config-change detection / re-embed.
- Concurrent multi-process access.
