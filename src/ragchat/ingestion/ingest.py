import hashlib

from qdrant_client.models import PointStruct

from ragchat.common.clients import get_qdrant
from ragchat.common.config import settings
from ragchat.ingestion.chunking import chunk_document
from ragchat.ingestion.store import (
    ensure_collection, load_manifest, save_manifest, point_id, delete_doc_points,
)
from ragchat.retrieval.embeddings import embed  # embed(texts, meter=None)


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
