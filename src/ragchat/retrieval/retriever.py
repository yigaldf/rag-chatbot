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
