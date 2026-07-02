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
    p = settings.qdrant_dir / "manifest.json"  # manifest lives beside the qdrant store
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
