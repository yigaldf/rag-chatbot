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
