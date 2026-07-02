import atexit

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


def _close_qdrant() -> None:
    """Close and drop the Qdrant client, releasing the on-disk dir lock.

    Called explicitly at interpreter exit (see atexit registration below) so the
    client is closed while the interpreter is still healthy — avoids the noisy
    "Exception ignored in __del__ ... sys.meta_path is None" message qdrant-client
    otherwise prints when its destructor runs during shutdown.
    """
    global _qdrant
    if _qdrant is not None:
        try:
            _qdrant.close()
        except Exception:
            pass
        _qdrant = None


def reset_clients() -> None:
    """Test helper: drop cached clients (releases the Qdrant dir lock)."""
    global _openai
    _close_qdrant()
    _openai = None


atexit.register(_close_qdrant)
