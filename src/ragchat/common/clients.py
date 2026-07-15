import atexit
import time

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
        if settings.qdrant_url:
            _qdrant = QdrantClient(url=settings.qdrant_url)
        else:
            settings.qdrant_dir.mkdir(parents=True, exist_ok=True)
            _qdrant = QdrantClient(path=str(settings.qdrant_dir))
    return _qdrant


def wait_for_qdrant(attempts: int = 10, delay: float = 1.0, sleep=time.sleep) -> None:
    """Block until the Qdrant server answers. No-op in embedded mode.

    Compose starts `qdrant` and `ingest` together, so the first connection may
    land before the server is listening. Retrying here keeps the readiness
    check in our code instead of betting on `curl` existing in someone else's image.
    """
    if not settings.qdrant_url:
        return
    last: Exception | None = None
    for _ in range(attempts):
        try:
            get_qdrant().get_collections()
            return
        except Exception as e:  # noqa: BLE001 - any transport error means "not up yet"
            last = e
            _close_qdrant()  # drop the half-open client before retrying
            sleep(delay)
    raise RuntimeError(f"Qdrant not reachable at {settings.qdrant_url} after {attempts} attempts: {last}")


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
