import threading
from pathlib import Path

from ragchat.metrics.models import QuestionMetrics, AggregateStats
from ragchat.common.config import settings


class MetricsRegistry:
    """Append-only JSONL metrics, safe for many processes.

    stats() re-reads the file on every call: the bot process serves `stats`
    while N worker processes do the appending, so any in-memory cache would
    go stale the moment a worker records an answer.
    """

    def __init__(self, path: Path):
        self._path = Path(path)
        self._lock = threading.Lock()

    def _read(self) -> list[QuestionMetrics]:
        items: list[QuestionMetrics] = []
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
        # Short O_APPEND writes are atomic on POSIX, so N workers may append
        # concurrently. The lock only guards threads inside one process.
        with self._lock:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            with self._path.open("a", encoding="utf-8") as f:
                f.write(m.model_dump_json() + "\n")

    def stats(self) -> AggregateStats:
        return AggregateStats.from_metrics(self._read())


_registry: MetricsRegistry | None = None


def get_registry() -> MetricsRegistry:
    global _registry
    if _registry is None:
        _registry = MetricsRegistry(settings.metrics_path)
    return _registry
