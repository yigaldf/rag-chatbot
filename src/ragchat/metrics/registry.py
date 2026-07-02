import threading
from pathlib import Path

from ragchat.metrics.models import QuestionMetrics, AggregateStats
from ragchat.common.config import settings


class MetricsRegistry:
    def __init__(self, path: Path):
        self._path = Path(path)
        self._lock = threading.Lock()
        self._items: list[QuestionMetrics] = self._load()

    def _load(self) -> list[QuestionMetrics]:
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
        with self._lock:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            with self._path.open("a", encoding="utf-8") as f:
                f.write(m.model_dump_json() + "\n")
            self._items.append(m)

    def stats(self) -> AggregateStats:
        with self._lock:
            return AggregateStats.from_metrics(list(self._items))


_registry: MetricsRegistry | None = None


def get_registry() -> MetricsRegistry:
    global _registry
    if _registry is None:
        _registry = MetricsRegistry(settings.metrics_path)
    return _registry
