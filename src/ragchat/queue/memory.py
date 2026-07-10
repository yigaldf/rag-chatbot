from collections import deque

from ragchat.queue.models import AskJob, ReceivedJob


class InMemoryJobQueue:
    """Process-local queue for tests and no-Docker local runs.

    `deleted` is exposed so tests can assert delete-on-success semantics.
    """

    def __init__(self) -> None:
        self._q: deque[ReceivedJob] = deque()
        self._seq = 0
        self.deleted: list[str] = []

    def send(self, job: AskJob) -> None:
        self._seq += 1
        self._q.append(ReceivedJob(job=job, receipt=f"r{self._seq}", receive_count=1))

    def receive(self, wait_seconds: int = 20) -> list[ReceivedJob]:
        return [self._q.popleft()] if self._q else []

    def delete(self, receipt: str) -> None:
        self.deleted.append(receipt)
