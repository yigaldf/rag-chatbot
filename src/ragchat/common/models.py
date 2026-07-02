from pydantic import BaseModel

from ragchat.metrics.models import QuestionMetrics


class Chunk(BaseModel):
    text: str
    title: str
    source_url: str
    doc_id: str = ""
    score: float | None = None


class Answer(BaseModel):
    text: str
    sources: list[str]
    found: bool
    metrics: QuestionMetrics
