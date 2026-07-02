from pydantic import BaseModel


class QuestionMetrics(BaseModel):
    sub_queries: int
    chunks_used: int
    embed_tokens: int
    chat_tokens: int
    total_tokens: int


class TokenMeter(BaseModel):
    """Per-question token accumulator (thread-safe by virtue of being per-call)."""
    embed: int = 0
    chat_in: int = 0
    chat_out: int = 0

    def add_embed(self, total_tokens: int) -> None:
        self.embed += total_tokens

    def add_chat(self, usage) -> None:
        self.chat_in += usage.prompt_tokens
        self.chat_out += usage.completion_tokens

    def to_metrics(self, sub_queries: int, chunks_used: int) -> QuestionMetrics:
        chat = self.chat_in + self.chat_out
        return QuestionMetrics(
            sub_queries=sub_queries, chunks_used=chunks_used,
            embed_tokens=self.embed, chat_tokens=chat, total_tokens=self.embed + chat,
        )


class AggregateStats(BaseModel):
    num_questions: int
    total_sub_queries: int
    avg_sub_queries: float
    total_chunks: int
    avg_chunks: float
    total_embed_tokens: int
    total_chat_tokens: int
    total_tokens: int
    avg_tokens: float

    @classmethod
    def from_metrics(cls, items: list[QuestionMetrics]) -> "AggregateStats":
        n = len(items)
        if n == 0:
            return cls(num_questions=0, total_sub_queries=0, avg_sub_queries=0.0,
                       total_chunks=0, avg_chunks=0.0, total_embed_tokens=0,
                       total_chat_tokens=0, total_tokens=0, avg_tokens=0.0)
        tsq = sum(m.sub_queries for m in items)
        tch = sum(m.chunks_used for m in items)
        tem = sum(m.embed_tokens for m in items)
        tca = sum(m.chat_tokens for m in items)
        tto = sum(m.total_tokens for m in items)
        return cls(
            num_questions=n,
            total_sub_queries=tsq, avg_sub_queries=tsq / n,
            total_chunks=tch, avg_chunks=tch / n,
            total_embed_tokens=tem, total_chat_tokens=tca,
            total_tokens=tto, avg_tokens=tto / n,
        )
