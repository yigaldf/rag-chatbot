import re

from ragchat.common.models import Answer
from ragchat.metrics.models import AggregateStats


def strip_mention(text: str, bot_user_id: str) -> str:
    """Remove the <@BOTID> mention token(s) and normalise whitespace."""
    cleaned = re.sub(rf"<@{re.escape(bot_user_id)}>", " ", text)
    return re.sub(r"\s+", " ", cleaned).strip()


def format_answer(a: Answer) -> str:
    lines = [a.text]
    if a.found and a.sources:
        lines.append("\n*Sources:*")
        lines.extend(f"• {s}" for s in a.sources)
    m = a.metrics
    lines.append(
        f"\n_{m.sub_queries} sub-quer{'y' if m.sub_queries == 1 else 'ies'} · "
        f"{m.chunks_used} chunk(s) · {m.total_tokens} tokens "
        f"(embed {m.embed_tokens} + chat {m.chat_tokens})_"
    )
    return "\n".join(lines)


def format_stats(s: AggregateStats) -> str:
    return (
        "*Aggregate metrics*\n"
        f"• questions: {s.num_questions}\n"
        f"• sub-queries: total {s.total_sub_queries}, avg {s.avg_sub_queries:.2f}\n"
        f"• chunks: total {s.total_chunks}, avg {s.avg_chunks:.2f}\n"
        f"• tokens: total {s.total_tokens}, avg {s.avg_tokens:.1f} "
        f"(embed {s.total_embed_tokens} + chat {s.total_chat_tokens})"
    )
