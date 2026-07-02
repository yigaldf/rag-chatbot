from pydantic import BaseModel

from ragchat.common.clients import get_openai
from ragchat.common.config import settings


class QueryPlan(BaseModel):
    queries: list[str]


_SYSTEM = (
    "You plan retrieval for a document Q&A system.\n"
    "Rules:\n"
    "- Single-topic question: return exactly ONE query (the cleaned question).\n"
    "- Comparison / multi-item question: return exactly ONE focused query per distinct "
    "item, and fold the aspect asked (e.g. pros and cons) INTO each item's query.\n"
    "- Do NOT add generic/overview queries, the original umbrella question, or duplicates.\n"
    "- Never invent items not present in the question. At most {max_q} queries."
)


def plan_queries(question: str, meter=None, max_q: int = 6):
    try:
        resp = get_openai().beta.chat.completions.parse(
            model=settings.router_model, temperature=0,
            response_format=QueryPlan,
            messages=[
                {"role": "system", "content": _SYSTEM.format(max_q=max_q)},
                {"role": "user", "content": question},
            ],
        )
        if meter is not None:
            meter.add_chat(resp.usage)
        qs = [q.strip() for q in resp.choices[0].message.parsed.queries if q and q.strip()]
        return qs[:max_q] or [question]
    except Exception:
        return [question]
