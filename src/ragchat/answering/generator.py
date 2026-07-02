from ragchat.common.clients import get_openai
from ragchat.common.config import settings
from ragchat.common.models import Answer
from ragchat.metrics.models import TokenMeter
from ragchat.retrieval.planner import plan_queries
from ragchat.retrieval.retriever import retrieve


def _build_user_prompt(question, chunks):
    context = "\n\n".join(f"[Source: {c.title} | {c.source_url}]\n{c.text}" for c in chunks)
    return f"Context:\n{context}\n\nQuestion: {question}"


def answer(question: str) -> Answer:
    meter = TokenMeter()
    queries = plan_queries(question, meter)
    chunks = retrieve(queries, meter)

    if not chunks:
        metrics = meter.to_metrics(sub_queries=len(queries), chunks_used=0)
        return Answer(text=settings.idk_message, sources=[], found=False, metrics=metrics)

    resp = get_openai().chat.completions.create(
        model=settings.chat_model,
        messages=[
            {"role": "system", "content": settings.system_prompt},
            {"role": "user", "content": _build_user_prompt(question, chunks)},
        ],
    )
    meter.add_chat(resp.usage)
    sources = list(dict.fromkeys(f"{c.title} — {c.source_url}" for c in chunks))
    metrics = meter.to_metrics(sub_queries=len(queries), chunks_used=len(chunks))
    return Answer(text=resp.choices[0].message.content, sources=sources, found=True, metrics=metrics)
