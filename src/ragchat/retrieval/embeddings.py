from ragchat.common.clients import get_openai
from ragchat.common.config import settings


def embed(texts, meter=None):
    """Return embedding vectors for a list of strings; record tokens on meter if given."""
    resp = get_openai().embeddings.create(model=settings.embed_model, input=texts)
    if meter is not None:
        meter.add_embed(resp.usage.total_tokens)
    return [d.embedding for d in resp.data]
