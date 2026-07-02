from ragchat.common.models import Chunk


def parse_frontmatter(raw: str):
    """Return (meta dict, body) from a markdown doc with --- frontmatter."""
    if raw.startswith("---"):
        end = raw.index("\n---", 3)
        fm_block = raw[3:end].strip()
        body = raw[end + 4:].strip()
        meta = {}
        for line in fm_block.splitlines():
            if ":" in line:
                k, v = line.split(":", 1)
                meta[k.strip()] = v.strip().strip('"')
        return meta, body
    return {}, raw.strip()


def chunk_text(body: str, size: int):
    words = body.split()
    if not words:
        return []
    return [" ".join(words[i:i + size]) for i in range(0, len(words), size)]


def chunk_document(raw: str, size: int, doc_id: str = ""):
    meta, body = parse_frontmatter(raw)
    title = meta.get("title", "")
    url = meta.get("source_url", "")
    return [Chunk(text=t, title=title, source_url=url, doc_id=doc_id)
            for t in chunk_text(body, size)]
