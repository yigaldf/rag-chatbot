from ragchat.ingestion.chunking import parse_frontmatter, chunk_text, chunk_document


def test_parse_frontmatter():
    raw = '---\ntitle: "T"\nsource_url: "http://x"\n---\nhello world'
    meta, body = parse_frontmatter(raw)
    assert meta["title"] == "T" and meta["source_url"] == "http://x"
    assert body == "hello world"


def test_parse_frontmatter_none():
    meta, body = parse_frontmatter("just text")
    assert meta == {} and body == "just text"


def test_chunk_text_word_bounded():
    assert chunk_text("a b c d e", 2) == ["a b", "c d", "e"]
    assert chunk_text("", 2) == []


def test_chunk_document_tags_doc_and_meta():
    raw = '---\ntitle: "T"\nsource_url: "u"\n---\nalpha beta gamma'
    chunks = chunk_document(raw, size=2, doc_id="d.md")
    assert len(chunks) == 2
    assert chunks[0].title == "T" and chunks[0].source_url == "u" and chunks[0].doc_id == "d.md"
    assert chunks[0].text == "alpha beta"
