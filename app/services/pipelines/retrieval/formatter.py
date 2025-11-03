from typing import List


def format_documents(docs: List) -> str:
    """Format retrieved documents for context display"""
    parts = []
    for i, d in enumerate(docs, start=1):
        meta = d.metadata or {}
        source = meta.get("source", "unknown")
        parts.append(f"[Doc {i} | {source}]\n{d.page_content}")
    return "\n\n".join(parts)
