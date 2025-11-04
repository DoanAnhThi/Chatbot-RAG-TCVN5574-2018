from __future__ import annotations

import hashlib
import json
import os
import re
from pathlib import Path
from typing import Iterable, List, Optional, Set

from bs4 import BeautifulSoup
from langchain_core.documents import Document

try:
    from langdetect import detect as _detect_language
    from langdetect import LangDetectException
except ImportError:  # pragma: no cover - optional dependency guard
    _detect_language = None

    class LangDetectException(Exception):
        """Fallback exception when langdetect is not installed."""


_NON_WORD_RE = re.compile(r"[^0-9a-zA-Z]+")


def clean_text(text: str) -> str:
    """Normalize raw text by stripping HTML and collapsing whitespace."""
    if not text:
        return ""

    soup = BeautifulSoup(text, "html.parser")
    text = soup.get_text(" ")  # remove HTML tags while preserving separators
    text = re.sub(r"\s+", " ", text)
    text = text.strip()
    return text


def preprocess_documents(
    docs: List[Document],
    target_language: Optional[str] = None,
    deduplicate: bool = True,
) -> List[Document]:
    """Apply cleaning, metadata normalization, language filtering, and deduplication."""

    processed: List[Document] = []
    seen_hashes: Set[str] = set()

    for idx, doc in enumerate(docs, start=1):
        content = clean_text(doc.page_content)
        if not content:
            continue

        language = _safe_detect_language(content)
        if target_language and language:
            if language.lower() != target_language.lower():
                continue

        content_hash = hashlib.md5(content.encode("utf-8")).hexdigest()
        if deduplicate and content_hash in seen_hashes:
            continue
        seen_hashes.add(content_hash)

        metadata = _normalize_metadata(doc.metadata or {}, idx)
        metadata["language"] = language
        metadata["content_hash"] = content_hash

        processed.append(Document(page_content=content, metadata=metadata))

    return processed


def save_processed_documents(
    docs: Iterable[Document],
    output_dir: str,
    filename: str = "knowledge_base.jsonl",
) -> Path:
    """Persist processed documents to disk in JSON Lines format."""

    os.makedirs(output_dir, exist_ok=True)
    output_path = Path(output_dir) / filename

    with output_path.open("w", encoding="utf-8") as fp:
        for doc in docs:
            metadata = doc.metadata or {}
            record = {
                "id": metadata.get("id"),
                "source": metadata.get("source"),
                "title": metadata.get("title"),
                "content": doc.page_content,
            }
            if metadata.get("chunk_id") is not None:
                record["chunk_id"] = metadata["chunk_id"]
            if metadata.get("language"):
                record["language"] = metadata["language"]
            if metadata.get("document_id"):
                record["document_id"] = metadata["document_id"]

            fp.write(json.dumps(record, ensure_ascii=False) + "\n")

    return output_path


def _normalize_metadata(metadata: dict, sequence: int) -> dict:
    normalized = dict(metadata)

    source = _coerce_source(normalized)
    title = _coerce_title(normalized, source, sequence)

    document_id = normalized.get("document_id") or normalized.get("id")
    if not document_id:
        document_id = _generate_document_id(title, sequence)

    normalized["source"] = source
    normalized["title"] = title
    normalized["document_id"] = document_id
    normalized["id"] = document_id
    normalized.setdefault("type", "text")
    normalized.setdefault("sequence", sequence)

    return normalized


def _coerce_source(metadata: dict) -> str:
    for key in ("source", "file_path", "filepath", "path", "uri"):
        if metadata.get(key):
            return str(metadata[key])
    return f"document_{metadata.get('sequence', 'unknown')}"


def _coerce_title(metadata: dict, source: str, sequence: int) -> str:
    title = metadata.get("title")
    if title:
        return str(title)

    stem = Path(source).stem if source else f"document_{sequence}"
    stem = stem.replace("_", " ").strip()
    return stem.title() or f"Document {sequence}"


def _generate_document_id(title: str, sequence: int) -> str:
    slug = _NON_WORD_RE.sub("-", title.lower()).strip("-")
    if not slug:
        slug = f"document-{sequence}"
    return f"{slug}-{sequence}"


def _safe_detect_language(text: str) -> Optional[str]:
    if not text or not _detect_language:
        return None
    sample = text[:1000]
    try:
        return _detect_language(sample)
    except LangDetectException:
        return None

