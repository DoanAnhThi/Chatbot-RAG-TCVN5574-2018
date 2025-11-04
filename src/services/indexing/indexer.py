from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path
from typing import Iterable, List

from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document

from src.config import settings
from .embedding.embedder import create_embeddings_model
from .storage.faiss_store import load_existing_vectorstore, create_vectorstore_from_documents


DEFAULT_PROCESSED_FILENAME = "knowledge_base.jsonl"


def build_or_load_vectorstore(
    *,
    force_rebuild: bool = False,
    processed_dir: str | None = None,
    processed_file: str | None = None,
    vectorstore_dir: str | None = None,
) -> FAISS:
    vectorstore_path = vectorstore_dir or settings.vectorstore_dir

    if force_rebuild and os.path.isdir(vectorstore_path):
        shutil.rmtree(vectorstore_path)

    vectorstore_parent = os.path.dirname(vectorstore_path) or "."
    os.makedirs(vectorstore_parent, exist_ok=True)

    if os.path.isdir(vectorstore_path):
        try:
            embeddings = create_embeddings_model()
            return load_existing_vectorstore(vectorstore_path, embeddings)
        except Exception:
            pass

    processed_base = Path(processed_dir or settings.processed_data_dir)
    processed_path = processed_base / (processed_file or DEFAULT_PROCESSED_FILENAME)
    docs = load_processed_documents(processed_path)
    if not docs:
        raise RuntimeError(
            f"No processed documents found at {processed_path}. "
            "Run the cleaning script before indexing."
        )

    embeddings = create_embeddings_model()
    return create_vectorstore_from_documents(docs, embeddings, vectorstore_path)


def load_processed_documents(path: Path | str) -> List[Document]:
    path = Path(path)

    files: Iterable[Path]
    if path.is_dir():
        files = sorted(p for p in path.glob("*.jsonl") if p.is_file())
    else:
        files = [path] if path.exists() else []

    documents: List[Document] = []

    for file in files:
        with file.open("r", encoding="utf-8") as fp:
            for line in fp:
                line = line.strip()
                if not line:
                    continue
                try:
                    payload = json.loads(line)
                except json.JSONDecodeError:
                    continue

                content = payload.get("content")
                if not content:
                    continue

                metadata = {k: v for k, v in payload.items() if k != "content"}
                documents.append(Document(page_content=content, metadata=metadata))

    return documents


def parse_args(argv: List[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build or load the vector store from processed documents.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Rebuild the index even if an existing vectorstore is detected.",
    )
    parser.add_argument(
        "--processed-dir",
        type=str,
        default=None,
        help="Directory containing processed JSONL documents (defaults to settings.processed_data_dir).",
    )
    parser.add_argument(
        "--processed-file",
        type=str,
        default=None,
        help="Specific JSONL file to load within the processed directory (defaults to knowledge_base.jsonl).",
    )
    parser.add_argument(
        "--vectorstore-dir",
        type=str,
        default=None,
        help="Override the vectorstore directory (defaults to settings.vectorstore_dir).",
    )
    return parser.parse_args(argv)


def main(argv: List[str] | None = None) -> None:
    args = parse_args(argv)

    vectorstore = build_or_load_vectorstore(
        force_rebuild=args.force,
        processed_dir=args.processed_dir,
        processed_file=args.processed_file,
        vectorstore_dir=args.vectorstore_dir,
    )

    target_vs_dir = args.vectorstore_dir or settings.vectorstore_dir
    print(f"Vector store ready at: {target_vs_dir}")
    if hasattr(vectorstore, "index"):
        dimensionality = getattr(vectorstore.index, "d", None)
        if dimensionality is not None:
            print(f"Index dimensionality: {dimensionality}")


if __name__ == "__main__":  # pragma: no cover
    main()
