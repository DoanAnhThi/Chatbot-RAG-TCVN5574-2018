from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

from langchain_core.documents import Document

from src.config import settings
from ..chunking.chunker import chunk_documents
from ..loaders.loader_image import load_image_documents
from ..loaders.loader_pdf import load_pdf_documents
from ..loaders.loader_table import load_table_documents
from ..loaders.loader_text import load_text_documents
from .pipeline import preprocess_documents, save_processed_documents


DEFAULT_OUTPUT_FILENAME = "knowledge_base.jsonl"


def load_raw_documents(data_dir: str) -> List[Document]:
    if not Path(data_dir).exists():
        return []

    docs: List[Document] = []
    docs.extend(load_pdf_documents(data_dir))
    docs.extend(load_text_documents(data_dir))
    docs.extend(load_table_documents(data_dir))
    docs.extend(load_image_documents(data_dir))
    return docs


def clean_documents(
    *,
    data_dir: str,
    processed_dir: str,
    output_filename: str = DEFAULT_OUTPUT_FILENAME,
    target_language: str | None = None,
) -> Path:
    docs = load_raw_documents(data_dir)
    if not docs:
        raise RuntimeError(f"No raw documents found in {data_dir}.")

    processed_docs = preprocess_documents(
        docs,
        target_language=target_language or settings.primary_language,
    )
    if not processed_docs:
        raise RuntimeError("No documents remained after preprocessing.")

    chunked_docs = chunk_documents(processed_docs)
    if not chunked_docs:
        raise RuntimeError("Chunking produced no documents. Check preprocessing configuration.")

    output_path = save_processed_documents(
        chunked_docs,
        processed_dir,
        filename=output_filename,
    )

    return output_path


def parse_args(argv: List[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Clean raw documents and export processed JSONL chunks.",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default=settings.data_dir,
        help="Directory containing raw documents (defaults to settings.data_dir).",
    )
    parser.add_argument(
        "--processed-dir",
        type=str,
        default=settings.processed_data_dir,
        help="Directory to store processed outputs (defaults to settings.processed_data_dir).",
    )
    parser.add_argument(
        "--output-file",
        type=str,
        default=DEFAULT_OUTPUT_FILENAME,
        help="Filename for the processed JSONL export (defaults to knowledge_base.jsonl).",
    )
    parser.add_argument(
        "--language",
        type=str,
        default=None,
        help="Filter documents by ISO language code (defaults to settings.primary_language).",
    )
    return parser.parse_args(argv)


def main(argv: List[str] | None = None) -> None:
    args = parse_args(argv)

    output_path = clean_documents(
        data_dir=args.data_dir,
        processed_dir=args.processed_dir,
        output_filename=args.output_file,
        target_language=args.language,
    )

    print("Processed dataset written to:", output_path)


if __name__ == "__main__":  # pragma: no cover
    main()

