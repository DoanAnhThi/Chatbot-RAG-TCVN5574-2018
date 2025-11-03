from __future__ import annotations

import os
from typing import List

from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document

from app.config import settings
from .loaders.loader_text import load_text_documents
from .loaders.loader_pdf import load_pdf_documents
from .loaders.loader_table import load_table_documents
from .loaders.loader_image import load_image_documents
from .chunking.chunker import chunk_documents
from .embedding.embedder import create_embeddings_model
from .storage.faiss_store import load_existing_vectorstore, create_vectorstore_from_documents


def build_or_load_vectorstore() -> FAISS:
    os.makedirs(os.path.dirname(settings.vectorstore_dir), exist_ok=True)

    if os.path.isdir(settings.vectorstore_dir):
        try:
            embeddings = create_embeddings_model()
            return load_existing_vectorstore(settings.vectorstore_dir, embeddings)
        except Exception:
            pass

    # Load documents
    docs = load_documents(settings.data_dir)
    if not docs:
        raise RuntimeError("No documents found to index. Place files in data directory.")

    # Chunk documents
    chunked_docs = chunk_documents(docs)

    # Create embeddings model
    embeddings = create_embeddings_model()

    # Create and save vectorstore
    return create_vectorstore_from_documents(chunked_docs, embeddings, settings.vectorstore_dir)


def load_documents(data_dir: str) -> List[Document]:
    """Load all documents from different sources"""
    if not os.path.isdir(data_dir):
        return []

    docs: List[Document] = []

    # Load documents from each source type
    pdf_docs = load_pdf_documents(data_dir)
    docs.extend(pdf_docs)

    text_docs = load_text_documents(data_dir)
    docs.extend(text_docs)

    table_docs = load_table_documents(data_dir)
    docs.extend(table_docs)

    image_docs = load_image_documents(data_dir)
    docs.extend(image_docs)

    return docs
