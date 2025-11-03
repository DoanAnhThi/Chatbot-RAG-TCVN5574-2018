from typing import List

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document


def chunk_documents(docs: List[Document]) -> List[Document]:
    """Split documents with different strategies based on content type"""
    splits = []
    for doc in docs:
        doc_type = doc.metadata.get("type", "text")

        if doc_type == "table":
            # For tables, use larger chunks to preserve structure
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=2000,  # Larger chunks for tables
                chunk_overlap=200,
                separators=["\n\n", "\n", " ", ""]  # Prefer to split on double newlines
            )
        elif doc_type == "image":
            # For images, use smaller chunks but keep OCR text together
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=800,  # Smaller chunks for images
                chunk_overlap=100,
                separators=["\n\n", "\n", ". ", " ", ""]  # Keep OCR text coherent
            )
        else:
            # Default splitter for text/PDF
            splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)

        doc_splits = splitter.split_documents([doc])
        splits.extend(doc_splits)

    return splits
