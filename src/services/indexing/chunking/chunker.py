from typing import List

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


def chunk_documents(docs: List[Document]) -> List[Document]:
    """Split documents into contextual chunks with metadata suitable for embeddings."""

    chunked_docs: List[Document] = []

    for doc in docs:
        doc_type = doc.metadata.get("type", "text")

        if doc_type == "table":
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=2000,
                chunk_overlap=200,
                separators=["\n\n", "\n", " ", ""],
            )
        elif doc_type == "image":
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=800,
                chunk_overlap=100,
                separators=["\n\n", "\n", ". ", " ", ""],
            )
        else:
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=500,
                chunk_overlap=100,
                separators=["\n\n", ".", "?", "!", ";", " "],
            )

        doc_splits = splitter.split_documents([doc])

        for idx, split in enumerate(doc_splits, start=1):
            metadata = dict(split.metadata)
            parent_id = metadata.get("document_id") or metadata.get("id")

            metadata["chunk_id"] = idx
            metadata["parent_id"] = parent_id
            metadata["id"] = f"{parent_id}-chunk-{idx}" if parent_id else f"chunk-{idx}"

            chunked_docs.append(
                Document(page_content=split.page_content, metadata=metadata)
            )

    return chunked_docs
