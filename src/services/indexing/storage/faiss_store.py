import os
from typing import List, Optional

from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document

from src.config import settings
from .base_vector_store import BaseVectorStore


class FAISSVectorStore(BaseVectorStore):
    """FAISS-based vector store implementation"""

    def __init__(self, embeddings: OpenAIEmbeddings, vectorstore_dir: Optional[str] = None, **kwargs):
        super().__init__(embeddings, **kwargs)
        self.vectorstore_dir = vectorstore_dir or settings.vectorstore_dir
        self.vectorstore: Optional[FAISS] = None

    def add_documents(self, documents: List[Document]) -> None:
        """Add documents to the FAISS vector store"""
        if self.vectorstore is None:
            self.vectorstore = FAISS.from_documents(documents, self.embeddings)
        else:
            self.vectorstore.add_documents(documents)

    def similarity_search(self, query: str, k: int = 4, **kwargs) -> List[Document]:
        """Perform similarity search and return top-k results"""
        if self.vectorstore is None:
            raise ValueError("Vector store not initialized. Call load() or add_documents() first.")
        return self.vectorstore.similarity_search(query, k=k, **kwargs)

    def delete(self, ids: Optional[List[str]] = None, **kwargs) -> None:
        """Delete documents from the FAISS vector store"""
        if self.vectorstore is None:
            return
        # FAISS doesn't support deletion by IDs directly
        # This would require rebuilding the index
        raise NotImplementedError("FAISS does not support deletion by IDs. Consider using a different vector store.")

    def persist(self) -> None:
        """Persist the FAISS vector store to disk"""
        if self.vectorstore is None:
            raise ValueError("No vectorstore to persist. Call add_documents() first.")
        os.makedirs(self.vectorstore_dir, exist_ok=True)
        self.vectorstore.save_local(self.vectorstore_dir)

    def load(self) -> None:
        """Load the FAISS vector store from disk"""
        if not os.path.exists(self.vectorstore_dir):
            raise FileNotFoundError(f"Vector store directory {self.vectorstore_dir} does not exist")
        self.vectorstore = FAISS.load_local(self.vectorstore_dir, self.embeddings, allow_dangerous_deserialization=True)

    def get_collection_info(self) -> dict:
        """Get information about the FAISS vector store"""
        if self.vectorstore is None:
            return {"status": "not_initialized", "documents_count": 0}

        try:
            # Try to get the number of documents
            docs_count = len(self.vectorstore.index_to_docstore_id) if hasattr(self.vectorstore, 'index_to_docstore_id') else 0
        except:
            docs_count = 0

        return {
            "status": "loaded" if self.vectorstore else "not_loaded",
            "documents_count": docs_count,
            "store_type": "FAISS",
            "persist_directory": self.vectorstore_dir
        }

    @property
    def is_persistent(self) -> bool:
        """Return True as FAISS supports persistence"""
        return True


# Legacy functions for backward compatibility
def load_existing_vectorstore(vectorstore_dir: str, embeddings: OpenAIEmbeddings) -> FAISS:
    """Load existing FAISS vectorstore from disk (legacy function)"""
    return FAISS.load_local(vectorstore_dir, embeddings, allow_dangerous_deserialization=True)


def create_vectorstore_from_documents(docs: List[Document], embeddings: OpenAIEmbeddings, vectorstore_dir: str) -> FAISS:
    """Create new FAISS vectorstore from documents and save to disk (legacy function)"""
    vectorstore = FAISS.from_documents(docs, embeddings)
    vectorstore.save_local(vectorstore_dir)
    return vectorstore
