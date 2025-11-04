import os
from typing import List, Optional

from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document

from .base_vector_store import BaseVectorStore


class ChromaDBVectorStore(BaseVectorStore):
    """ChromaDB-based vector store implementation"""

    def __init__(self, embeddings: OpenAIEmbeddings, collection_name: str = "pickdi_chatbot", persist_directory: Optional[str] = None, **kwargs):
        super().__init__(embeddings, **kwargs)
        self.collection_name = collection_name
        self.persist_directory = persist_directory or "./vectorstore/chroma_db"
        self.vectorstore: Optional[Chroma] = None

    def add_documents(self, documents: List[Document]) -> None:
        """Add documents to the ChromaDB vector store"""
        if self.vectorstore is None:
            self.vectorstore = Chroma.from_documents(
                documents=documents,
                embedding=self.embeddings,
                collection_name=self.collection_name,
                persist_directory=self.persist_directory
            )
        else:
            self.vectorstore.add_documents(documents)

    def similarity_search(self, query: str, k: int = 4, **kwargs) -> List[Document]:
        """Perform similarity search and return top-k results"""
        if self.vectorstore is None:
            # Try to load existing collection
            try:
                self.vectorstore = Chroma(
                    collection_name=self.collection_name,
                    embedding_function=self.embeddings,
                    persist_directory=self.persist_directory
                )
            except Exception as e:
                raise ValueError(f"Vector store not initialized and no existing collection found: {e}")

        return self.vectorstore.similarity_search(query, k=k, **kwargs)

    def delete(self, ids: Optional[List[str]] = None, **kwargs) -> None:
        """Delete documents from the ChromaDB vector store"""
        if self.vectorstore is None:
            return

        if ids:
            self.vectorstore.delete(ids=ids)
        else:
            # Delete entire collection if no IDs specified
            try:
                self.vectorstore.delete_collection()
                self.vectorstore = None
            except Exception as e:
                print(f"Warning: Failed to delete collection: {e}")

    def persist(self) -> None:
        """Persist the ChromaDB vector store to disk"""
        if self.vectorstore is not None:
            self.vectorstore.persist()

    def load(self) -> None:
        """Load the ChromaDB vector store from existing collection"""
        try:
            self.vectorstore = Chroma(
                collection_name=self.collection_name,
                embedding_function=self.embeddings,
                persist_directory=self.persist_directory
            )
        except Exception as e:
            raise ValueError(f"Failed to load ChromaDB collection '{self.collection_name}': {e}")

    def get_collection_info(self) -> dict:
        """Get information about the ChromaDB vector store"""
        if self.vectorstore is None:
            return {
                "status": "not_initialized",
                "documents_count": 0,
                "store_type": "ChromaDB",
                "collection_name": self.collection_name
            }

        try:
            count = self.vectorstore._collection.count() if hasattr(self.vectorstore, '_collection') else 0
        except:
            count = 0

        return {
            "status": "loaded",
            "documents_count": count,
            "store_type": "ChromaDB",
            "collection_name": self.collection_name,
            "persist_directory": self.persist_directory
        }

    @property
    def is_persistent(self) -> bool:
        """Return True as ChromaDB supports persistence"""
        return True

    def reset_collection(self) -> None:
        """Reset the collection by deleting and recreating it"""
        if self.vectorstore is not None:
            try:
                self.vectorstore.delete_collection()
            except:
                pass
        self.vectorstore = Chroma(
            collection_name=self.collection_name,
            embedding_function=self.embeddings,
            persist_directory=self.persist_directory
        )

    def get_collection(self):
        """Get the underlying Chroma collection object"""
        if self.vectorstore is None:
            return None
        return self.vectorstore._collection if hasattr(self.vectorstore, '_collection') else None
