from abc import ABC, abstractmethod
from typing import List, Any, Optional

from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings


class BaseVectorStore(ABC):
    """Abstract base class for vector store implementations"""

    def __init__(self, embeddings: OpenAIEmbeddings, **kwargs):
        self.embeddings = embeddings
        self.config = kwargs

    @abstractmethod
    def add_documents(self, documents: List[Document]) -> None:
        """Add documents to the vector store"""
        pass

    @abstractmethod
    def similarity_search(self, query: str, k: int = 4, **kwargs) -> List[Document]:
        """Perform similarity search and return top-k results"""
        pass

    @abstractmethod
    def delete(self, ids: Optional[List[str]] = None, **kwargs) -> None:
        """Delete documents from the vector store"""
        pass

    @abstractmethod
    def persist(self) -> None:
        """Persist the vector store to storage"""
        pass

    @abstractmethod
    def load(self) -> None:
        """Load the vector store from storage"""
        pass

    @abstractmethod
    def get_collection_info(self) -> dict:
        """Get information about the vector store collection"""
        pass

    @property
    @abstractmethod
    def is_persistent(self) -> bool:
        """Return True if the store supports persistence"""
        pass
