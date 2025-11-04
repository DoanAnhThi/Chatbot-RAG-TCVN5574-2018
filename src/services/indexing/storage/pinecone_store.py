import os
from typing import List, Optional

from langchain_community.vectorstores import Pinecone
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document
import pinecone

from .base_vector_store import BaseVectorStore


class PineconeVectorStore(BaseVectorStore):
    """Pinecone-based vector store implementation"""

    def __init__(self, embeddings: OpenAIEmbeddings, index_name: str = "pickdi-chatbot", api_key: Optional[str] = None, environment: Optional[str] = None, **kwargs):
        super().__init__(embeddings, **kwargs)
        self.index_name = index_name
        self.api_key = api_key or os.getenv("PINECONE_API_KEY")
        self.environment = environment or os.getenv("PINECONE_ENVIRONMENT")
        self.vectorstore: Optional[Pinecone] = None

        if not self.api_key:
            raise ValueError("Pinecone API key is required. Set PINECONE_API_KEY environment variable or pass api_key parameter.")
        if not self.environment:
            raise ValueError("Pinecone environment is required. Set PINECONE_ENVIRONMENT environment variable or pass environment parameter.")

        # Initialize Pinecone
        pinecone.init(api_key=self.api_key, environment=self.environment)

    def add_documents(self, documents: List[Document]) -> None:
        """Add documents to the Pinecone vector store"""
        if self.vectorstore is None:
            self.vectorstore = Pinecone.from_documents(
                documents=documents,
                embedding=self.embeddings,
                index_name=self.index_name
            )
        else:
            self.vectorstore.add_documents(documents)

    def similarity_search(self, query: str, k: int = 4, **kwargs) -> List[Document]:
        """Perform similarity search and return top-k results"""
        if self.vectorstore is None:
            # Try to load existing index
            try:
                self.vectorstore = Pinecone.from_existing_index(
                    index_name=self.index_name,
                    embedding=self.embeddings
                )
            except Exception as e:
                raise ValueError(f"Vector store not initialized and no existing index found: {e}")

        return self.vectorstore.similarity_search(query, k=k, **kwargs)

    def delete(self, ids: Optional[List[str]] = None, **kwargs) -> None:
        """Delete documents from the Pinecone vector store"""
        if self.vectorstore is None:
            return

        if ids:
            self.vectorstore.delete(ids=ids)
        else:
            # Delete all documents if no IDs specified
            index = pinecone.Index(self.index_name)
            index.delete(delete_all=True)

    def persist(self) -> None:
        """Pinecone automatically persists data, no action needed"""
        pass

    def load(self) -> None:
        """Load the Pinecone vector store from existing index"""
        try:
            self.vectorstore = Pinecone.from_existing_index(
                index_name=self.index_name,
                embedding=self.embeddings
            )
        except Exception as e:
            raise ValueError(f"Failed to load Pinecone index '{self.index_name}': {e}")

    def get_collection_info(self) -> dict:
        """Get information about the Pinecone vector store"""
        try:
            index = pinecone.Index(self.index_name)
            stats = index.describe_index_stats()

            return {
                "status": "loaded" if self.vectorstore else "not_loaded",
                "documents_count": stats.total_vector_count,
                "store_type": "Pinecone",
                "index_name": self.index_name,
                "dimension": stats.dimension,
                "environment": self.environment
            }
        except Exception as e:
            return {
                "status": "error",
                "error": str(e),
                "store_type": "Pinecone",
                "index_name": self.index_name
            }

    @property
    def is_persistent(self) -> bool:
        """Return True as Pinecone is a persistent cloud database"""
        return True

    def create_index_if_not_exists(self, dimension: int = 1536, metric: str = "cosine") -> None:
        """Create Pinecone index if it doesn't exist"""
        if self.index_name not in pinecone.list_indexes():
            pinecone.create_index(
                name=self.index_name,
                dimension=dimension,
                metric=metric
            )

    def delete_index(self) -> None:
        """Delete the Pinecone index"""
        if self.index_name in pinecone.list_indexes():
            pinecone.delete_index(self.index_name)
