import os
from typing import List, Optional

from langchain_community.vectorstores import Qdrant
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document
from qdrant_client import QdrantClient
from qdrant_client.http import models

from .base_vector_store import BaseVectorStore


class QdrantVectorStore(BaseVectorStore):
    """Qdrant-based vector store implementation"""

    def __init__(self, embeddings: OpenAIEmbeddings, collection_name: str = "pickdi_chatbot", url: Optional[str] = None, api_key: Optional[str] = None, path: Optional[str] = None, **kwargs):
        super().__init__(embeddings, **kwargs)
        self.collection_name = collection_name
        self.url = url or os.getenv("QDRANT_URL")
        self.api_key = api_key or os.getenv("QDRANT_API_KEY")
        self.path = path or "./vectorstore/qdrant_db"
        self.vectorstore: Optional[Qdrant] = None

        # Initialize Qdrant client
        if self.url:
            self.client = QdrantClient(url=self.url, api_key=self.api_key)
        else:
            self.client = QdrantClient(path=self.path)

    def add_documents(self, documents: List[Document]) -> None:
        """Add documents to the Qdrant vector store"""
        if self.vectorstore is None:
            self.vectorstore = Qdrant.from_documents(
                documents=documents,
                embedding=self.embeddings,
                collection_name=self.collection_name,
                client=self.client
            )
        else:
            self.vectorstore.add_documents(documents)

    def similarity_search(self, query: str, k: int = 4, **kwargs) -> List[Document]:
        """Perform similarity search and return top-k results"""
        if self.vectorstore is None:
            # Try to load existing collection
            try:
                self.vectorstore = Qdrant(
                    client=self.client,
                    collection_name=self.collection_name,
                    embeddings=self.embeddings
                )
            except Exception as e:
                raise ValueError(f"Vector store not initialized and no existing collection found: {e}")

        return self.vectorstore.similarity_search(query, k=k, **kwargs)

    def delete(self, ids: Optional[List[str]] = None, **kwargs) -> None:
        """Delete documents from the Qdrant vector store"""
        if ids:
            # Delete specific points
            point_ids = [int(id) if id.isdigit() else id for id in ids]
            self.client.delete(
                collection_name=self.collection_name,
                points_selector=models.PointIdsList(points=point_ids)
            )
        else:
            # Delete entire collection if no IDs specified
            try:
                self.client.delete_collection(self.collection_name)
                self.vectorstore = None
            except Exception as e:
                print(f"Warning: Failed to delete collection: {e}")

    def persist(self) -> None:
        """Qdrant automatically persists data, no action needed"""
        pass

    def load(self) -> None:
        """Load the Qdrant vector store from existing collection"""
        try:
            self.vectorstore = Qdrant(
                client=self.client,
                collection_name=self.collection_name,
                embeddings=self.embeddings
            )
        except Exception as e:
            raise ValueError(f"Failed to load Qdrant collection '{self.collection_name}': {e}")

    def get_collection_info(self) -> dict:
        """Get information about the Qdrant vector store"""
        try:
            # Check if collection exists
            collections = self.client.get_collections()
            collection_names = [col.name for col in collections.collections]

            if self.collection_name not in collection_names:
                return {
                    "status": "not_found",
                    "documents_count": 0,
                    "store_type": "Qdrant",
                    "collection_name": self.collection_name
                }

            # Get collection info
            info = self.client.get_collection(self.collection_name)
            count = self.client.count(self.collection_name).count

            return {
                "status": "loaded" if self.vectorstore else "not_loaded",
                "documents_count": count,
                "store_type": "Qdrant",
                "collection_name": self.collection_name,
                "vector_size": info.config.params.vectors.size,
                "distance": str(info.config.params.vectors.distance)
            }
        except Exception as e:
            return {
                "status": "error",
                "error": str(e),
                "store_type": "Qdrant",
                "collection_name": self.collection_name
            }

    @property
    def is_persistent(self) -> bool:
        """Return True as Qdrant supports persistence"""
        return True

    def create_collection_if_not_exists(self, vector_size: int = 1536, distance: str = "Cosine") -> None:
        """Create Qdrant collection if it doesn't exist"""
        collections = self.client.get_collections()
        collection_names = [col.name for col in collections.collections]

        if self.collection_name not in collection_names:
            distance_enum = getattr(models.Distance, distance.upper())
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=models.VectorParams(
                    size=vector_size,
                    distance=distance_enum
                )
            )

    def delete_collection(self) -> None:
        """Delete the Qdrant collection"""
        try:
            self.client.delete_collection(self.collection_name)
            self.vectorstore = None
        except Exception as e:
            print(f"Warning: Failed to delete collection: {e}")

    def search_with_score(self, query: str, k: int = 4) -> List[tuple]:
        """Search with similarity scores"""
        if self.vectorstore is None:
            self.load()
        return self.vectorstore.similarity_search_with_score(query, k=k)
