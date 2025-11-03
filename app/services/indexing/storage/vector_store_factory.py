from typing import Dict, Any, Optional
from langchain_openai import OpenAIEmbeddings

from .base_vector_store import BaseVectorStore
from .faiss_store import FAISSVectorStore
from .pinecone_store import PineconeVectorStore
from .chromadb_store import ChromaDBVectorStore
from .weaviate_store import WeaviateVectorStore
from .qdrant_store import QdrantVectorStore


class VectorStoreFactory:
    """Factory class for creating vector store instances"""

    @staticmethod
    def create_vector_store(
        store_type: str,
        embeddings: OpenAIEmbeddings,
        **kwargs
    ) -> BaseVectorStore:
        """
        Create a vector store instance based on the store type.

        Args:
            store_type: Type of vector store ('faiss', 'pinecone', 'chromadb', 'weaviate', 'qdrant')
            embeddings: OpenAI embeddings instance
            **kwargs: Additional configuration parameters for the vector store

        Returns:
            BaseVectorStore: Instance of the requested vector store

        Raises:
            ValueError: If store_type is not supported
        """
        store_type = store_type.lower()

        if store_type == "faiss":
            return FAISSVectorStore(embeddings, **kwargs)
        elif store_type == "pinecone":
            return PineconeVectorStore(embeddings, **kwargs)
        elif store_type == "chromadb":
            return ChromaDBVectorStore(embeddings, **kwargs)
        elif store_type == "weaviate":
            return WeaviateVectorStore(embeddings, **kwargs)
        elif store_type == "qdrant":
            return QdrantVectorStore(embeddings, **kwargs)
        else:
            supported_types = ["faiss", "pinecone", "chromadb", "weaviate", "qdrant"]
            raise ValueError(f"Unsupported vector store type: {store_type}. Supported types: {supported_types}")

    @staticmethod
    def get_supported_store_types() -> list:
        """Get list of supported vector store types"""
        return ["faiss", "pinecone", "chromadb", "weaviate", "qdrant"]

    @staticmethod
    def get_store_config_template(store_type: str) -> Dict[str, Any]:
        """
        Get configuration template for a specific vector store type.

        Args:
            store_type: Type of vector store

        Returns:
            Dict containing configuration parameters and their descriptions
        """
        templates = {
            "faiss": {
                "vectorstore_dir": {
                    "default": "./vectorstore/faiss_index",
                    "description": "Directory to store FAISS index files"
                }
            },
            "pinecone": {
                "index_name": {
                    "default": "pickdi-chatbot",
                    "description": "Name of the Pinecone index"
                },
                "api_key": {
                    "default": None,
                    "description": "Pinecone API key (can also be set via PINECONE_API_KEY env var)"
                },
                "environment": {
                    "default": None,
                    "description": "Pinecone environment (can also be set via PINECONE_ENVIRONMENT env var)"
                }
            },
            "chromadb": {
                "collection_name": {
                    "default": "pickdi_chatbot",
                    "description": "Name of the ChromaDB collection"
                },
                "persist_directory": {
                    "default": "./vectorstore/chroma_db",
                    "description": "Directory to persist ChromaDB data"
                }
            },
            "weaviate": {
                "index_name": {
                    "default": "PickdiChatbot",
                    "description": "Name of the Weaviate class"
                },
                "url": {
                    "default": "http://localhost:8080",
                    "description": "Weaviate server URL (can also be set via WEAVIATE_URL env var)"
                },
                "api_key": {
                    "default": None,
                    "description": "Weaviate API key (can also be set via WEAVIATE_API_KEY env var)"
                }
            },
            "qdrant": {
                "collection_name": {
                    "default": "pickdi_chatbot",
                    "description": "Name of the Qdrant collection"
                },
                "url": {
                    "default": None,
                    "description": "Qdrant server URL (can also be set via QDRANT_URL env var)"
                },
                "api_key": {
                    "default": None,
                    "description": "Qdrant API key (can also be set via QDRANT_API_KEY env var)"
                },
                "path": {
                    "default": "./vectorstore/qdrant_db",
                    "description": "Local path for Qdrant storage (used if url is not provided)"
                }
            }
        }

        store_type = store_type.lower()
        if store_type not in templates:
            raise ValueError(f"Unsupported vector store type: {store_type}")

        return templates[store_type]
