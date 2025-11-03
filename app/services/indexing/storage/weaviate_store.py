import os
from typing import List, Optional

from langchain_community.vectorstores import Weaviate
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document
import weaviate

from .base_vector_store import BaseVectorStore


class WeaviateVectorStore(BaseVectorStore):
    """Weaviate-based vector store implementation"""

    def __init__(self, embeddings: OpenAIEmbeddings, index_name: str = "PickdiChatbot", url: Optional[str] = None, api_key: Optional[str] = None, **kwargs):
        super().__init__(embeddings, **kwargs)
        self.index_name = index_name
        self.url = url or os.getenv("WEAVIATE_URL", "http://localhost:8080")
        self.api_key = api_key or os.getenv("WEAVIATE_API_KEY")
        self.vectorstore: Optional[Weaviate] = None

        # Initialize Weaviate client
        auth_config = weaviate.auth.AuthApiKey(api_key=self.api_key) if self.api_key else None
        self.client = weaviate.Client(url=self.url, auth_client_secret=auth_config)

    def add_documents(self, documents: List[Document]) -> None:
        """Add documents to the Weaviate vector store"""
        if self.vectorstore is None:
            self.vectorstore = Weaviate.from_documents(
                documents=documents,
                embedding=self.embeddings,
                index_name=self.index_name,
                client=self.client
            )
        else:
            self.vectorstore.add_documents(documents)

    def similarity_search(self, query: str, k: int = 4, **kwargs) -> List[Document]:
        """Perform similarity search and return top-k results"""
        if self.vectorstore is None:
            # Try to load existing index
            try:
                self.vectorstore = Weaviate(
                    client=self.client,
                    index_name=self.index_name,
                    embedding=self.embeddings,
                    text_key="text"
                )
            except Exception as e:
                raise ValueError(f"Vector store not initialized and no existing index found: {e}")

        return self.vectorstore.similarity_search(query, k=k, **kwargs)

    def delete(self, ids: Optional[List[str]] = None, **kwargs) -> None:
        """Delete documents from the Weaviate vector store"""
        if ids:
            # Delete specific documents
            for doc_id in ids:
                self.client.data_object.delete(
                    uuid=doc_id,
                    class_name=self.index_name
                )
        else:
            # Delete entire class if no IDs specified
            try:
                self.client.schema.delete_class(self.index_name)
                self.vectorstore = None
            except Exception as e:
                print(f"Warning: Failed to delete class: {e}")

    def persist(self) -> None:
        """Weaviate automatically persists data, no action needed"""
        pass

    def load(self) -> None:
        """Load the Weaviate vector store from existing class"""
        try:
            self.vectorstore = Weaviate(
                client=self.client,
                index_name=self.index_name,
                embedding=self.embeddings,
                text_key="text"
            )
        except Exception as e:
            raise ValueError(f"Failed to load Weaviate class '{self.index_name}': {e}")

    def get_collection_info(self) -> dict:
        """Get information about the Weaviate vector store"""
        try:
            # Check if class exists
            schema = self.client.schema.get()
            classes = [cls["class"] for cls in schema["classes"]] if "classes" in schema else []

            if self.index_name not in classes:
                return {
                    "status": "not_found",
                    "documents_count": 0,
                    "store_type": "Weaviate",
                    "class_name": self.index_name
                }

            # Get object count
            result = self.client.query.aggregate(self.index_name).with_meta_count().do()
            count = result["data"]["Aggregate"][self.index_name][0]["meta"]["count"] if result.get("data", {}).get("Aggregate", {}).get(self.index_name) else 0

            return {
                "status": "loaded" if self.vectorstore else "not_loaded",
                "documents_count": count,
                "store_type": "Weaviate",
                "class_name": self.index_name,
                "url": self.url
            }
        except Exception as e:
            return {
                "status": "error",
                "error": str(e),
                "store_type": "Weaviate",
                "class_name": self.index_name
            }

    @property
    def is_persistent(self) -> bool:
        """Return True as Weaviate supports persistence"""
        return True

    def create_class_if_not_exists(self, vectorizer: str = "text2vec-openai") -> None:
        """Create Weaviate class if it doesn't exist"""
        schema = self.client.schema.get()
        classes = [cls["class"] for cls in schema["classes"]] if "classes" in schema else []

        if self.index_name not in classes:
            class_obj = {
                "class": self.index_name,
                "vectorizer": vectorizer,
                "properties": [
                    {
                        "name": "text",
                        "dataType": ["text"]
                    },
                    {
                        "name": "source",
                        "dataType": ["string"]
                    }
                ]
            }
            self.client.schema.create_class(class_obj)

    def delete_class(self) -> None:
        """Delete the Weaviate class"""
        try:
            self.client.schema.delete_class(self.index_name)
            self.vectorstore = None
        except Exception as e:
            print(f"Warning: Failed to delete class: {e}")
