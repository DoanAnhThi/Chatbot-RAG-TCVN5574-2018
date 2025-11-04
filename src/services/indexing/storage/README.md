# Vector Store Implementations

This package provides multiple vector store implementations for storing and retrieving vector embeddings. All implementations follow a common interface defined by `BaseVectorStore`.

## Supported Vector Stores

### 1. FAISS (Facebook AI Similarity Search)
- **Type**: Local file-based storage
- **Best for**: Small to medium datasets, offline usage
- **Persistence**: Local disk storage

```python
from src.services.indexing.storage import FAISSVectorStore
from langchain_openai import OpenAIEmbeddings

embeddings = OpenAIEmbeddings()
store = FAISSVectorStore(embeddings, vectorstore_dir="./vectorstore/faiss_index")
```

### 2. Pinecone
- **Type**: Cloud-based vector database
- **Best for**: Production applications, large datasets
- **Persistence**: Cloud storage

```python
from src.services.indexing.storage import PineconeVectorStore
from langchain_openai import OpenAIEmbeddings

embeddings = OpenAIEmbeddings()
store = PineconeVectorStore(
    embeddings,
    index_name="pickdi-chatbot",
    api_key="your-api-key",
    environment="your-environment"
)
```

### 3. ChromaDB
- **Type**: Local vector database
- **Best for**: Development, small to medium datasets
- **Persistence**: Local disk storage

```python
from src.services.indexing.storage import ChromaDBVectorStore
from langchain_openai import OpenAIEmbeddings

embeddings = OpenAIEmbeddings()
store = ChromaDBVectorStore(
    embeddings,
    collection_name="pickdi_chatbot",
    persist_directory="./vectorstore/chroma_db"
)
```

### 4. Weaviate
- **Type**: Graph-based vector database
- **Best for**: Complex queries, metadata filtering
- **Persistence**: Local or cloud storage

```python
from src.services.indexing.storage import WeaviateVectorStore
from langchain_openai import OpenAIEmbeddings

embeddings = OpenAIEmbeddings()
store = WeaviateVectorStore(
    embeddings,
    index_name="PickdiChatbot",
    url="http://localhost:8080"  # or your Weaviate Cloud URL
)
```

### 5. Qdrant
- **Type**: High-performance vector database
- **Best for**: High-dimensional vectors, real-time updates
- **Persistence**: Local or cloud storage

```python
from src.services.indexing.storage import QdrantVectorStore
from langchain_openai import OpenAIEmbeddings

embeddings = OpenAIEmbeddings()
store = QdrantVectorStore(
    embeddings,
    collection_name="pickdi_chatbot",
    url="http://localhost:6333"  # or your Qdrant Cloud URL
)
```

## Using the Factory Pattern

For easier configuration and switching between stores, use the `VectorStoreFactory`:

```python
from src.services.indexing.storage import VectorStoreFactory
from langchain_openai import OpenAIEmbeddings

embeddings = OpenAIEmbeddings()

# Create any store type dynamically
store = VectorStoreFactory.create_vector_store(
    "chromadb",  # or "faiss", "pinecone", "weaviate", "qdrant"
    embeddings,
    collection_name="my_collection",
    persist_directory="./data"
)
```

## Common Interface

All vector stores implement the same interface:

```python
# Add documents
store.add_documents(documents)

# Search for similar documents
results = store.similarity_search("query", k=5)

# Delete documents
store.delete(ids=["doc1", "doc2"])

# Persist data (if applicable)
store.persist()

# Load from storage
store.load()

# Get collection info
info = store.get_collection_info()
```

## Environment Variables

Set these environment variables for cloud-based stores:

### Pinecone
- `PINECONE_API_KEY`: Your Pinecone API key
- `PINECONE_ENVIRONMENT`: Your Pinecone environment

### Weaviate
- `WEAVIATE_URL`: Weaviate server URL (default: http://localhost:8080)
- `WEAVIATE_API_KEY`: Weaviate API key (if using Weaviate Cloud)

### Qdrant
- `QDRANT_URL`: Qdrant server URL
- `QDRANT_API_KEY`: Qdrant API key (if using Qdrant Cloud)

## Requirements

Add these dependencies to your `requirements.txt`:

```
langchain-community
langchain-openai
faiss-cpu  # for FAISS
pinecone-client  # for Pinecone
chromadb  # for ChromaDB
weaviate-client  # for Weaviate
qdrant-client  # for Qdrant
```

## Migration Between Stores

Since all stores implement the same interface, you can easily switch between them:

```python
# Export from current store
documents = current_store.similarity_search("", k=10000)  # Get all docs

# Import to new store
new_store.add_documents(documents)
new_store.persist()
```
