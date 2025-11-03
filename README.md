## RAG Chatbot (LangChain + LangGraph)

Dockerized internal chatbot using LangChain for retrieval and LangGraph for orchestration:
- Node 1: Query retriever
- Node 2: LLM answer generation
- Node 3: Confidence evaluation
- Node 4: Clarifying question if confidence < 0.7
- Node 5: Output to frontend

### Prerequisites
- Docker and Docker Compose installed
- OpenAI API key (or extend to your LLM provider)

### Quick start
1. Copy environment and set secrets:
   ```bash
   cp .env.example .env
   # then edit .env to set OPENAI_API_KEY and any overrides
   ```
2. Add your documents into `retrieval/data/` (PDF, TXT, MD supported out-of-the-box).
3. Build and run:
   ```bash
   docker compose up --build
   ```
4. Access the API at `http://localhost:8080/`.

Indexing runs automatically on startup if no vector store is found. You can also run manual indexing:
```bash
docker compose run --rm rag-chatbot python scripts/index_documents.py
```

### Project structure
- `app/` FastAPI backend, LangGraph pipeline, RAG utilities
- `app/vectorstore/` Persisted FAISS index
- `data/` Source documents to index
- `scripts/` Indexing and retrieval utilities

### Environment
- `OPENAI_MODEL` default `gpt-4o-mini`
- `EMBEDDINGS_MODEL` default `text-embedding-3-small`
- `RETRIEVAL_TOP_K` default `4`
- `CONFIDENCE_THRESHOLD` default `0.7`

### Extending
- Swap embeddings/LLM (e.g., local via Ollama or HF) by editing `app/config.py` and the constructors in `app/graph/pipeline.py` and `app/rag/indexer.py`.
- Add loaders for more file types in `app/rag/indexer.py`.
