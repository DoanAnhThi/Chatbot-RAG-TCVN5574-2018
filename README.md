## RAG Chatbot (LangChain + LangGraph)

Internal chatbot that combines FastAPI, LangChain, and LangGraph to retrieve and generate answers from your technical documents. The pipeline retrieves relevant chunks, generates an answer with confidence scoring, and can prompt for clarification when confidence is low.

## Local Development (no Docker)
This project is optimised for running locally in development without Docker.

1. Install Python 3.11 and create your virtual environment.
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Export the required environment variables (at minimum `OPENAI_API_KEY`). You can create a `.env` file for convenience and load it with `python-dotenv` or your shell profile.
4. Start the API using uvicorn:
   ```bash
   uvicorn src.main:app --host 0.0.0.0 --port 8000
   ```
5. The service is available at `http://localhost:8000`.

## Production (Docker)
Use Docker only when you need a production-ready image.

1. Create a `.env` file alongside `docker-compose.yml` that contains your secrets (e.g. `OPENAI_API_KEY`).
2. Build the production image:
   ```bash
   docker compose --profile production build
   ```
3. Run the stack in the background:
   ```bash
   docker compose --profile production up -d
   ```
4. The API listens on port `8080` by default (`APP_PORT` overrides this). Traffic is routed to uvicorn inside the container on port `8000`.
5. Stop the stack when needed:
   ```bash
   docker compose --profile production down
   ```

### Production notes
- The container uses `UVICORN_WORKERS` (default `4`) for multiple worker processes. Tune this via your `.env` file for the target machine.
- A named Docker volume `rag-vectorstore` persists the FAISS index at `/app/vectorstore/faiss_index` between deployments.
- When you update documents under `data/raw/`, rebuild the image so they are packaged into the next release.
- Manual indexing can be triggered inside the container:
  ```bash
  docker compose --profile production run --rm rag-chatbot bash scripts/index_documents.sh
  ```

## Environment variables

| Variable | Default | Purpose |
| --- | --- | --- |
| `OPENAI_API_KEY` | – | Required key for LLM access |
| `OPENAI_MODEL` | `gpt-4o-mini` | Override the default LLM |
| `EMBEDDINGS_MODEL` | `text-embedding-3-small` | Embedding model used during indexing |
| `RETRIEVAL_TOP_K` | `4` | Number of chunks retrieved per query |
| `CONFIDENCE_THRESHOLD` | `0.7` | Minimum confidence before asking clarifying question |
| `DATA_DIR` | `/app/data/raw` (container) | Directory containing raw, unprocessed documents |
| `PROCESSED_DATA_DIR` | `/app/data/processed` (container) | Directory where cleaned & chunked JSONL is stored |
| `VECTORSTORE_DIR` | `/app/vectorstore/faiss_index` (container) | Override FAISS storage path |
| `PRIMARY_LANGUAGE` | – | Optional ISO language code used to filter documents during preprocessing |
| `HOST` | `0.0.0.0` | Bind address for local uvicorn |
| `PORT` | `8000` | Bind port for local uvicorn |
| `APP_PORT` | `8080` | Host port published by Docker Compose |
| `UVICORN_WORKERS` | `4` (compose) | Number of uvicorn worker processes in Docker |

## Project structure
- `src/` – FastAPI backend, LangChain graph, retrieval utilities
- `vectorstore/` – Default FAISS index location (mirrored in Docker volume)
- `data/raw/` – Raw source documents ready for preprocessing
- `data/processed/` – Cleaned, chunked JSONL data ready for embedding
- `scripts/` – Shell entrypoints such as `clean_data.sh`, `index_documents.sh`, and `run_validation.sh`
- `benchmark/` – Evaluation assets and tooling

## Extending
- Swap embeddings or LLM providers by editing `src/config.py` and the pipeline constructors under `src/services`.
- Add new document loaders under `src/services/indexing/loaders/`.

-## Indexing Workflow
- Place new source files under `data/raw/`.
- Generate cleaned & chunked JSONL: `bash scripts/clean_data.sh`.
- Build or refresh the FAISS index from processed data: `bash scripts/index_documents.sh --force` (optional `--force`).

## Benchmarking
- Prepare the validation dataset under `benchmark/data/` (default: `validation_data.json`).
- Run the cleaning pipeline to refresh processed data: `bash scripts/clean_data.sh`.
- Run asynchronous evaluation and store artefacts with `bash scripts/run_validation.sh`.
- Inspect per-question outputs in `benchmark/outputs/` and run summaries in `benchmark/reports/` for downstream analysis.
