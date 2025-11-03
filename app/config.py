from pathlib import Path
import os

from pydantic import BaseModel


PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _default_path(env_name: str, container_default: str, local_relative: str) -> str:
    env_value = os.getenv(env_name)
    if env_value:
        return env_value

    container_path = Path(container_default)
    try:
        container_path.mkdir(parents=True, exist_ok=True)
        return container_default
    except OSError:
        local_path = PROJECT_ROOT / local_relative
        local_path.mkdir(parents=True, exist_ok=True)
        return str(local_path)


class Settings(BaseModel):
    openai_api_key: str | None = os.getenv("OPENAI_API_KEY")
    openai_model: str = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    embeddings_model: str = os.getenv("EMBEDDINGS_MODEL", "text-embedding-3-small")

    data_dir: str = _default_path("DATA_DIR", "/app/data", "data")
    vectorstore_dir: str = _default_path(
        "VECTORSTORE_DIR", "/app/vectorstore/faiss_index", "app/vectorstore/faiss_index"
    )

    retrieval_top_k: int = int(os.getenv("RETRIEVAL_TOP_K", "4"))
    confidence_threshold: float = float(os.getenv("CONFIDENCE_THRESHOLD", "0.7"))

    host: str = os.getenv("HOST", "0.0.0.0")
    port: int = int(os.getenv("PORT", "8000"))


settings = Settings()
