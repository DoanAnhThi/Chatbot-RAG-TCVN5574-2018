from langchain_openai import OpenAIEmbeddings

from app.config import settings


def create_embeddings_model() -> OpenAIEmbeddings:
    """Create OpenAI embeddings model"""
    return OpenAIEmbeddings(model=settings.embeddings_model, api_key=settings.openai_api_key)
