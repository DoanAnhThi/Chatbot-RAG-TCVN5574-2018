from langchain_openai import ChatOpenAI

from app.config import settings


def create_llm_model():
    """Create ChatOpenAI LLM model for answer generation"""
    return ChatOpenAI(model=settings.openai_model, temperature=0.2, api_key=settings.openai_api_key)
