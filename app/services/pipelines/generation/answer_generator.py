from typing import Dict, Any, List

from langchain_core.documents import Document

from .llm_model import create_llm_model
from .prompt_builder import create_answer_prompt, calculate_confidence


def generate_answer(question: str, docs: List[Document]) -> Dict[str, Any]:
    """Generate answer from question and retrieved documents"""
    # Create LLM and prompt
    llm = create_llm_model()
    prompt = create_answer_prompt()

    # Format context from documents
    from ..retrieval.formatter import format_documents
    context = format_documents(docs)

    # Generate answer
    msg = prompt.format_messages(question=question, context=context)
    resp = llm.invoke(msg)

    # Calculate confidence
    conf = calculate_confidence(len(docs))

    return {
        "answer": resp.content,
        "confidence": conf,
        "needs_clarification": False
    }
