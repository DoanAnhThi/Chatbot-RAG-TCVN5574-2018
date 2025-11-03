from typing import Dict, Any

from app.services.pipelines.generation.answer_generator import generate_answer


def build_generation_graph(retriever):
    """Build a simple graph for question answering pipeline"""

    def process_question(question: str) -> Dict[str, Any]:
        # Retrieve relevant documents
        docs = retriever.get_relevant_documents(question)

        # Generate answer
        return generate_answer(question, docs)

    class SimpleGraph:
        def __init__(self, processor):
            self.processor = processor

        def invoke(self, state):
            result = self.processor(state["question"])
            return {
                "question": state["question"],
                "answer": result["answer"],
                "confidence": result["confidence"],
                "needs_clarification": result["needs_clarification"]
            }

    return SimpleGraph(process_question)
