from .retrieval_logic import create_retriever
from .formatter import format_documents
from .cosine import cosine_similarity, find_top_similar_vectors, batch_cosine_similarity
from .ranking import rank_documents_by_similarity, rerank_with_diversity, filter_documents_by_threshold

# Re-export for backward compatibility
def get_retriever(vectorstore, k: int):
    return create_retriever(vectorstore, k)

def format_docs(docs):
    return format_documents(docs)

# Export utilities for advanced retrieval
__all__ = [
    'get_retriever',
    'format_docs',
    'cosine_similarity',
    'find_top_similar_vectors',
    'batch_cosine_similarity',
    'rank_documents_by_similarity',
    'rerank_with_diversity',
    'filter_documents_by_threshold'
]
