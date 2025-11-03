from langchain_community.vectorstores import FAISS


def create_retriever(vectorstore: FAISS, k: int):
    """Create retriever from vectorstore with specified number of results"""
    return vectorstore.as_retriever(search_kwargs={"k": k})
