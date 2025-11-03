import os
from typing import List
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document


def load_pdf_documents(data_dir: str) -> List[Document]:
    """Load PDF documents"""
    docs: List[Document] = []

    # Load PDFs
    for root, _, files in os.walk(data_dir):
        for f in files:
            path = os.path.join(root, f)
            if f.lower().endswith(".pdf"):
                try:
                    loader = PyPDFLoader(path)
                    docs.extend(loader.load())
                except Exception as e:
                    print(f"Error loading PDF file {path}: {e}")
                    continue

    return docs
