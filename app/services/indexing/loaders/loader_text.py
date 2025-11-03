from typing import List
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_core.documents import Document


def load_text_documents(data_dir: str) -> List[Document]:
    """Load text documents from TXT and MD files, excluding table and image files"""
    docs: List[Document] = []

    # Load text files (TXT / MD) - exclude table and image files
    loader = DirectoryLoader(
        data_dir,
        glob="**/*",
        loader_cls=TextLoader,
        show_progress=True,
        silent_errors=True,
        exclude=["*.csv", "*.xlsx", "*.xls", "*.png", "*.jpg", "*.jpeg", "*.bmp", "*.tiff", "*.webp"]
    )
    try:
        docs.extend(loader.load())
    except Exception as e:
        print(f"Error loading text documents: {e}")
        pass

    return docs
