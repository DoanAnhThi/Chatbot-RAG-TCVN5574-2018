import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from app.services.indexing.indexer import build_or_load_vectorstore

if __name__ == "__main__":
    vs = build_or_load_vectorstore()
    print("Vector store ready at persistent path.")
