import os
from typing import List
import pandas as pd
from langchain_core.documents import Document


def load_table_documents(data_dir: str) -> List[Document]:
    """Load table data from CSV and Excel files"""
    docs = []

    for root, _, files in os.walk(data_dir):
        for f in files:
            path = os.path.join(root, f)
            try:
                if f.lower().endswith(('.csv', '.xlsx', '.xls')):
                    # Load table data
                    if f.lower().endswith('.csv'):
                        df = pd.read_csv(path)
                    else:
                        df = pd.read_excel(path)

                    # Convert to text format
                    table_text = f"File: {f}\n\n{df.to_string(index=False)}\n\n"

                    # Add column information
                    columns_info = f"Columns: {', '.join(df.columns.tolist())}\n"
                    table_text = columns_info + table_text

                    doc = Document(
                        page_content=table_text,
                        metadata={"source": path, "type": "table", "filename": f}
                    )
                    docs.append(doc)

            except Exception as e:
                print(f"Error loading table file {path}: {e}")
                continue

    return docs
