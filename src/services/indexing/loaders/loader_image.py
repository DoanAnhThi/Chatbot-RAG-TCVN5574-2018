import os
from typing import List
import easyocr
from PIL import Image
from langchain_core.documents import Document


def load_image_documents(data_dir: str) -> List[Document]:
    """Load image data using OCR"""
    docs = []

    # Initialize OCR reader
    reader = easyocr.Reader(['en', 'vi'])  # Support English and Vietnamese

    for root, _, files in os.walk(data_dir):
        for f in files:
            path = os.path.join(root, f)
            try:
                if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.webp')):
                    # Load image
                    image = Image.open(path)

                    # Extract text using OCR
                    ocr_results = reader.readtext(path)
                    extracted_text = " ".join([text for (_, text, confidence) in ocr_results if confidence > 0.5])

                    # Create document content
                    content = f"Image file: {f}\n\nExtracted text:\n{extracted_text}"

                    # Add basic image metadata
                    metadata = {
                        "source": path,
                        "type": "image",
                        "filename": f,
                        "image_size": image.size,
                        "image_mode": image.mode
                    }

                    doc = Document(
                        page_content=content,
                        metadata=metadata
                    )
                    docs.append(doc)

            except Exception as e:
                print(f"Error loading image file {path}: {e}")
                continue

    return docs
