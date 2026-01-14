# app/pdf_loader.py

from typing import List
import fitz  # PyMuPDF
import re


def clean_text(text: str) -> str:
    """
    Clean PDF artifacts without destroying meaning.
    """
    # Remove excessive whitespace
    text = re.sub(r"\s+", " ", text)

    # Remove standalone page numbers
    text = re.sub(r"\b\d{1,3}\b", "", text)

    return text.strip()


def load_pdf(pdf_path: str) -> str:
    """
    Load PDF and return cleaned, continuous text.
    Designed specifically for ebooks with sections.
    """

    doc = fitz.open(pdf_path)
    pages_text: List[str] = []

    for page in doc:
        text = page.get_text("text")

        # Skip empty pages
        if not text or len(text.strip()) < 50:
            continue

        cleaned = clean_text(text)
        pages_text.append(cleaned)

    full_text = "\n\n".join(pages_text)

    return full_text
