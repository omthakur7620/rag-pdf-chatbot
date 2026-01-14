# app/chunker.py

from typing import List


def is_section_header(text: str) -> bool:
    """
    Detect section headers like:
    - WHAT IS AGENTIC AI
    - INTRODUCTION TO AGENTIC AI
    """
    return (
        text.isupper()
        and len(text.split()) <= 10
    )


def chunk_text(
    text: str,
    chunk_size: int = 900,
    chunk_overlap: int = 200
) -> List[str]:
    """
    Chunk text intelligently for ebooks:
    - Preserve headers + explanations
    - Keep tables and bullet lists intact
    """

    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks = []

    current_chunk = ""
    last_header = ""

    for para in paragraphs:

        # Detect section headers
        if is_section_header(para):
            last_header = para
            continue

        # Start chunk with header if needed
        if not current_chunk and last_header:
            current_chunk = last_header + "\n\n" + para
        else:
            current_chunk += "\n\n" + para

        # Flush chunk if size exceeded
        if len(current_chunk) >= chunk_size:
            chunks.append(current_chunk.strip())

            # Overlap handling (paragraph-level)
            overlap_text = current_chunk[-chunk_overlap:]
            current_chunk = overlap_text

    # Add final chunk
    if current_chunk.strip():
        chunks.append(current_chunk.strip())

    return chunks
