# scripts/index_pdf.py

from dotenv import load_dotenv
load_dotenv()
from app.pdf_loader import load_pdf
from app.chunker import chunk_text
from app.embeddings import EmbeddingModel
from app.vector_store import PineconeVectorStore


PDF_PATH = "data/Ebook-Agentic-AI.pdf"
INDEX_NAME = "rag-pdf"


def main():
    print("📄 Loading PDF...")
    text = load_pdf(PDF_PATH)

    print("✂️ Chunking text...")
    chunks = chunk_text(text)

    print(f"🔢 Total chunks created: {len(chunks)}")

    embedder = EmbeddingModel()

    vector_store = PineconeVectorStore(
        index_name=INDEX_NAME,
        embedding_model=embedder
    )

    print("📥 Storing chunks in Pinecone...")
    vector_store.store_chunks(chunks)

    print("✅ Ebook-Agentic-AI.pdf indexed successfully.")


if __name__ == "__main__":
    main()
