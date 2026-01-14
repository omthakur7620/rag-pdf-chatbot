# app/vector_store.py

from typing import List, Tuple
import os
from pinecone import Pinecone, ServerlessSpec
from app.embeddings import EmbeddingModel


class PineconeVectorStore:
    """
    Vector store wrapper for Pinecone with retrieval tuning
    optimized for ebook-style PDFs.
    """

    def __init__(
        self,
        index_name: str,
        embedding_model: EmbeddingModel,
        dimension: int = 384,
        metric: str = "cosine"
    ):
        self.embedder = embedding_model
        self.pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
        self.index_name = index_name

        # Create index if not exists
        existing_indexes = [i["name"] for i in self.pc.list_indexes()]

        if index_name not in existing_indexes:
            self.pc.create_index(
                name=index_name,
                dimension=dimension,
                metric=metric,
                spec=ServerlessSpec(
                    cloud="aws",
                    region="us-east-1"
                )
            )

        self.index = self.pc.Index(index_name)

    # -------------------------------
    # Store document chunks
    # -------------------------------
    def store_chunks(self, chunks: List[str]) -> None:
        embeddings = self.embedder.embed_texts(chunks)

        vectors = [
            {
                "id": f"chunk-{i}",
                "values": embedding,
                "metadata": {"text": chunk}
            }
            for i, (chunk, embedding) in enumerate(zip(chunks, embeddings))
        ]

        if vectors:
            self.index.upsert(vectors=vectors)

    # -------------------------------
    # Retrieve relevant chunks
    # -------------------------------
    def retrieve_chunks(
        self,
        query: str,
        top_k: int = 6,
        score_threshold: float = 0.55
    ) -> Tuple[List[str], List[float]]:

        query_embedding = self.embedder.embed_query(query)

        response = self.index.query(
            vector=query_embedding,
            top_k=top_k,
            include_metadata=True
        )

        contexts = []
        scores = []

        for match in response["matches"]:
            score = match["score"]

            # Filter weak semantic matches
            if score >= score_threshold:
                contexts.append(match["metadata"]["text"])
                scores.append(score)

        return contexts, scores
