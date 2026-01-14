# app/embeddings.py

from typing import List
from sentence_transformers import SentenceTransformer


class EmbeddingModel:
    """
    Wrapper class for embedding generation.
    Keeps model loading clean and reusable.
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for a list of texts.

        Args:
            texts (List[str]): Input text chunks

        Returns:
            List[List[float]]: Embedding vectors
        """

        if not texts:
            return []

        embeddings = self.model.encode(
            texts,
            show_progress_bar=False,
            convert_to_numpy=True
        )

        return embeddings.tolist()

    def embed_query(self, query: str) -> List[float]:
        """
        Generate embedding for a single query.

        Args:
            query (str): User query

        Returns:
            List[float]: Query embedding vector
        """

        if not query or not query.strip():
            raise ValueError("Query cannot be empty")

        embedding = self.model.encode(
            query,
            show_progress_bar=False,
            convert_to_numpy=True
        )

        return embedding.tolist()
