# app/rag_pipeline.py

from typing import Dict

from app.embeddings import EmbeddingModel
from app.vector_store import PineconeVectorStore
from app.llm import GroqLLM
from app.utils import calculate_confidence


class RAGPipeline:
    """
    Core RAG pipeline that connects retrieval and generation.
    """

    def __init__(
        self,
        index_name: str,
        top_k: int = 3
    ):
        self.embedder = EmbeddingModel()
        self.vector_store = PineconeVectorStore(
            index_name=index_name,
            embedding_model=self.embedder
        )
        self.llm = GroqLLM()
        self.top_k = top_k

    def run(self, query: str) -> Dict:
        """
        Run the RAG pipeline for a user query.

        Args:
            query (str): User question

        Returns:
            Dict: {
                "answer": str,
                "contexts": List[str],
                "confidence": float
            }
        """

        # 1️⃣ Retrieve relevant chunks
        contexts, scores = self.vector_store.retrieve_chunks(
            query=query,
            top_k=self.top_k
        )

        # 2️⃣ Generate grounded answer
        answer = self.llm.generate_answer(
            query=query,
            contexts=contexts
        )

        # 3️⃣ Calculate confidence
        confidence = calculate_confidence(scores)

        return {
            "answer": answer,
            "contexts": contexts,
            "confidence": confidence
        }
