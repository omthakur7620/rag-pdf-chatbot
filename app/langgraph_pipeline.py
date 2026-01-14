from typing import TypedDict, List
from langgraph.graph import StateGraph, END

from app.vector_store import PineconeVectorStore
from app.embeddings import EmbeddingModel
from app.llm import GroqLLM
from app.utils import calculate_confidence


# Define Graph State
class RAGState(TypedDict):
    query: str
    contexts: List[str]
    scores: List[float]
    answer: str
    confidence: float


# Nodes

def retrieve_node(state: RAGState) -> RAGState:
    embedder = EmbeddingModel()
    store = PineconeVectorStore(
        index_name="rag-pdf",
        embedding_model=embedder
    )

    contexts, scores = store.retrieve_chunks(state["query"])

    return {
        **state,
        "contexts": contexts,
        "scores": scores
    }


def generate_node(state: RAGState) -> RAGState:
    llm = GroqLLM()

    answer = llm.generate_answer(
        query=state["query"],
        contexts=state["contexts"]
    )

    confidence = calculate_confidence(state["scores"])

    return {
        **state,
        "answer": answer,
        "confidence": confidence
    }


# Build Graph

def build_rag_graph():
    graph = StateGraph(RAGState)

    graph.add_node("retrieve", retrieve_node)
    graph.add_node("generate", generate_node)

    graph.set_entry_point("retrieve")
    graph.add_edge("retrieve", "generate")
    graph.add_edge("generate", END)

    return graph.compile()
