# streamlit_app.py

from dotenv import load_dotenv
load_dotenv()

import streamlit as st
from app.langgraph_pipeline import build_rag_graph

# -----------------------------
# Page Configuration
# -----------------------------
st.set_page_config(
    page_title="Agentic AI – PDF RAG Chatbot",
    layout="centered"
)

st.title("📘 Agentic AI – PDF RAG Chatbot")
st.caption(
    "Chat with an AI assistant that answers **strictly from the Ebook-Agentic-AI.pdf** using RAG + LangGraph."
)

# -----------------------------
# Load LangGraph Pipeline
# -----------------------------
@st.cache_resource
def load_graph():
    return build_rag_graph()

rag_graph = load_graph()

# -----------------------------
# Initialize Chat History
# -----------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

# -----------------------------
# Display Chat History
# -----------------------------
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

        # Show confidence & context only for assistant messages
        if msg["role"] == "assistant":
            st.markdown(f"**Confidence:** `{msg['confidence']}`")

            with st.expander("📚 Retrieved Context Chunks"):
                for i, ctx in enumerate(msg["contexts"], start=1):
                    st.markdown(f"**Chunk {i}:**")
                    st.write(ctx)

# -----------------------------
# Chat Input
# -----------------------------
user_query = st.chat_input("Ask a question from the Agentic AI ebook...")

if user_query:
    # Save user message
    st.session_state.messages.append({
        "role": "user",
        "content": user_query
    })

    with st.chat_message("user"):
        st.markdown(user_query)

    # Run RAG pipeline
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            result = rag_graph.invoke({"query": user_query})

            answer = result.get("answer", "")
            confidence = result.get("confidence", 0.0)
            contexts = result.get("contexts", [])

            st.markdown(answer)
            st.markdown(f"**Confidence:** `{confidence}`")

            with st.expander("📚 Retrieved Context Chunks"):
                for i, ctx in enumerate(contexts, start=1):
                    st.markdown(f"**Chunk {i}:**")
                    st.write(ctx)

    # Save assistant message
    st.session_state.messages.append({
        "role": "assistant",
        "content": answer,
        "confidence": confidence,
        "contexts": contexts
    })
