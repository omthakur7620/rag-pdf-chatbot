# rag-pdf-chatbot
# 📘 Agentic AI – PDF RAG Chatbot

A **Retrieval-Augmented Generation (RAG) based AI Chatbot** built in Python that answers questions **strictly grounded** in the **Agentic AI eBook**.  
The system uses **LangGraph** for orchestration, **Pinecone** as the vector database, **text embeddings** for semantic retrieval, and a **chat-style Streamlit UI**.

---

## 🚀 Features

- 📄 PDF-based knowledge base (Agentic AI eBook)
- 🔍 Semantic search using text embeddings
- 🧠 RAG pipeline with strict grounding (no hallucinations)
- 🧩 LangGraph-based orchestration (retrieve → generate)
- 💬 Chat-style Streamlit chatbot UI
- 📚 Retrieved context display for transparency
- 📊 Confidence score based on retrieval similarity

---

## 🏗️ Architecture Overview

User (Streamlit Chat UI)
↓
LangGraph State Graph
↓
Retrieve Relevant Chunks
↓
Generate Answer (LLM)
↓
Answer + Context + Confidence


---

## 📂 Project Structure

rag-pdf-chatbot/
│
├── app/
│ ├── pdf_loader.py
│ ├── chunker.py
│ ├── embeddings.py
│ ├── vector_store.py
│ ├── llm.py
│ ├── utils.py
│ ├── rag_pipeline.py
│ └── langgraph_pipeline.py
│
├── scripts/
│ ├── index_pdf.py
│ └── list_groq_models.py
│
├── data/
│ └── Ebook-Agentic-AI.pdf
│
├── streamlit_app.py
├── requirements.txt
├── README.md
└── .env


---

## ⚙️ Setup Instructions
Python 3.13.5

### 1️⃣ Clone the Repository

```bash
git clone <your-github-repo-url>
cd rag-pdf-chatbot

2️⃣ Create Virtual Environment
python -m venv venv
venv\Scripts\activate

pip install -r requirements.txt

4️⃣ Environment Variables
PINECONE_API_KEY=your_pinecone_api_key
GROQ_API_KEY=your_groq_api_key


Index the PDF (One-Time Step)
python -m scripts.index_pdf


Run the Chatbot
streamlit run streamlit_app.py


----Sample Questions------

Use the following questions to test the chatbot:
What is Agentic AI?
How does Agentic AI differ from traditional AI systems?
Why is Agentic AI considered a shift from reactive to proactive AI?
What are the key components of an agentic system?
What role do tools play in Agentic AI?
What are real-world use cases of Agentic AI?


LangGraph Usage
LangGraph is used to orchestrate the RAG workflow as a state-driven graph:
Retrieve node: fetches relevant chunks from Pinecone
Generate node: produces grounded answers using the LLM
End state: returns answer, context, and confidence
This makes the pipeline explicit, traceable, and extensible.


📊 Confidence Score
The confidence score is calculated using the average similarity score of retrieved chunks.
It indicates how strongly the answer is grounded in the source document.


✅ Key Design Decisions
Section-aware chunking for ebook-style PDFs
Tuned retrieval using top-k and similarity thresholds
Balanced grounding prompt to avoid false negatives
Chat-style UI for a true chatbot experience
Clear separation between ingestion, retrieval, and UI layers


🧾 Notes
PDF ingestion is performed once; querying is fast afterward
The chatbot never uses external knowledge
All answers are derived strictly from the provided ebook