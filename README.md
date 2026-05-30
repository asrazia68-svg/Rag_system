# 📝 DocuQuery AI — RAG System

> An intelligent document question-answering system powered by Retrieval-Augmented Generation (RAG). Upload your PDF or TXT files and ask questions — the AI retrieves the most relevant context and gives you precise answers.

---

## 🚀 Project Overview

**DocuQuery AI** is a full-stack RAG (Retrieval-Augmented Generation) application that lets you have a conversation with your documents. It combines the power of vector embeddings, ChromaDB, and the Groq LLaMA 3.3 model to deliver accurate, context-aware answers from your own uploaded files.

The system has two main components:

- **Ingestion Pipeline** — Loads, chunks, and vectorizes documents into ChromaDB for persistent storage.
- **Streamlit Chat App** — A real-time chat interface where you upload documents and ask questions interactively.

---

## ✨ Features

- 📄 **Multi-format Support** — Accepts both PDF and TXT documents
- 🔍 **Semantic Search** — Uses cosine similarity on HuggingFace embeddings to find the most relevant chunks
- 🤖 **LLaMA 3.3 70B via Groq** — Fast, accurate answers powered by one of the most capable open models
- 🗄️ **ChromaDB Vector Store** — Persistent and in-memory vector storage with cosine similarity
- 💬 **Conversational Memory** — Maintains full chat history across turns within a session
- 🎨 **Dark Mode UI** — Sleek, modern dark-themed Streamlit interface
- 🔄 **Clear & Reset** — One-click history and vector store reset

---

## 🛠️ Tech Stack

| Component | Technology |
|---|---|
| Frontend | Streamlit |
| LLM | Groq — LLaMA 3.3 70B Versatile |
| Embeddings | HuggingFace — `all-MiniLM-L6-v2` |
| Vector Store | ChromaDB (cosine similarity) |
| PDF Parsing | pypdf |
| Orchestration | LangChain |
| Language | Python |


## ▶️ Running the App

The app opens at: https://ragsystem-3wlqdclw44x6scnhosnqif.streamlit.app/



## 🔄 How It Works

```
User uploads document
        ↓
Text extracted (PDF → pypdf, TXT → raw read)
        ↓
Text split into chunks (size: 600, overlap: 60)
        ↓
Chunks embedded via HuggingFace all-MiniLM-L6-v2
        ↓
Embeddings stored in ChromaDB (cosine similarity)
        ↓
User asks a question
        ↓
Top 4 relevant chunks retrieved from ChromaDB
        ↓
Context + question sent to LLaMA 3.3 via Groq
        ↓
Answer displayed in chat
```


<div align="center">
  Made with ❤️ using Python, LangChain & Groq
</div>
