# WebRAG – AI-Powered Webpage Question Answering System

WebRAG is an AI-powered Retrieval-Augmented Generation (RAG) application that allows users to provide a webpage URL and ask questions about its content.

The system fetches the webpage, extracts and chunks its content, generates semantic embeddings, stores them in FAISS, combines dense retrieval with BM25 keyword retrieval, and uses an LLM to generate grounded answers.

## 🚀 Live Demo

**Frontend:**  
https://webrag-iqd6.onrender.com

**Backend API:**  
https://webrag-backend-ncmv.onrender.com

> The frontend communicates with the deployed FastAPI backend.

---

## ✨ Features

- 🌐 Ingest webpages using a URL
- 📄 Automatic HTML content extraction using Trafilatura
- ✂️ Intelligent document chunking
- 🔎 Hybrid retrieval using:
  - FAISS semantic/vector search
  - BM25 keyword search
  - Reciprocal Rank Fusion (RRF)
- 🧠 Hugging Face embeddings using:
  - `sentence-transformers/all-MiniLM-L6-v2`
  - 384-dimensional embeddings
- 🤖 LLM-based answer generation using Groq
- 💬 Conversational question answering
- 🔄 Query rewriting and expansion
- 🎯 Confidence scoring
- 🛡️ Protection against unsupported/out-of-context questions
- 💾 PostgreSQL for persistent document and conversation data
- ♻️ Automatic FAISS index recovery when the deployment filesystem is reset
- 🔐 CORS configuration for deployed frontend/backend
- ⚡ Background document ingestion
- 🧪 Comprehensive automated test suite

---

## 🏗️ System Architecture

```text
                    ┌─────────────────────┐
                    │     React Frontend  │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │    FastAPI Backend  │
                    └──────────┬──────────┘
                               │
                 ┌─────────────┼─────────────┐
                 ▼             ▼             ▼
          ┌────────────┐ ┌───────────┐ ┌────────────┐
          │ PostgreSQL │ │  Web      │ │   Groq     │
          │            │ │ Ingestion │ │    LLM     │
          └────────────┘ └─────┬─────┘ └────────────┘
                               │
                               ▼
                       ┌──────────────┐
                       │ Trafilatura  │
                       └──────┬───────┘
                              │
                              ▼
                       ┌──────────────┐
                       │  Chunking    │
                       └──────┬───────┘
                              │
                    ┌─────────┴─────────┐
                    ▼                   ▼
             ┌────────────┐      ┌────────────┐
             │ HuggingFace│      │    BM25    │
             │ Embeddings │      │  Retrieval │
             └─────┬──────┘      └─────┬──────┘
                   │                   │
                   ▼                   ▼
             ┌────────────┐      ┌────────────┐
             │   FAISS    │      │   Sparse   │
             │   Search   │      │   Search   │
             └─────┬──────┘      └─────┬──────┘
                   │                   │
                   └─────────┬─────────┘
                             ▼
                      Hybrid RRF Fusion
                             │
                             ▼
                         Groq LLM
                             │
                             ▼
                           Answer
