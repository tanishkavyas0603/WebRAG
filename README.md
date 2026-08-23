# WebRAG - Chat with Any Webpage 🌐🤖

WebRAG is a full-stack, AI-powered application that allows users to seamlessly extract, index, and chat with the contents of any public webpage. Provide a URL, and WebRAG will ingest the content, build a hybrid search index, and provide precise, grounded answers backed by specific citations.

## 🚀 Key Features

*   **Intelligent Web Scraping:** Extracts clean content from any URL while stripping out navigation, footers, and boilerplate HTML (powered by `trafilatura` and `BeautifulSoup`).
*   **SSRF Protection:** Strict security layer preventing malicious requests to local/private networks or non-HTTP endpoints.
*   **Advanced Hybrid RAG Pipeline:** Combines sparse (BM25) and dense (FAISS + SentenceTransformers) retrievals using Reciprocal Rank Fusion (RRF) for highly accurate context matching.
*   **LLM Integration:** Connects to powerful Groq LLMs for blazing-fast inference.
*   **Grounded Citations:** LLM answers are strictly grounded in the ingested text, automatically generating citations linked to the original document segments.
*   **Multi-Document Management:** Users can ingest multiple URLs and maintain independent chat histories for each.
*   **Secure Authentication:** JWT-based user authentication ensures that documents and conversations remain completely private and isolated.

## 🏗 Architecture & Tech Stack

WebRAG uses a modern, scalable architecture designed for easy transition from local development to production.

*   **Backend Framework:** FastAPI (Python 3.12)
*   **Frontend Framework:** React + Vite (Tailwind CSS)
*   **Database:** SQLite (Local) / PostgreSQL + Supabase (Production)
*   **ORM / Migrations:** SQLAlchemy + Alembic
*   **Embeddings Model:** `sentence-transformers/all-MiniLM-L6-v2`
*   **Vector Database (Local):** FAISS (Facebook AI Similarity Search)
*   **Sparse Retrieval:** BM25 (via `rank-bm25`)
*   **LLM Provider:** Groq (e.g., `llama-3.3-70b-versatile`)

## 🧠 The RAG Pipeline

1.  **Ingestion & Extraction:** The backend fetches the webpage and extracts meaningful text paragraphs, discarding HTML noise.
2.  **Chunking:** The text is split into overlapping chunks to preserve context across boundaries.
3.  **Indexing:** 
    *   *Dense:* Chunks are embedded using `SentenceTransformers` and indexed in FAISS.
    *   *Sparse:* Chunks are tokenized and indexed using BM25.
4.  **Retrieval (RRF):** When a user asks a question, WebRAG queries both FAISS and BM25. The results are merged using Reciprocal Rank Fusion (RRF) and optimized using Maximal Marginal Relevance (MMR) for diversity.
5.  **Generation:** The top chunks are formatted into a prompt containing the conversation history and sent to the Groq LLM to generate a cited response.

## 🛠 Local Setup & Development

### 1. Environment Variables

Create a `.env` file in the root directory (use `.env.example` as a template):

```ini
# Database (Defaults to local SQLite)
DATABASE_URL=sqlite:///./webrag.db

# Security
SECRET_KEY=generate_a_secure_random_string_here
CORS_ORIGINS=http://localhost:5173

# Storage Paths
DATA_PATH=data
HF_HOME=.cache/huggingface

# LLM Provider
GROQ_API_KEY=your_groq_api_key_here
```

### 2. Backend Setup

```bash
# Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run database migrations (creates SQLite tables locally)
alembic upgrade head

# Start the FastAPI server
uvicorn app.main:app --reload
```
The backend API will be available at `http://localhost:8000`.

### 3. Frontend Setup

In a new terminal, create a `.env` file in the `frontend/` directory:
```ini
VITE_API_URL=http://localhost:8000
```

```bash
cd frontend
npm install
npm run dev
```
The application UI will be available at `http://localhost:5173`.

### 4. Running Tests

WebRAG includes a comprehensive pytest suite covering authentication, database schema, security (SSRF), RAG components, and end-to-end extraction logic.

```bash
python -m pytest
```

## 🌍 Production Deployment

This repository is configured for production deployment using **Vercel** (Frontend), **Render** (Backend), and **Supabase** (PostgreSQL Database).

**Key Differences for Production:**
1.  **Database:** Set `DATABASE_URL` to a Supabase Postgres connection string. SQLAlchemy automatically handles the dialect shift from SQLite to Postgres.
2.  **Storage:** You MUST mount a persistent disk (e.g., a Render Disk) and point `DATA_PATH` and `HF_HOME` to this persistent volume to prevent data loss (FAISS indices and HuggingFace models) upon server restarts.
3.  **CORS:** Ensure `CORS_ORIGINS` strictly specifies your production Vercel domain.

*For exact deployment steps, refer to `DEPLOYMENT.md`.*

---
*Built as a professional portfolio project demonstrating applied AI, Retrieval-Augmented Generation, and full-stack system design.*
