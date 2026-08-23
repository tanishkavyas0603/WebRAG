# WEBRAG_PROJECT_GUIDE.md

## 1. Executive Summary & Project Overview
**What problem does WebRAG solve?**
WebRAG solves the problem of hallucination in Large Language Models (LLMs) by grounding their answers in external, verifiable knowledge. Users can input a URL, and the system downloads, extracts, and indexes the webpage text. When a user asks a question, the system searches the indexed document, retrieves the most relevant paragraphs, and forces the LLM to answer *only* using that context. 

**How does the user use it?**
1. User logs in.
2. User enters a URL (e.g., `https://example.com`) to "ingest" it.
3. The system processes the page in the background (extracts text, chunks it, embeds it).
4. User selects the ingested document to start a conversation.
5. User asks questions, and the system answers with citations pointing directly to the source text.

## 2. Architecture & Actual Folder Structure
### Actual Folder Structure
Based on the current repository:
```
WebRAG/
│
├── app/                      # Backend Root
│   ├── main.py               # FastAPI entry point
│   ├── api/                  # API routes (auth, documents, conversations, health)
│   ├── core/                 # Config, database setup, JWT security, logging
│   ├── models/               # SQLAlchemy DB models & Pydantic schemas
│   ├── services/             # Core logic (RAG, ingestion, chunking, embeddings, BM25)
│   └── vectorstore/          # FAISS index storage and retrieval
│
├── frontend/                 # Frontend Root (React/Vite)
│   ├── package.json          
│   └── src/
│       ├── api/              # Axios client setup (client.js, auth.js, etc.)
│       ├── components/       # UI Components (Sidebar, DocumentCard)
│       ├── context/          # React Context (AuthContext)
│       └── pages/            # Page Views (Dashboard, Chat, Login, Register)
│
├── tests/                    # 60+ Pytest verification tests
├── alembic/                  # Database migration scripts
├── alembic.ini               # Alembic configuration
├── data/                     # Local storage for FAISS indexes and BM25 pickles
├── requirements.txt          # Python dependencies
├── pytest.ini                # Pytest configuration
├── docker-compose.yml        # Docker setup
└── .env                      # Environment variables
```

### Important Files Overview
- `app/main.py`: **FastAPI Backend Entry Point**. Sets up CORS, connects routers, initializes the database. Required for the backend to run.
- `app/core/config.py`: **Configuration**. Centralized environment variable parsing. Used by all services.
- `app/models/db.py`: **Database schema**. SQLAlchemy models for User, Document, Chunk, Conversation, Message.
- `app/services/rag_service.py`: **RAG Pipeline**. The brain of the system. Handles query rewriting, retrieval, confidence calculation, and LLM prompting.
- `app/services/ingestion_service.py`: **Web Scraper**. Downloads and extracts text from URLs using `trafilatura` and `BeautifulSoup`.
- `app/services/chunking_service.py`: **Text Splitter**. Splits large text into smaller 200-word chunks.
- `app/api/documents.py`: **Document API**. Defines the `/api/documents/ingest` endpoint and orchestrates the background ingestion task.
- `frontend/src/api/client.js`: **Axios Instance**. Centralized frontend HTTP client with JWT interceptors.
- `frontend/src/pages/Chat.jsx`: **Chat UI**. Displays messages, citations, and handles the interactive conversation.

## 3. High Level Architecture Diagram
**The Actual Implemented Flow:**
```mermaid
graph TD
    User -->|React UI| Frontend
    Frontend -->|POST /api/documents/ingest| FastAPI_Auth_Documents
    FastAPI_Auth_Documents -->|Background Task| IngestionService
    IngestionService -->|Fetch HTML| Trafilatura
    Trafilatura --> ChunkingService
    ChunkingService --> EmbeddingService
    EmbeddingService --> FAISS_Index
    ChunkingService --> BM25_Index
    
    Frontend -->|POST /api/conversations/{id}/messages| RAGService
    RAGService -->|Query Expansion| Groq_LLM
    RAGService -->|Dense Search| FAISS_Index
    RAGService -->|Sparse Search| BM25_Index
    FAISS_Index --> RRF_Fusion
    BM25_Index --> RRF_Fusion
    RRF_Fusion --> MetadataRanker
    MetadataRanker --> ConfidenceService
    ConfidenceService --> PromptService
    PromptService --> Groq_LLM
    Groq_LLM --> Frontend
```

## 4. Complete User Flow
1. User opens frontend. React loads (Vite).
2. User registers/logs in (`/api/auth/register`, `/api/auth/login`).
3. FastAPI returns a JWT token. `AuthContext` stores it in `localStorage`.
4. User enters a URL on the Dashboard.
5. Frontend calls `POST /api/documents/ingest`.
6. FastAPI validates the URL, creates a `Document` with status `pending`, and queues `process_document_background`.
7. `IngestionService` fetches the URL, validates it isn't an SSRF attack, and extracts text.
8. `ChunkingService` breaks text into 200-word chunks.
9. `EmbeddingService` generates vectors for chunks.
10. `FAISSVectorStore` and `BM25Service` save indexes to disk in `.data/vectors/{doc_id}/`.
11. Document status becomes `ready`. Frontend detects this via polling `GET /api/documents/{id}/status`.
12. User clicks the document, frontend creates a Conversation (`POST /api/conversations`).
13. User asks a question (`POST /api/conversations/{id}/messages`).
14. `RAGService` rewrites the query if there is chat history.
15. `RetrievalService` searches FAISS (dense) and BM25 (sparse).
16. Results are fused using RRF (Reciprocal Rank Fusion).
17. `ConfidenceService` evaluates the top cosine score.
18. `PromptService` builds a strict prompt with the top chunks.
19. `Groq` LLM generates the answer.
20. `RAGService` formats citations and saves the `Message`.
21. Frontend renders the answer and citations.

## 5. Backend `main.py`
`app/main.py` is the root of the backend.
- **Imports:** Imports FastAPI, CORS middleware, database setup, and all API routers (auth, documents, conversations, health).
- **Database Initialization:** `Base.metadata.create_all(bind=engine)` creates SQLite tables if Alembic hasn't (though Alembic is preferred).
- **FastAPI Object (`app = FastAPI(...)`)**: Creates the server instance.
- **CORS Middleware:** `CORSMiddleware` allows the React frontend (running on a different port like 5173) to send requests to FastAPI (port 8000). Without this, browsers block cross-origin requests.
- **Routers (`app.include_router(...)`)**: Mounts the modular API files into the main app.
  - `/api/auth`: Login/Register
  - `/api/documents`: Ingestion
  - `/api/conversations`: Chat
  - `/api/health`: Health checks

## 6. Configuration (`app/core/config.py`)
Centralizes all settings from the `.env` file using Pydantic `BaseSettings`.
- **`DATABASE_URL`**: E.g., `sqlite:///./webrag.db`. Tells SQLAlchemy where the database is.
- **`GROQ_API_KEY`**: Authenticates with Groq. Used in `rag_service.py`. If wrong, LLM generation fails (HTTP 401). Never hardcoded to prevent leaking keys in git.
- **`JWT_SECRET`**: Used to sign and verify Auth tokens. If wrong or changed, all current users are logged out.
- **`WEBRAG_USER_AGENT`**: Used during web scraping to identify our bot.
- **`MAX_INGESTION_SIZE_BYTES`**: 10MB limit to prevent memory exhaustion from huge files.
- **`GROQ_MODEL`**: The specific LLM model (e.g., `llama-3.3-70b-versatile`).

## 7. Database Walkthrough
Models are in `app/models/db.py`.
- **User**: Stores users. Columns: `id` (PK), `email`, `hashed_password`. 
- **Document**: Represents an ingested webpage. Columns: `id` (PK), `user_id` (FK), `url`, `title`, `content`, `status`. Relationship: A User has many Documents.
- **Chunk**: A piece of a Document. Columns: `id` (PK), `document_id` (FK), `chunk_index`, `content`. Used for retrieval.
- **Conversation**: A chat thread. Columns: `id` (PK), `user_id` (FK), `document_id` (FK), `title`.
- **Message**: A single message in a chat. Columns: `id` (PK), `conversation_id` (FK), `role` (user/assistant), `content`, `citations` (JSON).

**ER Diagram:**
```text
User 
 ├── Documents ── Chunks
 └── Conversations ── Messages
```

## 8. Authentication Walkthrough
Files: `app/api/auth.py`, `app/core/security.py`.
- **Registration**: `POST /api/auth/register`. Receives email/password. Uses `bcrypt` to hash the password into a salted string. Saves to DB. Passwords are never stored as plain text to prevent leaks if the DB is compromised.
- **Login**: `POST /api/auth/login`. Looks up email. `bcrypt.checkpw()` verifies the password. Generates a JSON Web Token (JWT) containing `{"sub": user_id, "exp": expiration_time}`.
- **`get_current_user` Dependency**: FastAPI extracts the `Authorization: Bearer <token>` header, decodes it using `JWT_SECRET`, finds the user ID in the `sub` claim, and fetches the User from the DB. If invalid, throws 401.

## 9. Document Ingestion Walkthrough
File: `app/api/documents.py`.
1. **User Enters `https://example.com`**.
2. **`POST /api/documents/ingest`**: Validates the URL. Checks if the document already exists for this user (deduplication via URL hash).
3. **Database Creation**: Creates a `Document` with status `pending`.
4. **Background Task**: Returns the Document ID to the user immediately, and queues `process_document_background`.
5. **Fetching**: Status becomes `fetching`. `httpx` downloads the HTML. SSRF protections apply here.
6. **Extracting**: Status becomes `extracting`. Uses `trafilatura` (or `BeautifulSoup` fallback) to pull out clean text and strip sidebars/nav.
7. **Chunking**: Status becomes `chunking`. Breaks text into 200-word blocks.
8. **Indexing**: Status becomes `indexing`. Creates vectors and saves FAISS/BM25 to local disk.
9. **Ready**: Status becomes `ready`.

## 10. SSRF Security
**Server-Side Request Forgery (SSRF)** occurs when an attacker tricks the server into making HTTP requests to internal networks (like `127.0.0.1`, `localhost`, or AWS metadata `169.254.169.254`).
**How we protect it (`ingestion_service.py`):**
- **Scheme Validation**: Enforces `http` or `https` only (no `file://`).
- **IP Validation**: Resolves the hostname to an IP. Blocks private IP ranges (e.g., `10.0.0.0/8`, `192.168.0.0/16`, `127.0.0.0/8`).
- **Redirect Validation**: Uses an `httpx` event hook (`on_request`) to validate the IPs of *every* redirect hop, ensuring an attacker can't bypass checks by setting up a public URL that redirects to `localhost`.

## 11. HTML Extraction
Files: `app/services/ingestion_service.py`.
- **Why not just `BeautifulSoup(html).get_text()`?** Because this extracts all text, including hidden scripts, CSS, navigation menus, and footers. This pollutes the RAG context with noise (e.g., "Home Contact Us Login").
- **Current Solution**: 
  1. **Trafilatura**: An intelligent library designed to extract the *main body* of articles/webpages and strip boilerplate automatically.
  2. **Fallback**: If Trafilatura fails or returns <200 chars, it falls back to a heavily filtered `BeautifulSoup` script that explicitly removes `<script>`, `<style>`, `<nav>`, `<footer>`, etc.

## 12. Chunking
Files: `app/services/chunking_service.py`.
- **Why chunk?** LLMs have context limits, and embedding an entire webpage into a single vector dilutes the semantic meaning. Searching for specific facts requires smaller, focused vectors.
- **How it works**: Uses `re.split(r'\n+', text)` to split the document into paragraphs. Iterates through paragraphs, grouping them until they hit a `MAX_WORDS` limit (200 words). Saves each group as a `Chunk` in the database. 

## 13. Embeddings
Files: `app/services/embedding_service.py`.
- **What is an embedding?** It's a numerical representation of text semantics. Similar sentences end up close to each other in mathematical space.
- **Implementation**: Uses `sentence-transformers/all-MiniLM-L6-v2`. It takes a chunk of text and returns a vector of fixed dimension (384 dimensions for MiniLM).

## 14. FAISS (Dense Retrieval)
Files: `app/vectorstore/faiss_store.py`.
- **What is FAISS?** Facebook AI Similarity Search. A highly optimized library for finding the closest vectors.
- **How it works here**: We use `IndexFlatIP` (Inner Product). Since MiniLM normalizes vectors, Inner Product is equivalent to Cosine Similarity. 
- **Storage**: When a document is ingested, we build a FAISS index and save it to disk (`data/vectors/{doc_id}/faiss.index`). During chat, we load it into memory, embed the user's question, and use `faiss.search` to find the top K most similar chunks.

## 15. BM25 (Sparse Retrieval)
Files: `app/services/bm25_service.py`.
- **What is BM25?** A lexical (keyword-based) search algorithm. It matches exact words, scoring them based on term frequency and inverse document frequency (TF-IDF).
- **Why we need it**: Dense embeddings (FAISS) are great for *semantic* meaning ("How do I log in?" matches "Authentication steps"), but terrible at exact keywords (e.g., searching for a specific product ID or an acronym like "RRF"). BM25 catches exact keyword matches.

## 16. Hybrid Retrieval + RRF
Files: `app/services/retrieval_service.py`.
- **Hybrid Retrieval**: We run FAISS (gets top 15) and BM25 (gets top 15) in parallel.
- **RRF (Reciprocal Rank Fusion)**: A mathematical formula to combine the two ranked lists without needing to normalize their wildly different scores (FAISS uses cosine, BM25 uses absolute TF-IDF scores).
- **Formula Used**: `RRF_Score = 1 / (60 + Rank)`. If a chunk is rank 1 in FAISS and rank 5 in BM25, its score is `1/61 + 1/65`. The chunks are then sorted by this combined RRF score.

## 17. MMR / Ranking
Files: `app/services/retrieval_service.py` (`MetadataRanker`).
- **What it does**: Takes the RRF-fused list and ensures diversity and relevance. It checks if chunks overlap too heavily or boost chunks that have metadata matching the query. It returns the final ordered list of chunks to feed into the prompt.

## 18. Query Rewriting
Files: `app/services/rag_service.py` (`_rewrite_query`).
- **What it does**: Takes the user's latest question and the past 3 chat messages, and asks the LLM to rewrite it into a "standalone question".
- **Why**: If the user asks "What is FastAPI?", and then follows up with "Who created it?", the string "Who created it?" is useless for FAISS retrieval because "it" lacks context. The LLM rewrites it to "Who created FastAPI?" *before* we search the database.

## 19. Confidence System
Files: `app/services/confidence_service.py`.
- **What it does**: Analyzes the cosine similarity scores of the retrieved chunks to decide if we actually found relevant data.
- **Thresholds**: 
  - `High` if top_score > `0.35`
  - `Medium` if top_score > `0.20`
  - `Low` if top_score <= `0.20`
- **Why**: If confidence is Low (i.e. the user asked about "cooking" on a "programming" document), the system can proactively realize the document doesn't contain the answer, helping the LLM avoid hallucinations.

## 20. RAG Service Deep Dive
Files: `app/services/rag_service.py`.
This is the core orchestrator.
1. **Query Rewriting**: Calls LLM to contextualize the query using chat history.
2. **Retrieval**: Calls `RetrievalService.search()` -> Runs FAISS & BM25 -> Fuses via RRF -> Returns Top chunks.
3. **Empty Check**: If no chunks exist, returns an empty response immediately.
4. **Confidence**: Calls `ConfidenceService` to log how confident the search was.
5. **Truncation**: Truncates retrieved chunks to 1,500 chars to protect LLM context limits.
6. **Prompting**: Calls `PromptService.build_messages()` to create the system prompt containing the chunks.
7. **Groq Call**: Sends the payload to Groq.
8. **Citations**: Calls `_build_sources()` to map the retrieved chunks to the UI citation objects.
9. **Return**: Returns `QueryResponse` to the API router.

## 21. Prompting & Citations
Files: `app/services/prompt_service.py`.
- **System Prompt**: Strictly commands the LLM to "Act as an expert Q&A assistant... Base your answer SOLELY on the provided context... If the context does not contain the answer, say so explicitly."
- **Why?**: This mitigates hallucination. The LLM is forbidden from using its pre-trained knowledge.
- **Structure**: 
  - `[System Message]`: Instructions + `<CONTEXT> ... chunks ... </CONTEXT>`
  - `[History Messages]`: Previous user/assistant turns
  - `[User Message]`: The standalone question.
- **Citations**: The backend passes the `sources` array in the JSON response. The React frontend maps these sources (which contain `title`, `preview`, `relevance_score`) to UI cards below the message.

## 22. API Table

| Method | Endpoint | Purpose | Auth | Request | Response |
|--------|----------|---------|------|---------|----------|
| POST | `/api/auth/register` | Register user | No | `email`, `password` | `UserResponse` |
| POST | `/api/auth/login` | Authenticate | No | OAuth2 Form (`username`, `password`) | `access_token` |
| POST | `/api/documents/ingest` | Ingest URL | Yes | `url` | `DocumentResponse` |
| GET | `/api/documents` | List docs | Yes | None | `List[DocumentResponse]` |
| GET | `/api/documents/{id}/status` | Poll status | Yes | None | `{"status": "..."}` |
| POST | `/api/conversations` | Start chat | Yes | `document_id` | `ConversationResponse` |
| GET | `/api/conversations` | List chats | Yes | None | `List[ConversationResponse]` |
| GET | `/api/conversations/{id}` | Get chat | Yes | None | `ConversationResponse` |
| GET | `/api/conversations/{id}/messages` | Get history | Yes | None | `List[MessageResponse]` |
| POST | `/api/conversations/{id}/messages` | Ask question | Yes | `content` | `QueryResponse` |
| DELETE | `/api/conversations/{id}` | Delete chat | Yes | None | `{"status": "deleted"}` |

## 23. Frontend Walkthrough
- `frontend/src/api/client.js`: Creates an Axios instance. An interceptor automatically injects `Authorization: Bearer <token>` from localStorage into every request. Handles 401s by logging out.
- `frontend/src/context/AuthContext.jsx`: React Context that wraps the app. Provides `user`, `login()`, `logout()`. Uses `localStorage` to persist state across reloads.
- `frontend/src/App.jsx`: Sets up React Router. Routes like `/app` are wrapped in `<ProtectedRoute>`, which checks AuthContext and redirects to `/login` if unauthenticated.
- `frontend/src/pages/Dashboard.jsx`: Contains the URL input. Calls POST `/api/documents/ingest`, then uses a `setInterval` to poll `GET /api/documents/{id}/status`. Once "ready", it calls POST `/api/conversations` and uses `useNavigate` to go to `/app/chat/{conversation_id}`.
- `frontend/src/pages/Chat.jsx`: The main UI. Loads chat history on mount. When user sends a message, it renders optimistically, calls POST `/api/conversations/{id}/messages`, and then replaces the optimistic message with the LLM's answer and citation cards.
- `frontend/src/components/Sidebar.jsx`: Fetches and lists all active conversations from GET `/api/conversations`.

## 24. Security summary
- **Authentication**: JWT based. Stateless, secure.
- **Authorization**: Every API route validates that the requested Document or Conversation actually belongs to `current_user.id`. Prevents User A from reading User B's documents (IDOR).
- **Passwords**: Hashed with bcrypt.
- **SSRF**: Strict URL validation, private IP blocking, redirect hop blocking.
- **CORS**: Configured in `main.py` to only allow specific origins (like `http://localhost:5173`).
- **Secrets**: Uses `.env` via `config.py` so keys are never in git.

## 25. Error Handling
- **Invalid URL/SSRF**: Returns 400 Bad Request. Ingestion aborts.
- **403/404 Webpage**: Returns 400 with a specific error (e.g. "Webpage refused access").
- **LLM/Groq fails**: Caught in `rag_service.py`, logged, backend returns 500.
- **Missing/Invalid JWT**: FastAPI Dependency throws 401 Unauthorized. Axios interceptor catches 401 and clears localStorage, forcing a redirect to `/login`.

## 26. Testing
`tests/test_phase2_verification.py` contains 60 passing tests.
- **Health/Auth**: Tests JWT generation, duplicate email rejection.
- **SSRF**: Specifically tests that `http://127.0.0.1` and redirect chains to localhost are blocked (HTTP 400).
- **Isolation**: Tests that User B receives 404 when trying to access User A's document.
- **RAG Components**: Tests that RRF accurately fuses rankings, and Confidence generates High/Medium/Low based on cosine scores.
- **Database**: Validates schemas and foreign keys.

## 27. Deployment
To deploy this project to production:
1. **Frontend**: Build using `npm run build`. Serve static files via Nginx, Vercel, or AWS S3.
2. **Backend**: Run FastAPI using Gunicorn with Uvicorn workers (`gunicorn app.main:app -k uvicorn.workers.UvicornWorker`).
3. **Database**: Swap SQLite (`sqlite:///`) for PostgreSQL in `DATABASE_URL`. Run Alembic migrations on the Postgres instance.
4. **Vector Storage**: Current FAISS/BM25 uses local files (`.data/`). In a multi-server setup, this must be centralized (e.g., EFS, S3, or swap FAISS for a managed vector DB like Pinecone/Qdrant).
5. **CORS**: Update `BACKEND_CORS_ORIGINS` to the actual production frontend domain.

## 28. Limitations
- **Local FAISS/BM25**: Indexes are saved to the local filesystem. This prevents horizontal scaling (multiple backend servers) without a shared network drive.
- **Web Scraping**: Pure HTML scrapers (Trafilatura) fail on Single Page Applications (SPAs) that require JavaScript to render (like modern React apps). Also blocked by strong bot protections (Cloudflare).
- **Blocking Background Tasks**: `process_document` runs in a FastAPI `BackgroundTasks` thread. Heavy ingestion blocks the event loop slightly. Better suited for Celery/Redis.
- **Truncation**: Very large documents are truncated to 1500 chars per chunk during prompting, and only the top few chunks fit in the prompt. Huge semantic queries might miss context.

## 29. Why Each Technology?
| Technology | Why we use it |
|---|---|
| FastAPI | Fast, async, auto-generates Swagger docs, built-in Pydantic validation. |
| SQLAlchemy / Alembic | ORM for database interaction. Alembic handles schema versioning safely. |
| SQLite | Zero-config database perfect for local development. |
| JWT / bcrypt | Stateless authentication, industry standard secure password hashing. |
| Trafilatura | Heuristic-based scraping specifically designed to extract main body text. |
| SentenceTransformers | Local, free, highly accurate embeddings without API costs (MiniLM). |
| FAISS | Blazing fast local vector similarity search. |
| BM25 | Industry standard for lexical sparse search (keyword matching). |
| Groq | Extremely fast LLM inference (Llama 3.3). |
| React / Vite | Modern component-based UI, lightning-fast compilation. |
| Axios | Reliable HTTP client with interceptors for JWT injection. |
| TailwindCSS | Utility-first CSS for rapid, responsive UI development. |

## 30. Interview Explanation
**30 Seconds:**
"I built WebRAG, a full-stack Retrieval-Augmented Generation application. It allows users to ingest any webpage, extracting and vectorizing its content. When the user asks a question, the system performs a hybrid semantic and keyword search against the document, and uses a Groq-powered LLM to generate an answer strictly grounded in the retrieved text, complete with citations to prevent hallucination."

**2 Minutes:**
"WebRAG is a full-stack application I developed to solve LLM hallucination by grounding responses in user-provided documents. The backend is built with FastAPI and SQLite. When a user inputs a URL, an asynchronous pipeline fetches the HTML—with strict SSRF security protections—and uses Trafilatura to extract the core body text. 

I implemented a hybrid retrieval system. The text is chunked and embedded locally using SentenceTransformers, then indexed into FAISS for dense semantic search, and BM25 for sparse keyword search. When a user chats with the document, a Llama model rewrites the query for context, searches both indexes, and fuses the results using Reciprocal Rank Fusion. I also built a custom Confidence Service that evaluates the cosine similarity scores before prompting the LLM, ensuring the system can gracefully say 'I don't know' if the document lacks the answer. The frontend is a React SPA using Tailwind and Axios interceptors for JWT authentication, providing a seamless ChatGPT-like interface."

## 31. 40+ Interview Questions

**A. Project Overview**
1. **Q: What is the main problem WebRAG solves?** A: LLM hallucination. It forces the LLM to answer only using extracted context. (File: `rag_service.py`)
2. **Q: How does the system handle large documents?** A: Chunking. It breaks text into 200-word segments to fit context windows and improve retrieval accuracy. (File: `chunking_service.py`)

**B. FastAPI & Backend**
3. **Q: Why FastAPI over Flask/Django?** A: Native async support, high performance, and Pydantic validation.
4. **Q: How are background tasks handled?** A: Using FastAPI's `BackgroundTasks`, freeing the HTTP response while ingestion runs. (File: `api/documents.py`)

**C. Authentication**
5. **Q: How are passwords stored?** A: Hashed with bcrypt with a salt. Never plaintext. (File: `core/security.py`)
6. **Q: How do protected routes identify the user?** A: The `get_current_user` dependency decodes the Bearer JWT and queries the DB. (File: `api/auth.py`)

**D. Database**
7. **Q: How is data isolated between users?** A: Every Document and Conversation has a `user_id` foreign key. APIs filter by `current_user.id`. (File: `api/documents.py`)
8. **Q: What is Alembic?** A: A database migration tool. It safely applies schema changes (like adding a column) without deleting data.

**E. Web Scraping & Ingestion**
9. **Q: Why Trafilatura instead of BeautifulSoup?** A: Trafilatura intelligently targets the main article body and ignores noisy navbars and footers. (File: `ingestion_service.py`)
10. **Q: How does the system handle websites blocking bots?** A: We inject a custom `User-Agent` and handle 403 HTTP errors gracefully.

**F. Chunking**
11. **Q: How do you determine chunk boundaries?** A: We split on any number of newlines (`re.split(r'\n+')`) and group paragraphs until reaching a 200-word limit. (File: `chunking_service.py`)
12. **Q: What happens if a chunk is too large?** A: Semantic density dilutes, lowering cosine similarity, and the context might exceed LLM limits.

**G. Embeddings**
13. **Q: What embedding model do you use?** A: `sentence-transformers/all-MiniLM-L6-v2`. It's local, fast, and generates 384-dimensional vectors. (File: `embedding_service.py`)
14. **Q: Does the embedding model hit an API?** A: No, weights are downloaded to the server and executed locally on the CPU.

**H. FAISS**
15. **Q: What FAISS index type did you use?** A: `IndexFlatIP`. Inner product behaves exactly like Cosine Similarity since MiniLM normalizes its vectors. (File: `faiss_store.py`)
16. **Q: Where are FAISS indexes stored?** A: On the local disk under `.data/vectors/{doc_id}/`. 

**I. BM25**
17. **Q: Why add BM25 if you have FAISS?** A: FAISS is bad at exact keyword matches (like acronyms or IDs). BM25 handles sparse lexical search. (File: `bm25_service.py`)
18. **Q: How is the BM25 index persisted?** A: Pickled (serialized) to the local disk alongside FAISS.

**J. Hybrid Retrieval & RRF**
19. **Q: How do you combine FAISS and BM25 scores?** A: Reciprocal Rank Fusion (RRF). (File: `retrieval_service.py`)
20. **Q: Explain the RRF formula used.** A: `1 / (60 + rank)`. It rewards chunks that rank highly in both systems without requiring score normalization.

**K. RAG & LLM**
21. **Q: Why rewrite the query before searching?** A: To resolve pronouns. "What is it?" becomes "What is FastAPI?" using chat history. (File: `rag_service.py`)
22. **Q: How do you prevent hallucination?** A: Strict system prompts ("Answer SOLELY on the provided context") and Confidence scoring. (File: `prompt_service.py`)
23. **Q: What is the Confidence System?** A: It evaluates the top cosine score. If it's too low (e.g., < 0.2), we know the context is irrelevant. (File: `confidence_service.py`)

**L. Frontend**
24. **Q: How does the frontend handle JWTs?** A: Stored in `localStorage` and injected via an Axios interceptor. (File: `client.js`)
25. **Q: How does the UI update during ingestion?** A: It polls `/api/documents/{id}/status` on an interval until the status hits `ready`. (File: `Dashboard.jsx`)
26. **Q: How is chat history rendered?** A: Through standard React mapping, rendering user/assistant blocks and Markdown. (File: `Chat.jsx`)

**M. Security**
27. **Q: What is SSRF and how did you prevent it?** A: Server-Side Request Forgery. Prevented by blocking local/private IPs and validating redirect chains. (File: `ingestion_service.py`)
28. **Q: How do you prevent XSS?** A: React natively escapes variables. We also use a safe Markdown renderer.

**N. Deployment & Limitations**
29. **Q: How would you scale this to 10 instances?** A: Move SQLite to Postgres, and move local FAISS files to a centralized Vector DB like Pinecone.
30. **Q: What is the main limitation of Trafilatura?** A: It cannot execute Javascript, so Single Page Applications (SPAs) return empty HTML.

*(Additional conceptual questions omitted for brevity, but all rely on the architectures explained above).*

## 32. Code Block Walkthrough
**Example: `app/services/rag_service.py`**
- **Imports**: `Groq` for LLM, `RetrievalService` for search, `ConfidenceService` for evaluation.
- **Class `RAGService`**: Instantiates the Groq client and the RetrievalService using the target `document_id`.
- **`answer` method**:
  1. Calls `_rewrite_query()` to de-reference pronouns using chat history.
  2. Calls `_retrieve()` to fetch top chunks (FAISS + BM25 + RRF).
  3. Evaluates `confidence` using the top cosine score.
  4. Truncates the chunks to protect the context window.
  5. Calls `PromptService` to format the LLM prompt.
  6. Calls `_call_llm()` to hit the Groq API.
  7. Formats the citations and returns the response.
- **`_rewrite_query` method**: Short LLM call with `temperature=0` to accurately rewrite the question.

## 33. WEBRAG IN ONE PAGE
**Problem**: LLMs hallucinate. WebRAG grounds them in user-provided webpages.
**Tech Stack**: FastAPI, SQLite, React, Tailwind, Trafilatura, SentenceTransformers, FAISS, BM25, Groq (Llama 3).
**How it works**:
1. User submits URL.
2. SSRF check -> Download HTML -> Trafilatura Extracts body -> 200-word Chunks -> MiniLM Vectors -> FAISS & BM25 Indexes.
3. User asks a question.
4. LLM rewrites question -> Hybrid Search -> RRF Fusion -> Confidence Check -> LLM Answers using only context -> Citations displayed.
**Key Challenges Solved**:
- Extracting clean text without nav/footer noise (Trafilatura).
- Preventing internal network hacking (SSRF protections & redirect validation).
- Combining semantic and keyword searches (BM25 + FAISS via RRF).
- Handling "I don't know" gracefully (Cosine similarity Confidence scoring).
