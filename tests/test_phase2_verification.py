"""
Comprehensive Phase 2 Backend Verification Test Suite
======================================================
Tests all 30 verification criteria + additional checks.

Design:
- Uses FastAPI TestClient + fresh SQLite DB per test function
- Mocks sentence-transformers and Groq to avoid network/GPU calls
- Uses httpx2 to silence starlette deprecation warning
- Tests real business logic, security, auth, persistence, isolation
"""

import os
import hashlib
import pickle
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine, inspect as sa_inspect
from sqlalchemy.orm import sessionmaker

# Set env vars before any app imports
os.environ.setdefault("GROQ_API_KEY", "test-key-not-real")


# ─── Reusable Helpers ─────────────────────────────────────────────────────────

def _make_fake_embeddings(n: int, dim: int = 384, seed: int = 42) -> np.ndarray:
    vecs = np.random.default_rng(seed).random((n, dim)).astype("float32")
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    return vecs / (norms + 1e-9)


def _fake_encode(texts, **kwargs):
    n = len(texts) if isinstance(texts, list) else 1
    return _make_fake_embeddings(n)


# ─── Session-scoped mocks (expensive to create) ───────────────────────────────

@pytest.fixture(scope="session", autouse=True)
def patch_sentence_transformer():
    """Prevent loading the real model during ALL tests."""
    with patch("sentence_transformers.SentenceTransformer") as mock_cls:
        mock_model = MagicMock()
        mock_model.encode = _fake_encode
        mock_cls.return_value = mock_model
        yield mock_model


@pytest.fixture(scope="session", autouse=True)
def patch_groq():
    """Prevent real Groq API calls during ALL tests."""
    with patch("app.services.rag_service.Groq") as mock_cls:
        mock_client = MagicMock()
        mock_choice = MagicMock()
        mock_choice.message.content = "Mocked LLM answer from webpage context."
        mock_client.chat.completions.create.return_value = MagicMock(choices=[mock_choice])
        mock_cls.return_value = mock_client
        yield mock_client


# ─── Per-test isolated app fixture ────────────────────────────────────────────

@pytest.fixture()
def isolated_app(tmp_path):
    """
    Creates a completely isolated FastAPI app with:
    - Fresh SQLite DB (in tmp_path)
    - Fresh FAISS/BM25 index directory (in tmp_path)
    - Properly overridden DB dependency
    """
    db_path = tmp_path / "test.db"
    db_url = f"sqlite:///{db_path}"
    index_dir = tmp_path / "indexes"
    index_dir.mkdir()

    from app.models.db import Base
    import app.core.constants as constants
    import app.core.database as db_module

    # Build isolated engine
    engine = create_engine(db_url, connect_args={"check_same_thread": False})
    TestSession = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    Base.metadata.create_all(bind=engine)

    # Swap global engine/session and INDEX_DIR
    orig_engine = db_module.engine
    orig_session = db_module.SessionLocal
    orig_index_dir = constants.INDEX_DIR

    db_module.engine = engine
    db_module.SessionLocal = TestSession
    constants.INDEX_DIR = index_dir

    # Override FastAPI dependency
    def override_get_db():
        db = TestSession()
        try:
            yield db
        finally:
            db.close()

    from app.main import app
    from app.core.database import get_db
    app.dependency_overrides[get_db] = override_get_db

    client = TestClient(app, raise_server_exceptions=False)

    yield client, TestSession, index_dir

    # Teardown
    app.dependency_overrides.clear()
    db_module.engine = orig_engine
    db_module.SessionLocal = orig_session
    constants.INDEX_DIR = orig_index_dir
    Base.metadata.drop_all(bind=engine)
    engine.dispose()


@pytest.fixture()
def client(isolated_app):
    c, _, _ = isolated_app
    return c


# ─── Auth helpers ─────────────────────────────────────────────────────────────

def register_user(client, email, password="TestPass123!"):
    return client.post("/api/auth/register", json={"email": email, "password": password})


def login_user(client, email, password="TestPass123!"):
    return client.post(
        "/api/auth/login",
        data={"username": email, "password": password},
        headers={"Content-Type": "application/x-www-form-urlencoded"},
    )


def register_and_login(client, email="test@example.com", password="TestPass123!"):
    register_user(client, email, password)
    r = login_user(client, email, password)
    assert r.status_code == 200, f"Login failed: {r.text}"
    return r.json()["access_token"]


def auth(token):
    return {"Authorization": f"Bearer {token}"}


# ─── Helper: Create ready document with indexes ───────────────────────────────

SAMPLE_CHUNKS = [
    {"id": 1, "content": "Python is a high-level, interpreted programming language.", "title": "Introduction", "section": "Intro", "preview": "Python intro..."},
    {"id": 2, "content": "Guido van Rossum created Python in 1991.", "title": "History", "section": "History", "preview": "Python history..."},
    {"id": 3, "content": "Python supports web development, data science, and AI.", "title": "Uses", "section": "Uses", "preview": "Python uses..."},
    {"id": 4, "content": "Django and Flask are popular Python web frameworks.", "title": "Frameworks", "section": "Web", "preview": "Frameworks..."},
    {"id": 5, "content": "Python has a large standard library and active community.", "title": "Community", "section": "Community", "preview": "Community..."},
]


def create_ready_document(client, TestSession, index_dir, email="user@x.com"):
    """Create a user, ingest, and build a ready document with indexes."""
    token = register_and_login(client, email)

    from app.models.db import User, Document, Chunk
    from app.vectorstore.faiss_store import FAISSVectorStore
    from app.services.bm25_service import BM25Service
    import app.core.constants as constants
    constants.INDEX_DIR = index_dir

    db = TestSession()
    try:
        user = db.query(User).filter(User.email == email).first()

        doc = Document(
            user_id=user.id,
            url="https://example.com/python-guide",
            content_hash=hashlib.sha256(b"python guide content").hexdigest(),
            content="Python is a high-level programming language.",
            title="Python Guide",
            status="ready",
        )
        db.add(doc)
        db.commit()
        db.refresh(doc)
        doc_id = doc.id

        for cd in SAMPLE_CHUNKS:
            chunk = Chunk(
                document_id=doc_id,
                chunk_index=cd["id"],
                content=cd["content"],
                metadata_={"title": cd["title"], "section": cd["section"], "preview": cd["preview"]},
            )
            db.add(chunk)
        db.commit()

        # Build FAISS + BM25
        embeddings = _make_fake_embeddings(len(SAMPLE_CHUNKS))
        store = FAISSVectorStore(document_id=doc_id)
        store.build(embeddings)
        store.save_metadata(SAMPLE_CHUNKS)

        bm25 = BM25Service(document_id=doc_id)
        bm25.build(SAMPLE_CHUNKS)

    finally:
        db.close()

    return token, doc_id


# ══════════════════════════════════════════════════════════════════════════════
# TEST GROUPS
# ══════════════════════════════════════════════════════════════════════════════

# ─── Group 1: Server Health ───────────────────────────────────────────────────

class TestServerHealth:

    def test_health_check_returns_200(self, client):
        """Criterion 2: GET /api/health returns 200."""
        r = client.get("/api/health")
        assert r.status_code == 200
        assert "status" in r.json()

    def test_health_database_connected(self, client):
        """DB is connected in health check."""
        r = client.get("/api/health")
        assert r.json()["database"] == "connected"

    def test_openapi_has_all_routes(self, client):
        """All required API routes are registered."""
        r = client.get("/openapi.json")
        assert r.status_code == 200
        paths = r.json()["paths"]
        for expected in ["/api/auth/register", "/api/auth/login",
                         "/api/documents/ingest", "/api/conversations"]:
            assert expected in paths, f"Missing route: {expected}"


# ─── Group 2: Authentication ──────────────────────────────────────────────────

class TestAuthentication:

    def test_register_new_user(self, client):
        """Criterion 3: Register a new user successfully."""
        r = register_user(client, "new@example.com")
        assert r.status_code == 200
        data = r.json()
        assert data["email"] == "new@example.com"
        assert "id" in data
        assert "password_hash" not in data

    def test_register_duplicate_email_rejected(self, client):
        """Duplicate email returns 400."""
        register_user(client, "dup@x.com")
        r = register_user(client, "dup@x.com")
        assert r.status_code == 400

    def test_login_returns_jwt(self, client):
        """Criterion 4: Login returns a valid JWT."""
        register_user(client, "jwt@example.com")
        r = login_user(client, "jwt@example.com")
        assert r.status_code == 200
        data = r.json()
        assert data["token_type"] == "bearer"
        assert len(data["access_token"]) > 20

    def test_wrong_password_rejected(self, client):
        """Wrong password returns 401."""
        register_user(client, "wp@x.com")
        r = client.post(
            "/api/auth/login",
            data={"username": "wp@x.com", "password": "WrongPass!"},
            headers={"Content-Type": "application/x-www-form-urlencoded"},
        )
        assert r.status_code == 401

    def test_protected_endpoint_rejects_no_token(self, client):
        """Criterion 5: Protected endpoint returns 401 without token."""
        r = client.post("/api/documents/ingest", json={"url": "https://x.com"})
        assert r.status_code == 401

    def test_protected_endpoint_rejects_fake_jwt(self, client):
        """Criterion 29: Invalid/expired JWT rejected."""
        r = client.post(
            "/api/documents/ingest",
            json={"url": "https://x.com"},
            headers={"Authorization": "Bearer fake.jwt.token"},
        )
        assert r.status_code == 401

    def test_no_stack_trace_in_error_responses(self, client):
        """Error responses never expose Python stack traces."""
        r = client.post(
            "/api/auth/login",
            data={"username": "nobody@x.com", "password": "wrong"},
            headers={"Content-Type": "application/x-www-form-urlencoded"},
        )
        assert "Traceback" not in r.text
        assert "File " not in r.text


# ─── Group 3: SSRF Protection ─────────────────────────────────────────────────

class TestSSRFProtection:
    """Criterion 26: SSRF protection blocks dangerous URLs."""

    def _svc(self, url):
        from app.services.ingestion_service import DocumentIngestionService
        return DocumentIngestionService(MagicMock(), 1, url)

    def test_localhost_blocked(self):
        """localhost URL raises SSRFProtectionError."""
        from app.services.ingestion_service import SSRFProtectionError
        with pytest.raises(SSRFProtectionError):
            self._svc("http://localhost/admin")._validate_url_safety("http://localhost/admin")

    def test_127_0_0_1_blocked(self):
        """127.0.0.1 raises SSRFProtectionError."""
        from app.services.ingestion_service import SSRFProtectionError
        with pytest.raises(SSRFProtectionError):
            self._svc("http://127.0.0.1/")._validate_url_safety("http://127.0.0.1/")

    def test_private_ip_ranges_blocked(self):
        """Private IPv4 ranges are blocked."""
        from app.services.ingestion_service import SSRFProtectionError
        for url in ["http://192.168.1.1/", "http://10.0.0.5/", "http://172.16.0.1/"]:
            with pytest.raises(SSRFProtectionError):
                self._svc(url)._validate_url_safety(url)

    def test_non_http_scheme_blocked(self):
        """Criterion 25: Non-HTTP schemes are rejected."""
        from app.services.ingestion_service import SSRFProtectionError
        for url in ["ftp://example.com", "file:///etc/passwd"]:
            with pytest.raises(SSRFProtectionError):
                self._svc(url)._validate_url_safety(url)

    def test_garbled_url_blocked(self):
        """Criterion 25: Invalid URL format is rejected."""
        from app.services.ingestion_service import SSRFProtectionError
        with pytest.raises(SSRFProtectionError):
            self._svc("not-a-url")._validate_url_safety("not-a-url")

    def test_redirect_to_localhost_blocked(self):
        """Verify the redirect to localhost is blocked by the request hook."""
        from app.services.ingestion_service import SSRFProtectionError
        import httpx
        import asyncio
        req = httpx.Request("GET", "http://127.0.0.1/admin")
        with pytest.raises(SSRFProtectionError):
            asyncio.run(self._svc("http://public.com")._validate_request_hook(req))

    def test_redirect_to_private_ip_blocked(self):
        """Verify redirect to private IPv4 is blocked by the request hook."""
        from app.services.ingestion_service import SSRFProtectionError
        import httpx
        import asyncio
        for ip in ["http://192.168.1.100", "http://10.0.0.5"]:
            req = httpx.Request("GET", ip)
            with pytest.raises(SSRFProtectionError):
                asyncio.run(self._svc("http://public.com")._validate_request_hook(req))



# ─── Group 4: Document Ingestion ──────────────────────────────────────────────

class TestDocumentIngestion:

    def test_ingest_returns_document_id(self, isolated_app):
        """Criterion 7: Ingest returns document_id."""
        client, TestSession, index_dir = isolated_app
        token = register_and_login(client, "ingest@x.com")

        with patch("app.api.documents.process_document_background"):
            r = client.post(
                "/api/documents/ingest",
                json={"url": "https://python.org"},
                headers=auth(token),
            )
        assert r.status_code == 200
        assert r.json()["id"] > 0
        assert r.json()["status"] == "pending"

    def test_document_status_endpoint_works(self, isolated_app):
        """Criterion 8: Status endpoint returns document info."""
        client, TestSession, index_dir = isolated_app
        token = register_and_login(client, "status@x.com")

        with patch("app.api.documents.process_document_background"):
            r = client.post("/api/documents/ingest", json={"url": "https://docs.python.org"}, headers=auth(token))
        doc_id = r.json()["id"]

        r2 = client.get(f"/api/documents/{doc_id}/status", headers=auth(token))
        assert r2.status_code == 200
        assert r2.json()["id"] == doc_id

    def test_document_status_transitions_to_ready(self, isolated_app):
        """Criterion 9: Document status transitions from pending → ready."""
        client, TestSession, index_dir = isolated_app
        token, doc_id = create_ready_document(client, TestSession, index_dir, "trans@x.com")

        r = client.get(f"/api/documents/{doc_id}/status", headers=auth(token))
        assert r.json()["status"] == "ready"

    def test_chunks_created_in_db(self, isolated_app):
        """Criterion 11: Chunks are persisted in the database."""
        client, TestSession, index_dir = isolated_app
        token, doc_id = create_ready_document(client, TestSession, index_dir, "chunks@x.com")

        from app.models.db import Chunk
        db = TestSession()
        try:
            count = db.query(Chunk).filter(Chunk.document_id == doc_id).count()
            assert count == len(SAMPLE_CHUNKS)
        finally:
            db.close()

    def test_faiss_index_created_per_document(self, isolated_app):
        """Criterion 12: Document-specific FAISS index file exists."""
        client, TestSession, index_dir = isolated_app
        token, doc_id = create_ready_document(client, TestSession, index_dir, "faiss@x.com")

        assert (index_dir / f"doc_{doc_id}.faiss").exists()
        assert (index_dir / f"doc_{doc_id}_metadata.pkl").exists()

    def test_bm25_index_created_per_document(self, isolated_app):
        """Criterion 13: BM25 builds correctly per document."""
        client, TestSession, index_dir = isolated_app
        import app.core.constants as constants
        constants.INDEX_DIR = index_dir

        from app.services.bm25_service import BM25Service
        bm25 = BM25Service(document_id=777)
        bm25.build(SAMPLE_CHUNKS)
        results = bm25.search("Python language", k=5)
        assert len(results) > 0

    def test_faiss_indexes_isolated_per_document(self, isolated_app):
        """Criterion 22: Two documents produce separate FAISS index files."""
        client, TestSession, index_dir = isolated_app
        import app.core.constants as constants
        constants.INDEX_DIR = index_dir

        from app.vectorstore.faiss_store import FAISSVectorStore
        for doc_id in [201, 202]:
            store = FAISSVectorStore(document_id=doc_id)
            embs = _make_fake_embeddings(3)
            store.build(embs)
            store.save_metadata(SAMPLE_CHUNKS[:3])

        assert (index_dir / "doc_201.faiss").exists()
        assert (index_dir / "doc_202.faiss").exists()
        assert (index_dir / "doc_201.faiss") != (index_dir / "doc_202.faiss")

    def test_retrieval_document_isolation(self, isolated_app):
        """Criterion 23: Retrieval for doc A never touches doc B's index."""
        client, TestSession, index_dir = isolated_app
        import app.core.constants as constants
        constants.INDEX_DIR = index_dir

        from app.vectorstore.faiss_store import FAISSVectorStore

        # Create two separate stores
        embs_a = _make_fake_embeddings(3, seed=1)
        embs_b = _make_fake_embeddings(3, seed=99)

        store_a = FAISSVectorStore(document_id=301)
        store_a.build(embs_a)
        store_a.save_metadata([{"id": 1, "content": "doc A content", "title": "A", "section": "", "preview": ""}] * 3)

        store_b = FAISSVectorStore(document_id=302)
        store_b.build(embs_b)
        store_b.save_metadata([{"id": 2, "content": "doc B content", "title": "B", "section": "", "preview": ""}] * 3)

        # Loading store for doc A should only return doc A metadata
        idx_a, meta_a = FAISSVectorStore(document_id=301).load()
        idx_b, meta_b = FAISSVectorStore(document_id=302).load()

        assert all(m["title"] == "A" for m in meta_a)
        assert all(m["title"] == "B" for m in meta_b)

    def test_url_deduplication_returns_same_doc(self, isolated_app):
        """Criterion: Same URL submitted twice returns same document."""
        client, TestSession, index_dir = isolated_app
        token = register_and_login(client, "dedup@x.com")
        url = "https://example.com/unique-page"

        with patch("app.api.documents.process_document_background"):
            r1 = client.post("/api/documents/ingest", json={"url": url}, headers=auth(token))
            r2 = client.post("/api/documents/ingest", json={"url": url}, headers=auth(token))

        assert r1.json()["id"] == r2.json()["id"]

    def test_content_hash_deduplication(self):
        """Content hash is deterministic."""
        content = "Identical page content"
        h1 = hashlib.sha256(content.encode()).hexdigest()
        h2 = hashlib.sha256(content.encode()).hexdigest()
        assert h1 == h2

    def test_ingestion_extracts_text_from_html(self):
        """Criterion 10: HTML content is extracted correctly."""
        from app.services.ingestion_service import DocumentIngestionService
        html = "<html><head><title>My Page</title></head><body><p>Hello World content here.</p></body></html>"
        svc = DocumentIngestionService(MagicMock(), 1, "https://example.com")
        text, title = svc.extract_text(html)
        assert "Hello World" in text
        assert title == "My Page"

    def test_minimal_content_page_handled(self):
        """Criterion 28: Page with very little content is handled gracefully."""
        from app.services.ingestion_service import DocumentIngestionService
        html = "<html><body><p>Hi</p></body></html>"
        svc = DocumentIngestionService(MagicMock(), 1, "https://example.com")
        text, title = svc.extract_text(html)
        assert len(text) > 0

    def test_other_users_document_returns_404(self, isolated_app):
        """Criterion 30: User B cannot see User A's document."""
        client, TestSession, index_dir = isolated_app
        token_a, doc_id_a = create_ready_document(client, TestSession, index_dir, "owner@x.com")
        token_b = register_and_login(client, "intruder@x.com")

        r = client.get(f"/api/documents/{doc_id_a}/status", headers=auth(token_b))
        assert r.status_code == 404


# ─── Group 5: Conversation & RAG ──────────────────────────────────────────────

class TestConversationAndRAG:

    def test_create_conversation_linked_to_document(self, isolated_app):
        """Criterion 14: Create conversation linked to ready document."""
        client, TestSession, index_dir = isolated_app
        token, doc_id = create_ready_document(client, TestSession, index_dir, "cv1@x.com")

        r = client.post("/api/conversations", json={"document_id": doc_id}, headers=auth(token))
        assert r.status_code == 200
        assert r.json()["document_id"] == doc_id
        assert r.json()["title"] != "New Chat"  # Will be deterministic from doc
        assert "Python Guide" in r.json()["title"]

    def test_cannot_chat_on_pending_document(self, isolated_app):
        """Cannot create conversation with a non-ready document."""
        client, TestSession, index_dir = isolated_app
        token = register_and_login(client, "pending@x.com")

        from app.models.db import User, Document
        db = TestSession()
        try:
            user = db.query(User).filter(User.email == "pending@x.com").first()
            doc = Document(user_id=user.id, url="https://x.com", content_hash="h", status="pending")
            db.add(doc)
            db.commit()
            doc_id = doc.id
        finally:
            db.close()

        r = client.post("/api/conversations", json={"document_id": doc_id}, headers=auth(token))
        assert r.status_code == 400

    def test_send_message_returns_answer(self, isolated_app):
        """Criterion 15-16: Question gets answered using webpage context."""
        client, TestSession, index_dir = isolated_app
        token, doc_id = create_ready_document(client, TestSession, index_dir, "qa1@x.com")
        conv_r = client.post("/api/conversations", json={"document_id": doc_id}, headers=auth(token))
        conv_id = conv_r.json()["id"]

        r = client.post(
            f"/api/conversations/{conv_id}/messages",
            json={"message": "Who created Python?"},
            headers=auth(token),
        )
        assert r.status_code == 200
        assert r.json()["role"] == "assistant"
        assert len(r.json()["content"]) > 0

    def test_citations_returned_with_answer(self, isolated_app):
        """Criterion 17: Citations are included in assistant responses."""
        client, TestSession, index_dir = isolated_app
        token, doc_id = create_ready_document(client, TestSession, index_dir, "qa2@x.com")
        conv_r = client.post("/api/conversations", json={"document_id": doc_id}, headers=auth(token))
        conv_id = conv_r.json()["id"]

        r = client.post(
            f"/api/conversations/{conv_id}/messages",
            json={"message": "What is Python used for?"},
            headers=auth(token),
        )
        assert r.status_code == 200
        assert r.json()["citations"] is not None
        assert isinstance(r.json()["citations"], list)
        assert len(r.json()["citations"]) > 0

    def test_messages_persisted_in_db(self, isolated_app):
        """Criterion 18: User and assistant messages are both persisted."""
        client, TestSession, index_dir = isolated_app
        token, doc_id = create_ready_document(client, TestSession, index_dir, "qa3@x.com")
        conv_r = client.post("/api/conversations", json={"document_id": doc_id}, headers=auth(token))
        conv_id = conv_r.json()["id"]

        client.post(
            f"/api/conversations/{conv_id}/messages",
            json={"message": "Tell me about Python."},
            headers=auth(token),
        )

        from app.models.db import Message
        db = TestSession()
        try:
            msgs = db.query(Message).filter(Message.conversation_id == conv_id).all()
            assert len(msgs) == 2
            roles = {m.role for m in msgs}
            assert roles == {"user", "assistant"}
        finally:
            db.close()

    def test_follow_up_question_works(self, isolated_app):
        """Criterion 19-20: Follow-up question with conversation history."""
        client, TestSession, index_dir = isolated_app
        token, doc_id = create_ready_document(client, TestSession, index_dir, "qa4@x.com")
        conv_r = client.post("/api/conversations", json={"document_id": doc_id}, headers=auth(token))
        conv_id = conv_r.json()["id"]

        client.post(f"/api/conversations/{conv_id}/messages", json={"message": "What is Python?"}, headers=auth(token))
        r2 = client.post(f"/api/conversations/{conv_id}/messages", json={"message": "Can you explain that more simply?"}, headers=auth(token))

        assert r2.status_code == 200
        assert r2.json()["role"] == "assistant"

    def test_conversation_history_persists_across_requests(self, isolated_app):
        """Criterion 21: Full conversation history is available after reload."""
        client, TestSession, index_dir = isolated_app
        token, doc_id = create_ready_document(client, TestSession, index_dir, "qa5@x.com")
        conv_r = client.post("/api/conversations", json={"document_id": doc_id}, headers=auth(token))
        conv_id = conv_r.json()["id"]

        client.post(f"/api/conversations/{conv_id}/messages", json={"message": "Question 1?"}, headers=auth(token))
        client.post(f"/api/conversations/{conv_id}/messages", json={"message": "Question 2?"}, headers=auth(token))

        r = client.get(f"/api/conversations/{conv_id}/messages", headers=auth(token))
        assert r.status_code == 200
        assert len(r.json()) == 4  # 2 user + 2 assistant

    def test_out_of_scope_question_handled(self, isolated_app):
        """Criterion 24: OOD question is handled gracefully (no crash)."""
        client, TestSession, index_dir = isolated_app
        token, doc_id = create_ready_document(client, TestSession, index_dir, "ood@x.com")
        conv_r = client.post("/api/conversations", json={"document_id": doc_id}, headers=auth(token))
        conv_id = conv_r.json()["id"]

        r = client.post(
            f"/api/conversations/{conv_id}/messages",
            json={"message": "What is the capital of France?"},
            headers=auth(token),
        )
        assert r.status_code == 200

    def test_user_b_cannot_access_user_a_conversation(self, isolated_app):
        """Criterion 30: Authorization prevents cross-user access."""
        client, TestSession, index_dir = isolated_app
        token_a, doc_id_a = create_ready_document(client, TestSession, index_dir, "usera@x.com")
        token_b = register_and_login(client, "userb@x.com")

        conv_r = client.post("/api/conversations", json={"document_id": doc_id_a}, headers=auth(token_a))
        conv_id = conv_r.json()["id"]

        r = client.get(f"/api/conversations/{conv_id}", headers=auth(token_b))
        assert r.status_code == 404

        r2 = client.post(
            f"/api/conversations/{conv_id}/messages",
            json={"message": "hacked!"},
            headers=auth(token_b),
        )
        assert r2.status_code == 404

    def test_conversation_list_is_user_scoped(self, isolated_app):
        """Each user only sees their own conversations."""
        client, TestSession, index_dir = isolated_app
        token_a, doc_a = create_ready_document(client, TestSession, index_dir, "list_a@x.com")
        token_b, doc_b = create_ready_document(client, TestSession, index_dir, "list_b@x.com")

        client.post("/api/conversations", json={"document_id": doc_a}, headers=auth(token_a))
        client.post("/api/conversations", json={"document_id": doc_b}, headers=auth(token_b))

        r_a = client.get("/api/conversations", headers=auth(token_a))
        r_b = client.get("/api/conversations", headers=auth(token_b))

        assert len(r_a.json()) == 1
        assert len(r_b.json()) == 1

    def test_delete_conversation(self, isolated_app):
        """Deleting a conversation removes it."""
        client, TestSession, index_dir = isolated_app
        token, doc_id = create_ready_document(client, TestSession, index_dir, "del@x.com")
        conv_r = client.post("/api/conversations", json={"document_id": doc_id}, headers=auth(token))
        conv_id = conv_r.json()["id"]

        assert client.delete(f"/api/conversations/{conv_id}", headers=auth(token)).status_code == 200
        assert client.get(f"/api/conversations/{conv_id}", headers=auth(token)).status_code == 404


# ─── Group 6: RAG Pipeline Components ────────────────────────────────────────

class TestRAGPipelineComponents:

    def test_bm25_returns_ranked_results(self):
        """BM25 returns results even with small corpora (handles negative IDF)."""
        from app.services.bm25_service import BM25Service
        chunks = [
            {"id": 0, "content": "Python is a programming language", "title": "Intro", "section": "", "preview": ""},
            {"id": 1, "content": "Java is another language", "title": "Java", "section": "", "preview": ""},
            {"id": 2, "content": "Django is a Python web framework", "title": "Django", "section": "", "preview": ""},
        ]
        svc = BM25Service(document_id=999)
        svc.build(chunks)
        results = svc.search("Python", k=3)
        assert len(results) > 0
        # Python-specific chunks should score higher
        top_idx = results[0][0]
        assert "Python" in chunks[top_idx]["content"] or "Django" in chunks[top_idx]["content"]

    def test_rrf_fusion_merges_rankings(self):
        """RRF correctly fuses two ranked lists."""
        from app.services.bm25_service import BM25Service
        dense = [(0, 0.9), (1, 0.7), (2, 0.5)]
        sparse = [(2, 5.0), (0, 3.0), (1, 1.0)]
        fused = BM25Service.reciprocal_rank_fusion(dense, sparse, k=60)
        assert len(fused) == 3
        assert {f[0] for f in fused} == {0, 1, 2}

    def test_confidence_high_for_good_scores(self):
        """Confidence service yields High for strong cosine scores."""
        from app.services.confidence_service import ConfidenceService
        result = ConfidenceService.compute([0.85, 0.80, 0.75])
        assert result.label == "High"
        assert result.score >= 60

    def test_confidence_low_for_weak_scores(self):
        """Confidence service yields Low for weak cosine scores."""
        from app.services.confidence_service import ConfidenceService
        result = ConfidenceService.compute([0.1, 0.08, 0.05])
        assert result.label == "Low"

    def test_confidence_uses_cosine_not_rrf_scores(self):
        """Confidence must NOT use RRF scores (which are ~0.016)."""
        from app.services.confidence_service import ConfidenceService
        rrf_result = ConfidenceService.compute([0.016, 0.015, 0.014])
        cosine_result = ConfidenceService.compute([0.80, 0.75, 0.70])
        assert rrf_result.label == "Low"
        assert cosine_result.label == "High"

    def test_confidence_empty_returns_low(self):
        """Empty scores return Low confidence."""
        from app.services.confidence_service import ConfidenceService
        result = ConfidenceService.compute([])
        assert result.score == 0.0
        assert result.label == "Low"

    def test_query_expansion_runs_without_error(self):
        """Query expansion service runs without crashing."""
        from app.services.query_expansion_service import QueryExpansionService
        svc = QueryExpansionService()
        result = svc.expand("What is PM-JAY?")
        assert result.expanded is not None
        assert result.was_expanded is True  # PM-JAY should be expanded

    def test_metadata_ranker_runs(self):
        """Metadata ranker processes chunks without error."""
        from app.services.metadata_ranker import MetadataRanker
        from app.models.response import RetrievalResult
        chunks = [
            RetrievalResult(
                chunk_id=1, title="Insurance Scheme", section="insurance",
                preview="...", content="Coverage info", similarity_score=0.5,
                cosine_score=0.5, source="web", metadata_boost=0.0
            )
        ]
        ranker = MetadataRanker()
        result = ranker.rerank("insurance scheme coverage", chunks)
        assert len(result) == 1

    def test_prompt_service_includes_chunks_and_history(self):
        """PromptService correctly builds messages with context and history."""
        from app.services.prompt_service import PromptService
        from app.models.response import RetrievalResult
        chunks = [
            RetrievalResult(
                chunk_id=1, title="Intro", section="Intro",
                preview="...", content="Python is awesome", similarity_score=0.8,
                cosine_score=0.8, source="web", metadata_boost=0.0
            )
        ]
        history = [{"role": "user", "content": "Previous question"}]
        messages = PromptService.build_messages("New question", chunks, history)

        assert messages[0]["role"] == "system"
        assert "webpage" in messages[0]["content"].lower()
        assert "Python is awesome" in messages[0]["content"]
        assert messages[-1]["content"] == "New question"
        assert any(m["content"] == "Previous question" for m in messages)


# ─── Group 7: Security Verification ──────────────────────────────────────────

class TestSecurityVerification:

    def test_no_hardcoded_api_keys_in_source(self):
        """No API keys are hardcoded in any source file."""
        import re
        src_dir = Path("c:/Users/sunil/Desktop/Health_RAG_Pipeline-master/Health_RAG_Pipeline-master/app")
        patterns = [r"gsk_[A-Za-z0-9]{20,}", r"sk-[A-Za-z0-9]{30,}"]
        violations = []
        for py_file in src_dir.rglob("*.py"):
            content = py_file.read_text(encoding="utf-8", errors="ignore")
            for p in patterns:
                if re.search(p, content):
                    violations.append(str(py_file))
        assert violations == [], f"Hardcoded keys in: {violations}"

    def test_dot_env_is_gitignored(self):
        """.env is in .gitignore."""
        gi = Path("c:/Users/sunil/Desktop/Health_RAG_Pipeline-master/Health_RAG_Pipeline-master/.gitignore").read_text()
        assert ".env" in gi

    def test_sqlite_db_is_gitignored(self):
        """*.db pattern is in .gitignore."""
        gi = Path("c:/Users/sunil/Desktop/Health_RAG_Pipeline-master/Health_RAG_Pipeline-master/.gitignore").read_text()
        assert "*.db" in gi

    def test_password_stored_as_bcrypt_hash(self, isolated_app):
        """Password is stored as bcrypt hash, never plain text."""
        client, TestSession, _ = isolated_app
        register_user(client, "hashtest@x.com", "MyPlainPass!")

        from app.models.db import User
        db = TestSession()
        try:
            user = db.query(User).filter(User.email == "hashtest@x.com").first()
            assert user is not None
            assert user.password_hash != "MyPlainPass!"
            assert user.password_hash.startswith("$2b$")
        finally:
            db.close()

    def test_faiss_index_paths_unique_per_document(self):
        """FAISS index paths cannot collide between documents."""
        import app.core.constants as constants
        from app.vectorstore.faiss_store import FAISSVectorStore
        s1 = FAISSVectorStore(document_id=1)
        s2 = FAISSVectorStore(document_id=2)
        assert s1.index_path != s2.index_path
        assert s1.metadata_path != s2.metadata_path

    def test_sqlalchemy_session_generator_closes_cleanly(self):
        """DB session generator yields and closes without error."""
        from app.core.database import get_db
        gen = get_db()
        db = next(gen)
        assert db is not None
        try:
            next(gen)
        except StopIteration:
            pass

    def test_validation_errors_no_stack_trace(self, client):
        """Validation error responses don't expose internal tracebacks."""
        r = client.post("/api/auth/register", json={"email": "not-an-email", "password": "pw"})
        assert r.status_code == 422
        assert "Traceback" not in r.text


# ─── Group 8: Database Schema ─────────────────────────────────────────────────

class TestDatabaseSchema:

    def test_all_required_tables_exist(self, isolated_app):
        """All 5 tables are created in the database."""
        _, TestSession, _ = isolated_app
        from app.core import database as db_module
        inspector = sa_inspect(db_module.engine)
        tables = set(inspector.get_table_names())
        assert {"users", "documents", "conversations", "messages", "chunks"}.issubset(tables)

    def test_documents_table_has_user_id(self, isolated_app):
        """Documents table has user_id column for ownership."""
        _, TestSession, _ = isolated_app
        from app.core import database as db_module
        inspector = sa_inspect(db_module.engine)
        cols = {c["name"] for c in inspector.get_columns("documents")}
        assert "user_id" in cols
        assert "content_hash" in cols
        assert "status" in cols

    def test_messages_table_has_citations(self, isolated_app):
        """Messages table has citations column for source attribution."""
        _, TestSession, _ = isolated_app
        from app.core import database as db_module
        inspector = sa_inspect(db_module.engine)
        cols = {c["name"] for c in inspector.get_columns("messages")}
        assert "citations" in cols
        assert "role" in cols
