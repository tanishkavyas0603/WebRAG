"""
Regression test: some assistant messages saved before the reasoning-model fix
contain raw, unstripped <think>...</think> blocks (or an unclosed <think> with
no real answer). Feeding that back into the LLM as chat history on every
follow-up wastes tokens and can degrade or derail new answers.
get_recent_messages() must sanitize it on read. See app/services/conversation_service.py.
"""
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.models.db import Base, User, Document, Conversation, Message
from app.services.conversation_service import ConversationService, _sanitize_history_content


def test_sanitize_strips_closed_think_block():
    raw = "<think>internal reasoning here</think>The real answer."
    assert _sanitize_history_content(raw) == "The real answer."


def test_sanitize_strips_unclosed_think_block():
    raw = "<think>\nThinking Process:\n1. Identify the question...\n(never finishes)"
    assert _sanitize_history_content(raw) == "(no answer was generated for this turn)"


def test_sanitize_leaves_clean_content_untouched():
    raw = "FilePost gives you a permanent public CDN URL."
    assert _sanitize_history_content(raw) == raw


def test_get_recent_messages_sanitizes_polluted_assistant_message(tmp_path):
    db_path = tmp_path / "history_sanitize_test.db"
    engine = create_engine(f"sqlite:///{db_path}", connect_args={"check_same_thread": False})
    TestSession = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    Base.metadata.create_all(bind=engine)

    db = TestSession()
    try:
        user = User(email="history_sanitize_test@example.com", password_hash="x")
        db.add(user)
        db.commit()
        db.refresh(user)

        doc = Document(user_id=user.id, url="https://example.com", content_hash="h", status="ready")
        db.add(doc)
        db.commit()
        db.refresh(doc)

        conv = Conversation(user_id=user.id, document_id=doc.id, title="t")
        db.add(conv)
        db.commit()
        db.refresh(conv)

        db.add(Message(conversation_id=conv.id, role="user", content="what is HTTP"))
        db.add(Message(
            conversation_id=conv.id,
            role="assistant",
            content="<think>\nThinking Process:\n1. Scan the text...\n(truncated, no final answer)",
        ))
        db.commit()

        history = ConversationService(db).get_recent_messages(conv.id, limit=6)
        assistant_entries = [h for h in history if h["role"] == "assistant"]
        assert len(assistant_entries) == 1
        assert "<think>" not in assistant_entries[0]["content"]
        assert assistant_entries[0]["content"] == "(no answer was generated for this turn)"
    finally:
        db.close()
        engine.dispose()
