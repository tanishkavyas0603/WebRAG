from sqlalchemy.orm import Session
from app.models.db import Conversation, Message
from app.core.config import settings
from groq import Groq
from app.core.logging import get_logger

logger = get_logger(__name__)

class ConversationService:
    def __init__(self, db: Session):
        self.db = db
        if settings.GROQ_API_KEY:
            self.client = Groq(api_key=settings.GROQ_API_KEY)
        else:
            self.client = None

    def get_recent_messages(self, conversation_id: int, limit: int = 10) -> list[dict[str, str]]:
        messages = self.db.query(Message).filter(Message.conversation_id == conversation_id)\
            .order_by(Message.created_at.desc()).limit(limit).all()
            
        history = []
        for msg in reversed(messages):
            history.append({
                "role": msg.role,
                "content": msg.content
            })
        return history

    def add_message(self, conversation_id: int, role: str, content: str, citations: list = None) -> Message:
        msg = Message(
            conversation_id=conversation_id,
            role=role,
            content=content,
            citations=citations
        )
        self.db.add(msg)
        self.db.commit()
        self.db.refresh(msg)
        
        # Update conversation timestamp
        conv = self.db.query(Conversation).filter(Conversation.id == conversation_id).first()
        if conv:
            conv.updated_at = msg.created_at
            self.db.commit()
            
        return msg

    def generate_title(self, conversation: Conversation):
        if not self.client or conversation.title != "New Chat":
            return
            
        first_user_msg = self.db.query(Message).filter(
            Message.conversation_id == conversation.id,
            Message.role == "user"
        ).order_by(Message.created_at.asc()).first()
        
        if not first_user_msg:
            return
            
        prompt = (
            "Based on the following user message, generate a very short (2-4 words) "
            "title for a conversation. Do not use quotes or punctuation.\n\n"
            f"User: {first_user_msg.content}\nTitle:"
        )
        
        try:
            response = self.client.chat.completions.create(
                model=settings.GROQ_MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=10,
            )
            title_raw = response.choices[0].message.content.strip()
            import re
            # Remove complete and incomplete think blocks
            title = re.sub(r'<think>.*?</think>\s*', '', title_raw, flags=re.DOTALL).strip()
            title = re.sub(r'<think>.*', '', title, flags=re.DOTALL).strip()
            title = title.replace('"', '')
            
            if not title:
                title = first_user_msg.content[:30] + ("..." if len(first_user_msg.content) > 30 else "")
                
            conversation.title = title
            self.db.commit()
            logger.info(f"Generated title for conversation {conversation.id}: {title}")
        except Exception as e:
            logger.error(f"Failed to generate title: {e}")
