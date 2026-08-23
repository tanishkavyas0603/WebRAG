import re
from sqlalchemy.orm import Session
from app.models.db import Document, Chunk
from app.core.logging import get_logger

logger = get_logger(__name__)

class ChunkingService:
    MAX_WORDS = 200

    def __init__(self, db: Session, document: Document):
        self.db = db
        self.document = document

    def split_paragraphs(self, text: str) -> list[str]:
        # Split by one or more newlines to handle Trafilatura and BeautifulSoup outputs seamlessly
        paragraphs = [p.strip() for p in re.split(r'\n+', text) if p.strip()]
        return paragraphs

    def create_chunks(self, paragraphs: list[str]) -> list[Chunk]:
        chunks = []
        chunk_idx = 1
        current_chunk_paras = []
        current_words = 0
        current_heading = "Content"

        for p in paragraphs:
            # Heuristic for detecting a heading: short and no terminal punctuation
            if len(p.split()) < 15 and not p.endswith((".", ":", "?", "!", ",")):
                current_heading = p

            p_words = len(p.split())
            
            # If adding this paragraph exceeds limit (and we already have some text), finalize the chunk
            if current_words + p_words > self.MAX_WORDS and current_chunk_paras:
                content = "\n\n".join(current_chunk_paras)
                title = self.extract_title(content)
                
                chunk = Chunk(
                    document_id=self.document.id,
                    chunk_index=chunk_idx,
                    content=content,
                    metadata_={
                        "title": title if title else current_heading[:70],
                        "section": current_heading[:70],
                        "preview": self.extract_preview(content)
                    }
                )
                chunks.append(chunk)
                chunk_idx += 1
                current_chunk_paras = [p]
                current_words = p_words
            else:
                current_chunk_paras.append(p)
                current_words += p_words

        # Finalize the last chunk
        if current_chunk_paras:
            content = "\n\n".join(current_chunk_paras)
            title = self.extract_title(content)
            chunk = Chunk(
                document_id=self.document.id,
                chunk_index=chunk_idx,
                content=content,
                metadata_={
                    "title": title if title else current_heading[:70],
                    "section": current_heading[:70],
                    "preview": self.extract_preview(content)
                }
            )
            chunks.append(chunk)

        logger.info(f"Created {len(chunks)} chunks for document {self.document.id}")
        return chunks
   
    def save_chunks(self, chunks: list[Chunk]):
        # Clear any existing chunks for this document if we are re-processing
        self.db.query(Chunk).filter(Chunk.document_id == self.document.id).delete()
        
        self.db.add_all(chunks)
        self.db.commit()
        logger.info(f"Saved {len(chunks)} chunks to database.")

    def run(self) -> list[Chunk]:
        if not self.document.content:
            logger.warning("Document has no content to chunk.")
            return []
            
        paragraphs = self.split_paragraphs(self.document.content)
        chunks = self.create_chunks(paragraphs)
        self.save_chunks(chunks)
        return chunks

    def extract_title(self, content: str) -> str:
        lines = [line.strip() for line in content.splitlines() if line.strip()]
        if not lines:
            return "Content"
        first_line = lines[0]
        # If the first line is very long, it's likely a sentence, not a heading. Take the first few words.
        if len(first_line) > 70:
            first_line = first_line[:70].rsplit(" ", 1)[0]
        first_line = re.sub(r"\s+", " ", first_line)
        return first_line

    def extract_preview(self, content: str) -> str:
        preview = re.sub(r"\s+", " ", content)
        return preview[:170] + "..."

    def extract_section(self, title: str) -> str:
        # Fallback simplistic section extraction
        return "Content"