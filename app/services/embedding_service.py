from sentence_transformers import SentenceTransformer
from app.core.config import settings
from app.core.logging import get_logger

logger = get_logger(__name__)

class EmbeddingService:
    def __init__(self):
        self.model = SentenceTransformer(settings.EMBEDDING_MODEL)

    def generate_embeddings(self, chunks):
        logger.info("Generating embeddings...")
        
        texts = []
        for chunk in chunks:
            # Handle both dict and ORM model
            if isinstance(chunk, dict):
                texts.append(chunk["content"])
            else:
                texts.append(chunk.content)

        embeddings = self.model.encode(
            texts,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )

        logger.info(f"Generated {len(embeddings)} embeddings.")
        return embeddings