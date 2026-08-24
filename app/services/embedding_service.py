from app.core.config import settings
from app.core.logging import get_logger
import httpx
import numpy as np
import time

logger = get_logger(__name__)

class EmbeddingService:
    def __init__(self):
        self.model = settings.EMBEDDING_MODEL
        self.api_url = f"https://router.huggingface.co/hf-inference/models/{self.model}/pipeline/feature-extraction"
        self.headers = {"Content-Type": "application/json"}
        if settings.HF_TOKEN:
            self.headers["Authorization"] = f"Bearer {settings.HF_TOKEN}"

    def generate_embeddings(self, chunks):
        logger.info("Generating embeddings using Hugging Face Inference API...")
        
        texts = []
        for chunk in chunks:
            if isinstance(chunk, dict):
                texts.append(chunk["content"])
            elif isinstance(chunk, str):
                texts.append(chunk)
            else:
                texts.append(chunk.content)

        if not texts:
            return np.array([])

        embeddings = []
        batch_size = 32
        
        # Batching logic
        with httpx.Client(timeout=30.0) as client:
            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i:i + batch_size]
                payload = {"inputs": batch_texts}
                
                max_retries = 3
                for attempt in range(max_retries):
                    try:
                        response = client.post(self.api_url, headers=self.headers, json=payload)
                        if response.status_code == 503:
                            # Model is loading
                            logger.info(f"HF API 503 (Model loading). Waiting... Attempt {attempt + 1}/{max_retries}")
                            time.sleep(10 * (attempt + 1))
                            continue
                            
                        response.raise_for_status()
                        batch_embeddings = response.json()
                        embeddings.extend(batch_embeddings)
                        break
                    except httpx.HTTPStatusError as e:
                        logger.error(f"HTTP error generating embeddings: {e.response.text}")
                        raise RuntimeError(f"Embedding API failed: {e.response.text}")
                    except httpx.RequestError as e:
                        logger.error(f"Request error generating embeddings: {str(e)}")
                        if attempt == max_retries - 1:
                            raise RuntimeError(f"Embedding API request failed: {str(e)}")
                        time.sleep(2)
        
        # Ensure we return a 2D numpy array and normalize to mimic SentenceTransformers behavior
        arr = np.array(embeddings, dtype=np.float32)
        if len(arr) > 0 and len(arr.shape) == 2:
            norms = np.linalg.norm(arr, axis=1, keepdims=True)
            arr = np.where(norms > 0, arr / norms, arr)
            
        logger.info(f"Generated {len(arr)} embeddings.")
        return arr