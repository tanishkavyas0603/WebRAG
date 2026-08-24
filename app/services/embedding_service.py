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
        
        if not settings.HF_TOKEN:
            raise ValueError("HF_TOKEN environment variable is missing. Authentication is required for Hugging Face Inference API.")
            
        self.headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {settings.HF_TOKEN}"
        }

    def generate_embeddings(self, chunks):
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
        total_batches = (len(texts) + batch_size - 1) // batch_size
        
        timeout = httpx.Timeout(30.0)
        
        with httpx.Client(timeout=timeout) as client:
            for i in range(0, len(texts), batch_size):
                batch_index = (i // batch_size) + 1
                batch_texts = texts[i:i + batch_size]
                payload = {"inputs": batch_texts}
                
                logger.info(f"[EMBEDDING] Requesting embeddings for batch {batch_index}/{total_batches}")
                
                max_retries = 3
                for attempt in range(1, max_retries + 1):
                    logger.info("[EMBEDDING] HuggingFace request started")
                    try:
                        response = client.post(self.api_url, headers=self.headers, json=payload)
                        logger.info(f"[EMBEDDING] HuggingFace response received: HTTP {response.status_code}")
                        
                        if response.status_code == 200:
                            batch_embeddings = response.json()
                            embeddings.extend(batch_embeddings)
                            logger.info(f"[EMBEDDING] Batch {batch_index}/{total_batches} completed")
                            break
                            
                        elif response.status_code in (401, 403):
                            logger.error(f"[EMBEDDING] ERROR: HuggingFace returned HTTP {response.status_code}")
                            raise RuntimeError(f"Hugging Face authentication failed (HTTP {response.status_code}). Please check your HF_TOKEN.")
                            
                        elif response.status_code == 429:
                            logger.error(f"[EMBEDDING] ERROR: HuggingFace returned HTTP 429")
                            raise RuntimeError("Hugging Face API rate limit exceeded (HTTP 429).")
                            
                        elif response.status_code == 503:
                            if attempt == max_retries:
                                logger.error("[EMBEDDING] ERROR: HuggingFace returned HTTP 503 consistently.")
                                raise RuntimeError("Hugging Face API is unavailable (Model loading failed after retries).")
                            backoff = 2 ** attempt
                            logger.info(f"HF API 503 (Model loading). Waiting {backoff}s... Attempt {attempt}/{max_retries}")
                            time.sleep(backoff)
                            continue
                            
                        else:
                            response.raise_for_status()
                            
                    except httpx.TimeoutException:
                        logger.error("[EMBEDDING] ERROR: HuggingFace request timed out")
                        if attempt == max_retries:
                            raise RuntimeError("Hugging Face API request timed out.")
                        backoff = 2 ** attempt
                        logger.info(f"Request timed out. Waiting {backoff}s... Attempt {attempt}/{max_retries}")
                        time.sleep(backoff)
                        
                    except httpx.RequestError as e:
                        logger.error(f"[EMBEDDING] ERROR: Network error - {str(e)}")
                        if attempt == max_retries:
                            raise RuntimeError(f"Embedding API network request failed: {str(e)}")
                        backoff = 2 ** attempt
                        logger.info(f"Network error. Waiting {backoff}s... Attempt {attempt}/{max_retries}")
                        time.sleep(backoff)
                        
                    except httpx.HTTPStatusError as e:
                        logger.error(f"[EMBEDDING] ERROR: HuggingFace returned HTTP {e.response.status_code}")
                        raise RuntimeError(f"Embedding API failed with HTTP {e.response.status_code}: {e.response.text}")

        arr = np.array(embeddings, dtype=np.float32)
        if len(arr) > 0 and len(arr.shape) == 2:
            norms = np.linalg.norm(arr, axis=1, keepdims=True)
            arr = np.where(norms > 0, arr / norms, arr)
            
        return arr