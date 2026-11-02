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

    def _generate_embeddings_local(self, texts):
        try:
            logger.info(f"[EMBEDDING] Generating embeddings locally via FastEmbed for {len(texts)} texts...")
            from fastembed import TextEmbedding
            model = TextEmbedding(self.model)
            embeddings = list(model.embed(texts))
            arr = np.array(embeddings, dtype=np.float32)
            if len(arr) > 0 and len(arr.shape) == 2:
                norms = np.linalg.norm(arr, axis=1, keepdims=True)
                arr = np.where(norms > 0, arr / norms, arr)
            logger.info(f"[EMBEDDING_SUCCESS] Local FastEmbed generated {len(arr)} vectors")
            return arr
        except Exception as e:
            logger.error(f"[EMBEDDING_FAILURE] Local FastEmbed fallback failed: {e}")
            raise RuntimeError(f"Local embedding generation failed: {e}")

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
                    logger.info(f"[EMBEDDING_START] batch={batch_index}/{total_batches} attempt={attempt}/{max_retries} model={self.model}")
                    try:
                        response = client.post(self.api_url, headers=self.headers, json=payload)
                        logger.info(f"[EMBEDDING] HuggingFace response received: HTTP {response.status_code}")
                        
                        if response.status_code == 200:
                            batch_embeddings = response.json()
                            embeddings.extend(batch_embeddings)
                            logger.info(f"[EMBEDDING_SUCCESS] batch={batch_index}/{total_batches} vectors={len(batch_embeddings)}")
                            break
                            
                        elif response.status_code == 402:
                            logger.warning(
                                "[EMBEDDING] Hugging Face returned HTTP 402 (credit limit reached). "
                                "Falling back to local FastEmbed embedding generation."
                            )
                            return self._generate_embeddings_local(texts)
                            
                        elif response.status_code in (401, 403):
                            logger.error(
                                f"[EMBEDDING_FAILURE] status={response.status_code} "
                                "hint=Check_HF_TOKEN_env_var_may_be_invalid_or_missing. "
                                "Note: environment variables are loaded once at process startup — "
                                "if HF_TOKEN was just added/rotated (in .env or on Render), this "
                                "process must be restarted/redeployed to pick up the new value."
                            )
                            raise RuntimeError(f"Hugging Face authentication failed (HTTP {response.status_code}). Please check your HF_TOKEN.")
                            
                        elif response.status_code == 429:
                            logger.error(f"[EMBEDDING_FAILURE] status=429 reason=rate_limit model={self.model}")
                            raise RuntimeError("Hugging Face API rate limit exceeded (HTTP 429).")
                            
                        elif response.status_code == 503:
                            if attempt == max_retries:
                                logger.error(f"[EMBEDDING_FAILURE] status=503 reason=model_loading_timeout attempts={max_retries} model={self.model}")
                                raise RuntimeError("Hugging Face API is unavailable (Model loading failed after retries).")
                            backoff = 2 ** attempt
                            logger.info(f"[EMBEDDING] HF 503 model loading — waiting {backoff}s (attempt {attempt}/{max_retries})")
                            time.sleep(backoff)
                            continue
                            
                        else:
                            response.raise_for_status()
                            
                    except httpx.TimeoutException:
                        logger.error(f"[EMBEDDING_FAILURE] reason=timeout attempt={attempt}/{max_retries} model={self.model}")
                        if attempt == max_retries:
                            raise RuntimeError("Hugging Face API request timed out.")
                        backoff = 2 ** attempt
                        logger.info(f"[EMBEDDING] Timeout — waiting {backoff}s (attempt {attempt}/{max_retries})")
                        time.sleep(backoff)
                        
                    except httpx.RequestError as e:
                        logger.error(f"[EMBEDDING_FAILURE] reason=network_error type={type(e).__name__} attempt={attempt}/{max_retries}")
                        if attempt == max_retries:
                            raise RuntimeError(f"Embedding API network request failed: {str(e)}")
                        backoff = 2 ** attempt
                        logger.info(f"[EMBEDDING] Network error — waiting {backoff}s (attempt {attempt}/{max_retries})")
                        time.sleep(backoff)
                        
                    except httpx.HTTPStatusError as e:
                        logger.error(f"[EMBEDDING_FAILURE] reason=http_error status={e.response.status_code} model={self.model}")
                        raise RuntimeError(f"Embedding API failed with HTTP {e.response.status_code}: {e.response.text}")

        arr = np.array(embeddings, dtype=np.float32)
        if len(arr) > 0 and len(arr.shape) == 2:
            norms = np.linalg.norm(arr, axis=1, keepdims=True)
            arr = np.where(norms > 0, arr / norms, arr)
            
        return arr