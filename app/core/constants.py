from pathlib import Path
from app.core.config import settings

DATA_DIR = Path(settings.DATA_PATH)

RAW_DATA_DIR = DATA_DIR / "raw"

PROCESSED_DATA_DIR = DATA_DIR / "processed"

INDEX_DIR = Path(settings.FAISS_INDEX_DIR)

DEFAULT_CHUNK_SIZE = 350

DEFAULT_CHUNK_OVERLAP = 50