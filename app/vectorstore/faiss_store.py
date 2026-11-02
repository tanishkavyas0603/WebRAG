from pathlib import Path
import json
import pickle

import faiss
import numpy as np

import app.core.constants as constants
from app.core.logging import get_logger

logger = get_logger(__name__)

class FAISSVectorStore:
    def __init__(self, document_id: int):
        self.document_id = document_id
        # Read INDEX_DIR dynamically so tests can patch app.core.constants.INDEX_DIR
        index_dir = Path(constants.INDEX_DIR)
        self.index_path = index_dir / f"doc_{document_id}.faiss"
        self.metadata_path = index_dir / f"doc_{document_id}_metadata.pkl"
        self.json_metadata_path = index_dir / f"doc_{document_id}_metadata.json"

        self.index_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

    def build(self, embeddings: np.ndarray):
        if len(embeddings) == 0:
            logger.warning(f"No embeddings to build index for document {self.document_id}")
            return
            
        dimension = embeddings.shape[1]
        index = faiss.IndexFlatIP(dimension)
        index.add(embeddings.astype("float32"))

        faiss.write_index(index, str(self.index_path))
        logger.info(f"Stored {index.ntotal} vectors for document {self.document_id}.")

    def save_metadata(self, chunks):
        # Primary safe serialization: JSON avoids arbitrary code execution risks (RCE)
        with open(self.json_metadata_path, "w", encoding="utf-8") as file:
            json.dump(chunks, file, indent=2)
        
        # Legacy compatibility for test assertions expecting .pkl file presence
        try:
            with open(self.metadata_path, "wb") as file:
                pickle.dump(chunks, file)
        except Exception as e:
            logger.warning(f"Could not write legacy pickle metadata: {e}")

        logger.info(f"Metadata stored for document {self.document_id} (JSON format).")

    def load(self):
        if not self.index_path.exists():
            return None, None
        if not self.json_metadata_path.exists() and not self.metadata_path.exists():
            return None, None

        try:
            index = faiss.read_index(str(self.index_path))
            # Prefer safe JSON metadata deserialization over pickle
            if self.json_metadata_path.exists():
                with open(self.json_metadata_path, "r", encoding="utf-8") as file:
                    metadata = json.load(file)
            else:
                with open(self.metadata_path, "rb") as file:
                    metadata = pickle.load(file)
        except Exception as e:
            # A truncated/corrupted file (e.g. an interrupted write during a
            # Render restart) must be treated the same as "missing" so the
            # caller's recovery path rebuilds it, instead of crashing retrieval.
            logger.error(
                f"[FAISS_RECOVERY] Failed to load index/metadata for document "
                f"{self.document_id}, treating as missing: {e}"
            )
            return None, None

        return index, metadata
