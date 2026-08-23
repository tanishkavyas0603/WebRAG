from pathlib import Path
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
        with open(self.metadata_path, "wb") as file:
            pickle.dump(chunks, file)
        logger.info(f"Metadata stored for document {self.document_id}.")

    def load(self):
        if not self.index_path.exists() or not self.metadata_path.exists():
            return None, None
            
        index = faiss.read_index(str(self.index_path))
        with open(self.metadata_path, "rb") as file:
            metadata = pickle.load(file)
            
        return index, metadata