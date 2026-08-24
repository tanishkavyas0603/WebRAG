import os
from pathlib import Path
import pytest
import numpy as np

from app.vectorstore.faiss_store import FAISSVectorStore
from app.core import constants

def test_faiss_vector_store_initializes_with_temporary_directory(tmp_path):
    """
    Verifies that FAISSVectorStore creates the necessary directories
    even when using a custom/temporary writable path (e.g., Render /tmp).
    """
    # 1. Override the INDEX_DIR to use pytest's temporary directory
    original_index_dir = constants.INDEX_DIR
    custom_index_dir = tmp_path / "custom_webrag_index"
    constants.INDEX_DIR = custom_index_dir

    try:
        # 2. Ensure the directory doesn't exist yet
        assert not custom_index_dir.exists()

        # 3. Initialize FAISSVectorStore
        doc_id = 999
        store = FAISSVectorStore(document_id=doc_id)

        # 4. Verify that FAISSVectorStore created the directory
        assert custom_index_dir.exists()
        assert custom_index_dir.is_dir()

        # 5. Verify the internal paths
        assert store.index_path == custom_index_dir / f"doc_{doc_id}.faiss"
        assert store.metadata_path == custom_index_dir / f"doc_{doc_id}_metadata.pkl"

        # 6. Verify we can build and save to it
        dummy_embeddings = np.random.rand(5, 384).astype("float32")
        store.build(dummy_embeddings)
        store.save_metadata([{"text": "chunk1"}, {"text": "chunk2"}])

        # 7. Verify files actually exist on disk
        assert store.index_path.exists()
        assert store.metadata_path.exists()

    finally:
        # Restore original INDEX_DIR
        constants.INDEX_DIR = original_index_dir
