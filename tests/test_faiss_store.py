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


def test_load_treats_corrupted_files_as_missing(tmp_path):
    """
    A truncated/corrupted index or metadata file (e.g. from an interrupted
    write during a Render restart) must be treated the same as "missing" —
    returning (None, None) so the caller's recovery path rebuilds it —
    instead of raising and crashing retrieval.
    """
    original_index_dir = constants.INDEX_DIR
    custom_index_dir = tmp_path / "corrupted_webrag_index"
    constants.INDEX_DIR = custom_index_dir

    try:
        doc_id = 998
        store = FAISSVectorStore(document_id=doc_id)

        # Write garbage bytes instead of valid FAISS/pickle content
        store.index_path.write_bytes(b"not a real faiss index")
        store.metadata_path.write_bytes(b"not real pickle data")

        index, metadata = store.load()
        assert index is None
        assert metadata is None
    finally:
        constants.INDEX_DIR = original_index_dir


def test_bm25_load_treats_corrupted_metadata_as_missing(tmp_path):
    """Same corruption-handling contract as FAISS, for the shared metadata file."""
    from app.services.bm25_service import BM25Service

    original_index_dir = constants.INDEX_DIR
    custom_index_dir = tmp_path / "corrupted_bm25_index"
    constants.INDEX_DIR = custom_index_dir

    try:
        doc_id = 997
        bm25 = BM25Service(document_id=doc_id)
        bm25.metadata_path.parent.mkdir(parents=True, exist_ok=True)
        bm25.metadata_path.write_bytes(b"not real pickle data")

        assert bm25.load() is False
        assert bm25.bm25 is None
        assert bm25.search("anything") == []
    finally:
        constants.INDEX_DIR = original_index_dir
