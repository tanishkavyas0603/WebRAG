import pytest
from unittest.mock import patch, MagicMock
from app.services.retrieval_service import RetrievalService
from app.models.db import Chunk

@pytest.fixture
def mock_dependencies():
    with patch('app.services.retrieval_service.FAISSVectorStore') as MockStore, \
         patch('app.services.retrieval_service.BM25Service') as MockBM25, \
         patch('app.services.retrieval_service.EmbeddingService') as MockEmbedding, \
         patch('app.services.retrieval_service.SessionLocal') as MockSession:
         
        yield {
            'store': MockStore,
            'bm25': MockBM25,
            'embedding': MockEmbedding,
            'session': MockSession
        }

def test_existing_faiss_index_normal_retrieval(mock_dependencies):
    store_instance = mock_dependencies['store'].return_value
    store_instance.load.return_value = (MagicMock(), [{"id": 1, "content": "test"}])
    
    # Should not raise
    service = RetrievalService(document_id=1)
    
    # Verify we loaded it
    assert store_instance.load.called
    # Verify we didn't try to query DB
    mock_dependencies['session'].assert_not_called()

def test_missing_faiss_index_chunks_exist_rebuilds(mock_dependencies):
    store_instance = mock_dependencies['store'].return_value
    # First load fails, second load (after rebuild) succeeds
    store_instance.load.side_effect = [
        (None, None),  # Initial load fails
        (None, None),  # Lock acquired, double-check fails
        (MagicMock(), [{"id": 1, "content": "test"}])  # Final load succeeds
    ]
    
    db_instance = mock_dependencies['session'].return_value.__enter__.return_value
    mock_chunk = Chunk(id=1, document_id=1, chunk_index=1, content="test content", metadata_={"title": "Test"})
    db_instance.query.return_value.filter.return_value.order_by.return_value.all.return_value = [mock_chunk]
    
    embed_instance = mock_dependencies['embedding'].return_value
    embed_instance.generate_embeddings.return_value = [[0.1, 0.2]]
    
    service = RetrievalService(document_id=1)
    
    # Verified we built it
    assert store_instance.build.called
    assert store_instance.save_metadata.called
    
    saved_metadata = store_instance.save_metadata.call_args[0][0]
    assert len(saved_metadata) == 1
    assert saved_metadata[0]["content"] == "test content"
    assert saved_metadata[0]["title"] == "Test"

def test_missing_faiss_index_no_chunks_fails(mock_dependencies):
    store_instance = mock_dependencies['store'].return_value
    store_instance.load.return_value = (None, None)
    
    db_instance = mock_dependencies['session'].return_value.__enter__.return_value
    db_instance.query.return_value.filter.return_value.order_by.return_value.all.return_value = []
    
    with pytest.raises(RuntimeError, match="needs to be re-ingested. No chunks found."):
        RetrievalService(document_id=1)

def test_concurrent_recovery_does_not_corrupt_index(mock_dependencies):
    import threading
    import time
    
    store_instance = mock_dependencies['store'].return_value
    
    # Let's simulate that store_instance.load takes a bit, so thread 1 acquires the lock
    build_calls = 0
    
    def mock_build(*args, **kwargs):
        nonlocal build_calls
        build_calls += 1
        time.sleep(0.1) # Simulate time taken to build
        
    store_instance.build.side_effect = mock_build
    
    # State machine for load:
    # T1: load -> None, lock -> load -> None, build, load -> Index
    # T2: load -> None, blocked on lock. Once T1 finishes, T2 lock -> load -> Index -> bypass build!
    def mock_load():
        if build_calls == 0:
            return (None, None)
        return (MagicMock(), [{"id": 1}])
        
    store_instance.load.side_effect = mock_load
    
    db_instance = mock_dependencies['session'].return_value.__enter__.return_value
    mock_chunk = Chunk(id=1, document_id=1, chunk_index=1, content="test", metadata_={"title": "Test"})
    db_instance.query.return_value.filter.return_value.order_by.return_value.all.return_value = [mock_chunk]
    
    def init_service():
        RetrievalService(document_id=1)
        
    t1 = threading.Thread(target=init_service)
    t2 = threading.Thread(target=init_service)
    
    t1.start()
    t2.start()
    
    t1.join()
    t2.join()
    
    # Both threads succeeded, but build was only called ONCE!
    assert build_calls == 1
