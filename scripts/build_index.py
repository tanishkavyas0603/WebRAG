from app.services.embedding_service import EmbeddingService
from app.vectorstore.faiss_store import FAISSVectorStore

if __name__ == "__main__":

    embedding_service = EmbeddingService()

    chunks = embedding_service.load_chunks()

    embeddings = embedding_service.generate_embeddings(chunks)

    vector_store = FAISSVectorStore()

    vector_store.build(embeddings)

    vector_store.save_metadata(chunks)

    print("FAISS index created successfully.")