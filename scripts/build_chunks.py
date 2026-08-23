from app.services.chunking_service import ChunkingService


if __name__ == "__main__":

    service = ChunkingService()

    chunks = service.run()

    print(f"\nCreated {len(chunks)} chunks.")