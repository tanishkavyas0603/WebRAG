from app.services.retrieval_service import RetrievalService

service = RetrievalService()

results = service.search(
    "What is Ayushman Bharat?"
)

for result in results:
    print("-" * 60)
    print(f"Title      : {result.title}")
    print(f"Similarity : {result.similarity_score:.4f}")
    print(f"Content:\n{result.content[:400]}...")