from app.services.rag_service import RAGService

service = RAGService()

response = service.answer(
    "What is Ayushman Bharat PM-JAY?"
)

print("\nANSWER\n")
print(response["answer"])

print("\nSOURCES\n")

for source in response["sources"]:
    print(source)