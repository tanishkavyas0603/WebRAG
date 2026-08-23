from fastapi import APIRouter

from app.models.request import QueryRequest
from app.models.response import QueryResponse
from app.services.rag_service import RAGService

router = APIRouter()

rag = RAGService()


@router.post(
    "/query",
    response_model=QueryResponse,
)
def query(
    request: QueryRequest,
):

    return rag.answer(request.question)