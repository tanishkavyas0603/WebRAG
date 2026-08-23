from pydantic import BaseModel, Field


class DocumentChunk(BaseModel):
    """
    Represents one semantic chunk of the document.
    """

    id: int = Field(..., description="Unique chunk ID")

    title: str = Field(..., description="Chunk title")

    section: str = Field(..., description="Document section")

    preview: str = Field(..., description="Preview shown in UI")

    content: str = Field(..., description="Complete chunk")

    source: str = Field(default="PIB")

    chunk_number: int