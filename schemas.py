from pydantic import BaseModel
from typing import Optional, List

class Document(BaseModel):
    id: Optional[int] = None
    title: str
    content: str


# Input para /upload
class DocumentUploadRequest(BaseModel):
    title: str
    content: str

# Output para /upload
class DocumentUploadResponse(BaseModel):
    message: str
    document_id: str
    

# Input para /generate-embeddings
class GenerateEmbeddingsRequest(BaseModel):
    document_id: str

# Output para /generate-embeddings
class GenerateEmbeddingsResponse(BaseModel):
    message: str


# Input para /search
class SearchRequest(BaseModel):
    query: str

class SearchResultItem(BaseModel):
    document_id: str
    title: str
    content_snippet: str
    similarity_score: float

# Output (Lista de resultados)
class SearchResponse(BaseModel):
    results: List[SearchResultItem]


# Input para /ask
class AskQuestionRequest(BaseModel):
    question: str

class AskQuestionResponse(BaseModel):
    answer: str
    context_used: str
    similarity_score: float
    grounded: bool

