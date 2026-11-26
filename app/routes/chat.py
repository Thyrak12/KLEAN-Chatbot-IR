"""Chat and RAG recommendation API routes."""

from typing import Any, Dict, List, Optional
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from app.services.search_service import get_search_engine
from src.search_engine import SearchEngine

router = APIRouter(prefix="/api/v1", tags=["Conversational Chatbot (RAG)"])


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, description="User question or dining request")
    top_k: int = Field(default=3, ge=1, le=10, description="Number of source restaurants to ground the recommendation")
    temperature: float = Field(default=0.3, ge=0.0, le=1.0, description="Sampling temperature for LLM")


class SourceRestaurant(BaseModel):
    id: int
    name: str
    category: Optional[str] = None
    sub_category: Optional[List[str]] = []
    address: Optional[str] = None
    phone: Optional[str] = None
    website: Optional[str] = None


class ChatResponse(BaseModel):
    query: str
    answer: str
    sources: List[SourceRestaurant]
    num_sources: int
    latency_ms: float
    retrieval_method: str
    llm_provider: str


@router.post("/chat", response_model=ChatResponse, summary="Chat with KLEAN dining assistant")
def chat_endpoint(
    req: ChatRequest,
    engine: SearchEngine = Depends(get_search_engine),
):
    """Generate a conversational, grounded restaurant recommendation using RAG."""
    res = engine.chat(
        query=req.message,
        top_k=req.top_k,
        temperature=req.temperature,
    )

    sources = []
    for item in res.get("retrieved_restaurants", []):
        sources.append(
            SourceRestaurant(
                id=item["doc_id"],
                name=item.get("name", "Unknown"),
                category=item.get("category"),
                sub_category=item.get("sub_category", []),
                address=item.get("address"),
                phone=item.get("phone"),
                website=item.get("website"),
            )
        )

    return ChatResponse(
        query=res["query"],
        answer=res["answer"],
        sources=sources,
        num_sources=len(sources),
        latency_ms=round(res["latency_ms"], 2),
        retrieval_method=res.get("retrieval_method", "Hybrid RRF"),
        llm_provider=res.get("llm_provider", "Unknown"),
    )
