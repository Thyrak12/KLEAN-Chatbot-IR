"""Search API routes for restaurant retrieval."""

import time
from typing import Any, Dict, List, Optional
from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field

from app.services.search_service import get_search_engine
from src.search_engine import SearchEngine

router = APIRouter(prefix="/api/v1", tags=["Search & Restaurants"])


class RestaurantModel(BaseModel):
    id: int = Field(..., description="Unique restaurant ID")
    name: str = Field(..., description="Restaurant name")
    category: str = Field(..., description="Primary category")
    sub_category: Optional[List[str]] = Field(default=[], description="Cuisines / Subcategories")
    address: Optional[str] = Field(None, description="Full address in Phnom Penh")
    phone: Optional[str] = Field(None, description="Contact phone number")
    website: Optional[str] = Field(None, description="Website / Facebook page")
    coordinates: Optional[Dict[str, Optional[float]]] = Field(None, description="Latitude and Longitude")


class RankedRestaurantResult(RestaurantModel):
    score: float = Field(..., description="Retrieval similarity or fusion score")
    rank: int = Field(..., description="1-based ranking position")
    bm25_rank: Optional[int] = Field(None, description="Rank in BM25 results if retrieved")
    vector_rank: Optional[int] = Field(None, description="Rank in Vector search if retrieved")


class SearchRequest(BaseModel):
    query: str = Field(..., min_length=1, description="Natural language search query")
    algorithm: str = Field(default="hybrid", description="Algorithm: 'hybrid', 'bm25', 'tfidf', 'vector', 'boolean'")
    top_k: int = Field(default=10, ge=1, le=50, description="Number of results to retrieve")
    category: Optional[str] = Field(None, description="Filter by category or cuisine keyword")
    district: Optional[str] = Field(None, description="Filter by district/neighborhood")


class SearchResponse(BaseModel):
    query: str
    algorithm: str
    num_results: int
    latency_ms: float
    results: List[RankedRestaurantResult]


@router.get("/search", response_model=SearchResponse, summary="Search restaurants (GET query)")
def search_get(
    q: str = Query(..., min_length=1, description="Search query string"),
    algorithm: str = Query(default="hybrid", description="Algorithm: 'hybrid', 'bm25', 'tfidf', 'vector', 'boolean'"),
    top_k: int = Query(default=10, ge=1, le=50),
    category: Optional[str] = Query(None, description="Category filter"),
    district: Optional[str] = Query(None, description="District filter"),
    engine: SearchEngine = Depends(get_search_engine),
):
    """Retrieve ranked restaurants matching a query string."""
    t0 = time.perf_counter()
    try:
        raw_results = engine.search(
            query=q,
            algorithm=algorithm,
            top_k=top_k,
            category=category,
            district=district,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    latency_ms = (time.perf_counter() - t0) * 1000.0

    formatted_results = []
    for item in raw_results:
        formatted_results.append(
            RankedRestaurantResult(
                id=item["doc_id"],
                name=item["name"],
                category=item["category"],
                sub_category=item.get("sub_category", []),
                address=item.get("address"),
                phone=item.get("phone"),
                website=item.get("website"),
                coordinates=item.get("coordinates"),
                score=item["score"],
                rank=item["rank"],
                bm25_rank=item.get("bm25_rank"),
                vector_rank=item.get("vector_rank"),
            )
        )

    return SearchResponse(
        query=q,
        algorithm=algorithm,
        num_results=len(formatted_results),
        latency_ms=round(latency_ms, 2),
        results=formatted_results,
    )


@router.post("/search", response_model=SearchResponse, summary="Search restaurants (POST payload)")
def search_post(
    req: SearchRequest,
    engine: SearchEngine = Depends(get_search_engine),
):
    """Retrieve ranked restaurants using JSON request body with filtering."""
    t0 = time.perf_counter()
    try:
        raw_results = engine.search(
            query=req.query,
            algorithm=req.algorithm,
            top_k=req.top_k,
            category=req.category,
            district=req.district,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    latency_ms = (time.perf_counter() - t0) * 1000.0

    formatted_results = []
    for item in raw_results:
        formatted_results.append(
            RankedRestaurantResult(
                id=item["doc_id"],
                name=item["name"],
                category=item["category"],
                sub_category=item.get("sub_category", []),
                address=item.get("address"),
                phone=item.get("phone"),
                website=item.get("website"),
                coordinates=item.get("coordinates"),
                score=item["score"],
                rank=item["rank"],
                bm25_rank=item.get("bm25_rank"),
                vector_rank=item.get("vector_rank"),
            )
        )

    return SearchResponse(
        query=req.query,
        algorithm=req.algorithm,
        num_results=len(formatted_results),
        latency_ms=round(latency_ms, 2),
        results=formatted_results,
    )


@router.get("/restaurants/{restaurant_id}", response_model=RestaurantModel, summary="Get restaurant by ID")
def get_restaurant_by_id(
    restaurant_id: int,
    engine: SearchEngine = Depends(get_search_engine),
):
    """Retrieve a specific restaurant record by its unique database ID."""
    rec = engine.get_restaurant(restaurant_id)
    if not rec:
        raise HTTPException(status_code=404, detail=f"Restaurant with ID {restaurant_id} not found.")

    return RestaurantModel(
        id=rec["id"],
        name=rec["name"],
        category=rec["category"],
        sub_category=rec.get("sub_category", []),
        address=rec.get("address"),
        phone=rec.get("phone"),
        website=rec.get("website"),
        coordinates=rec.get("coordinates"),
    )


@router.get("/stats", summary="Get search engine & corpus statistics")
def get_stats(engine: SearchEngine = Depends(get_search_engine)):
    """Return index counts, vocabulary size, and available models."""
    return engine.get_statistics()
