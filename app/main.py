"""FastAPI application entry point for KLEAN Chatbot.

Provides RESTful API endpoints for:
  - Ranked Restaurant Retrieval (/api/v1/search)
  - Conversational Dining Recommendations (/api/v1/chat)
  - Restaurant Details & Statistics (/api/v1/restaurants, /api/v1/stats)
"""

from contextlib import asynccontextmanager
import logging
from pathlib import Path
import sys

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv
load_dotenv(PROJECT_ROOT / ".env")

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

from app.routes.chat import router as chat_router
from app.routes.search import router as search_router
from app.routes.ws_chat import router as ws_chat_router
from app.services.search_service import get_search_engine

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("app.main")

STATIC_DIR = Path(__file__).resolve().parent / "static"
INDEX_HTML_PATH = STATIC_DIR / "index.html"


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager to preload SearchEngine on server startup."""
    logger.info("Preloading SearchEngine & Inverted Index on startup...")
    _ = get_search_engine()
    logger.info("SearchEngine ready to handle requests.")
    yield
    logger.info("Shutting down application...")


app = FastAPI(
    title="KLEAN Chatbot — Restaurant Search & Recommendation API",
    description=(
        "Production-grade Information Retrieval (IR) and Retrieval-Augmented Generation (RAG) "
        "REST API for discovering restaurants, local eats, and cafes in Phnom Penh, Cambodia."
    ),
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

# Enable Cross-Origin Resource Sharing (CORS) for web frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register route modules
app.include_router(search_router)
app.include_router(chat_router)
app.include_router(ws_chat_router)

# Mount static files
if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/ui", response_class=HTMLResponse, tags=["Web Interface"])
def web_ui():
    """Serve the interactive Web UI."""
    if INDEX_HTML_PATH.exists():
        with open(INDEX_HTML_PATH, "r", encoding="utf-8") as f:
            return HTMLResponse(content=f.read())
    return HTMLResponse("<h2>KLEAN Chatbot Web UI not found</h2>", status_code=404)


@app.get("/", tags=["System"])
def root(request: Request):
    """Root endpoint: serves interactive Web UI in browsers, or JSON metadata for API clients."""
    accept_header = request.headers.get("accept", "")
    user_agent = request.headers.get("user-agent", "").lower()

    # If accessed by a real web browser (asking for text/html and not testclient/curl)
    if "text/html" in accept_header and "testclient" not in user_agent and INDEX_HTML_PATH.exists():
        with open(INDEX_HTML_PATH, "r", encoding="utf-8") as f:
            return HTMLResponse(content=f.read())

    return {
        "app": "KLEAN Chatbot API",
        "version": "1.0.0",
        "status": "online",
        "documentation": "/docs",
        "web_ui": "/ui",
        "endpoints": {
            "search_get": "/api/v1/search?q={query}&algorithm={hybrid|bm25|tfidf|vector|boolean}",
            "search_post": "/api/v1/search",
            "chat": "/api/v1/chat",
            "restaurant_by_id": "/api/v1/restaurants/{id}",
            "stats": "/api/v1/stats",
            "health": "/health",
        },
    }


@app.get("/health", tags=["System"])
def health():
    """Health check endpoint for container and uptime monitoring."""
    return {"status": "healthy", "service": "klean-chatbot-api"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
