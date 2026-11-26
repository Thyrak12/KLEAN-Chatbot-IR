"""Search service provider and dependency injection for FastAPI."""

import logging
from typing import Optional

from src.search_engine import SearchEngine

logger = logging.getLogger(__name__)

_search_engine_instance: Optional[SearchEngine] = None


def get_search_engine() -> SearchEngine:
    """Get or initialize singleton SearchEngine instance for FastAPI dependency injection."""
    global _search_engine_instance
    if _search_engine_instance is None:
        logger.info("Initializing global SearchEngine instance for API...")
        _search_engine_instance = SearchEngine()
    return _search_engine_instance


def set_search_engine(engine: SearchEngine) -> None:
    """Explicitly set the search engine instance (used in tests or custom setups)."""
    global _search_engine_instance
    _search_engine_instance = engine
