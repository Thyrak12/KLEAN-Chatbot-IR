"""WebSocket streaming chat route for KLEAN Chatbot.

Provides real-time token streaming and intermediate status updates over WebSockets,
grounded on our final Hybrid Retrieval (BM25 + FAISS Vector Search via RRF) and LLM.
"""

import asyncio
import json
import logging
import time
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from app.services.search_service import get_search_engine

logger = logging.getLogger(__name__)

router = APIRouter(tags=["WebSocket Chat"])


@router.websocket("/ws/chat")
async def websocket_chat_endpoint(websocket: WebSocket):
    """Real-time ChatGPT-style conversational dining recommendation over WebSocket.

    Protocol:
      Client sends JSON:
        {"query": "best noodle soup in BKK1", "top_k": 3, "temperature": 0.3}
      Server streams JSON events:
        - {"type": "connected", "message": "..."}
        - {"type": "status", "stage": "searching", "message": "..."}
        - {"type": "sources", "data": [...], "retrieval_method": "Hybrid RRF"}
        - {"type": "status", "stage": "generating", "message": "..."}
        - {"type": "chunk", "content": "..."}
        - {"type": "done", "latency_ms": 782.4, ...}
        - {"type": "error", "message": "..."}
    """
    await websocket.accept()
    engine = get_search_engine()

    # Send initial handshake message
    await websocket.send_json({
        "type": "connected",
        "message": "Connected to KLEAN Chatbot (Hybrid IR + DeepSeek)",
        "retrieval_system": "Hybrid RRF (BM25 + FAISS)",
        "active_llm": engine.llm_client.__class__.__name__,
    })

    try:
        while True:
            raw_data = await websocket.receive_text()
            if not raw_data.strip():
                continue

            try:
                payload = json.loads(raw_data)
            except Exception:
                # Fallback if raw text was sent
                payload = {"query": raw_data.strip()}

            query = (payload.get("query") or payload.get("message") or "").strip()
            top_k = int(payload.get("top_k", 3))
            temperature = float(payload.get("temperature", 0.3))

            if not query:
                await websocket.send_json({
                    "type": "error",
                    "message": "Query cannot be empty. Please specify a cuisine, dish, or neighborhood.",
                })
                continue

            t_start = time.perf_counter()

            # 1. Status: Searching
            await websocket.send_json({
                "type": "status",
                "stage": "searching",
                "message": "Retrieving candidate restaurants with Hybrid IR (BM25 + FAISS RRF)...",
            })

            try:
                # 2. Run Hybrid Retrieval & Initialize Stream (in thread pool to avoid blocking async loop)
                sources, stream_gen, meta = await asyncio.to_thread(
                    engine.stream_chat,
                    query=query,
                    top_k=top_k,
                    temperature=temperature,
                )

                # Format clean source cards
                source_cards: List[Dict[str, Any]] = []
                for s in sources:
                    source_cards.append({
                        "id": s["doc_id"],
                        "name": s.get("name", "Unknown"),
                        "category": s.get("category", ""),
                        "sub_category": s.get("sub_category", []),
                        "address": s.get("address", ""),
                        "phone": s.get("phone"),
                        "website": s.get("website"),
                        "latitude": s.get("latitude"),
                        "longitude": s.get("longitude"),
                        "score": round(float(s.get("score", 0.0)), 4),
                    })

                # 3. Emit Source Cards
                await websocket.send_json({
                    "type": "sources",
                    "data": source_cards,
                    "num_sources": len(source_cards),
                    "retrieval_method": meta.get("retrieval_method", "Hybrid RRF"),
                    "retrieval_latency_ms": round(float(meta.get("retrieval_latency_ms", 0.0)), 2),
                })

                # 4. Status: Generating
                await websocket.send_json({
                    "type": "status",
                    "stage": "generating",
                    "message": "Generating recommendation with LLM...",
                })

                # Helper to advance generator safely in thread
                def _get_next():
                    try:
                        return next(stream_gen)
                    except StopIteration:
                        return None

                # 5. Stream Tokens
                while True:
                    chunk = await asyncio.to_thread(_get_next)
                    if chunk is None:
                        break
                    await websocket.send_json({
                        "type": "chunk",
                        "content": chunk,
                    })
                    # Brief yield for event loop socket buffer flushing
                    await asyncio.sleep(0.005)

                # 6. Complete
                total_latency_ms = (time.perf_counter() - t_start) * 1000.0
                await websocket.send_json({
                    "type": "done",
                    "query": query,
                    "latency_ms": round(total_latency_ms, 2),
                    "retrieval_method": meta.get("retrieval_method", "Hybrid RRF"),
                    "llm_provider": meta.get("llm_provider", "Unknown"),
                })

            except Exception as e:
                logger.error("Error during WebSocket streaming chat: %s", e, exc_info=True)
                await websocket.send_json({
                    "type": "error",
                    "message": f"Generation error: {str(e)}",
                })

    except WebSocketDisconnect:
        logger.info("WebSocket chat client disconnected.")
    except Exception as e:
        logger.error("WebSocket unexpected connection error: %s", e)
