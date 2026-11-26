"""Unit tests for WebSocket streaming chat endpoint."""

import json
from pathlib import Path
import sys
import unittest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from fastapi.testclient import TestClient

from app.main import app
from app.services.search_service import set_search_engine
from src.rag.llm_client import MockLLMClient
from src.search_engine import SearchEngine


class TestWebSocketChat(unittest.TestCase):
    """Test suite for /ws/chat WebSocket endpoint."""

    @classmethod
    def setUpClass(cls):
        dataset_path = PROJECT_ROOT / "dataset" / "restaurants.json"
        vector_dir = PROJECT_ROOT / "vector_db"
        mock_llm = MockLLMClient()
        cls.engine = SearchEngine(
            dataset_path=dataset_path,
            vector_dir=vector_dir,
            llm_client=mock_llm,
            load_vector_db=True,
        )
        set_search_engine(cls.engine)
        cls.client = TestClient(app)

    def test_websocket_connection_and_handshake(self):
        """Test that WebSocket connects and receives handshake event."""
        with self.client.websocket_connect("/ws/chat") as websocket:
            handshake = websocket.receive_json()
            self.assertEqual(handshake["type"], "connected")
            self.assertIn("Hybrid IR", handshake["message"])
            self.assertEqual(handshake["retrieval_system"], "Hybrid RRF (BM25 + FAISS)")

    def test_websocket_chat_streaming_flow(self):
        """Test full query-retrieve-stream flow over WebSocket."""
        with self.client.websocket_connect("/ws/chat") as websocket:
            # 1. Handshake
            handshake = websocket.receive_json()
            self.assertEqual(handshake["type"], "connected")

            # 2. Send query
            websocket.send_json({
                "query": "delicious noodle soup in Daun Penh",
                "top_k": 3,
                "temperature": 0.3
            })

            # 3. Expect status: searching
            status_1 = websocket.receive_json()
            self.assertEqual(status_1["type"], "status")
            self.assertEqual(status_1["stage"], "searching")

            # 4. Expect sources event with matching restaurant cards
            sources_event = websocket.receive_json()
            self.assertEqual(sources_event["type"], "sources")
            self.assertIn("data", sources_event)
            self.assertGreater(len(sources_event["data"]), 0)
            first_source = sources_event["data"][0]
            self.assertIn("name", first_source)
            self.assertIn("address", first_source)
            self.assertIn("score", first_source)

            # 5. Expect status: generating
            status_2 = websocket.receive_json()
            self.assertEqual(status_2["type"], "status")
            self.assertEqual(status_2["stage"], "generating")

            # 6. Expect chunks until done
            chunks = []
            while True:
                msg = websocket.receive_json()
                if msg["type"] == "chunk":
                    chunks.append(msg["content"])
                elif msg["type"] == "done":
                    done_event = msg
                    break
                else:
                    self.fail(f"Unexpected event type: {msg['type']}")

            # Verify response chunks and completion
            full_text = "".join(chunks)
            self.assertGreater(len(chunks), 0)
            self.assertTrue(len(full_text) > 10)
            self.assertIn("recommendations", full_text.lower())
            self.assertEqual(done_event["retrieval_method"], "Hybrid RRF")
            self.assertGreater(done_event["latency_ms"], 0.0)

    def test_websocket_empty_query_error(self):
        """Test sending an empty query returns an error message."""
        with self.client.websocket_connect("/ws/chat") as websocket:
            _ = websocket.receive_json()  # Handshake
            websocket.send_json({"query": "   "})
            error_msg = websocket.receive_json()
            self.assertEqual(error_msg["type"], "error")
            self.assertIn("Query cannot be empty", error_msg["message"])


if __name__ == "__main__":
    unittest.main()
