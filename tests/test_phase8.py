"""Unit tests for Phase 8: FastAPI Backend Integration."""

import unittest
from fastapi.testclient import TestClient

from app.main import app


class TestPhase8FastAPIBackend(unittest.TestCase):
    """Test suite for FastAPI REST endpoints."""

    @classmethod
    def setUpClass(cls):
        # Create TestClient once
        cls.client = TestClient(app)

    def test_root_endpoint(self):
        response = self.client.get("/")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["app"], "KLEAN Chatbot API")
        self.assertEqual(data["status"], "online")
        self.assertIn("endpoints", data)

    def test_health_check(self):
        response = self.client.get("/health")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["status"], "healthy")

    def test_web_ui_endpoints(self):
        # Explicit /ui route
        ui_res = self.client.get("/ui")
        self.assertEqual(ui_res.status_code, 200)
        self.assertIn("text/html", ui_res.headers.get("content-type", ""))
        self.assertIn("KLEAN Chatbot", ui_res.text)

        # Browser visiting / with Accept: text/html
        browser_res = self.client.get(
            "/",
            headers={
                "Accept": "text/html,application/xhtml+xml",
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/120.0",
            },
        )
        self.assertEqual(browser_res.status_code, 200)
        self.assertIn("text/html", browser_res.headers.get("content-type", ""))
        self.assertIn("KLEAN Chatbot", browser_res.text)

    def test_search_get_hybrid(self):
        response = self.client.get("/api/v1/search?q=pizza&algorithm=hybrid&top_k=3")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["query"], "pizza")
        self.assertEqual(data["algorithm"], "hybrid")
        self.assertLessEqual(data["num_results"], 3)
        self.assertTrue(len(data["results"]) > 0)
        first = data["results"][0]
        self.assertIn("id", first)
        self.assertIn("name", first)
        self.assertIn("score", first)
        self.assertIn("rank", first)

    def test_search_get_bm25(self):
        response = self.client.get("/api/v1/search?q=noodle&algorithm=bm25&top_k=2")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["algorithm"], "bm25")
        self.assertEqual(len(data["results"]), 2)

    def test_search_post_with_filters(self):
        payload = {
            "query": "soup",
            "algorithm": "hybrid",
            "top_k": 5,
            "category": "Restaurant",
        }
        response = self.client.post("/api/v1/search", json=payload)
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["query"], "soup")
        self.assertTrue(len(data["results"]) > 0)

    def test_search_invalid_algorithm(self):
        response = self.client.get("/api/v1/search?q=soup&algorithm=nonexistent")
        self.assertEqual(response.status_code, 400)
        self.assertIn("Unknown algorithm", response.json()["detail"])

    def test_get_restaurant_by_id_success(self):
        response = self.client.get("/api/v1/restaurants/1")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["id"], 1)
        self.assertIn("name", data)
        self.assertIn("category", data)

    def test_get_restaurant_by_id_not_found(self):
        response = self.client.get("/api/v1/restaurants/99999999")
        self.assertEqual(response.status_code, 404)
        self.assertIn("not found", response.json()["detail"].lower())

    def test_get_stats(self):
        response = self.client.get("/api/v1/stats")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["total_restaurants"], 698)
        self.assertGreater(data["vocabulary_size"], 1000)
        self.assertTrue(data["rag_available"])

    def test_chat_endpoint_rag(self):
        payload = {
            "message": "Where can I get good noodles in Phnom Penh?",
            "top_k": 2,
            "temperature": 0.2,
        }
        response = self.client.post("/api/v1/chat", json=payload)
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["query"], payload["message"])
        self.assertTrue(len(data["answer"]) > 0)
        self.assertTrue(len(data["sources"]) > 0)
        self.assertIn("llm_provider", data)


if __name__ == "__main__":
    unittest.main()
