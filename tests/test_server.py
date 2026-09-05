"""Tests for the chat API and bundled Markdown-aware UI."""

import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient

import server


class _FakeAgent:
    def run(self, _message):
        return "### Result\n\nRead $\\rightarrow$ display"

    def get_audit_log(self):
        return []


class ServerRenderingTests(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(server.app)

    def test_chat_returns_plain_text_and_sanitized_html(self):
        with patch.object(server, "get_agent", return_value=_FakeAgent()):
            response = self.client.post(
                "/api/chat",
                json={"session_id": "test", "message": "Read file"},
            )

        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body["response"], "### Result\n\nRead $\\rightarrow$ display")
        self.assertIn("<h3>Result</h3>", body["response_html"])
        self.assertIn("Read → display", body["response_html"])

    def test_ui_inserts_only_server_sanitized_assistant_html(self):
        response = self.client.get("/")

        self.assertEqual(response.status_code, 200)
        self.assertIn("div.innerHTML = safeHtml", response.text)
        self.assertIn("data.response_html", response.text)
        self.assertIn("div.textContent = text", response.text)


if __name__ == "__main__":
    unittest.main()
