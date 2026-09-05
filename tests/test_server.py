"""Tests for the chat API and bundled Markdown-aware UI."""

import unittest
from types import SimpleNamespace
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

    def test_memory_api_returns_only_user_facts_and_preferences(self):
        memories = [
            {
                "id": "preference-1",
                "text": "Người dùng thích Python.",
                "metadata": {
                    "category": "user_preference",
                    "created_at": "2026-09-05T10:00:00+00:00",
                },
            },
            {
                "id": "document-1",
                "text": "Document chunk",
                "metadata": {"category": "document"},
            },
            {
                "id": "fact-1",
                "text": "Tên người dùng là An.",
                "metadata": {
                    "category": "fact",
                    "created_at": "2026-09-05T11:00:00+00:00",
                },
            },
        ]
        fake_agent = SimpleNamespace(service_api_key="sk-admin-001")

        with (
            patch.object(server, "get_agent", return_value=fake_agent),
            patch.object(server, "list_all_memories", return_value=memories) as listing,
        ):
            response = self.client.get("/api/memories?session_id=test")

        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body["total"], 2)
        self.assertEqual(body["facts"][0]["category"], "fact")
        self.assertEqual(body["facts"][1]["category"], "user_preference")
        listing.assert_called_once_with(
            limit=100,
            user_id="user_admin",
            categories={"fact", "user_preference"},
        )

    def test_ui_has_memory_panel_and_uses_text_content_for_facts(self):
        response = self.client.get("/")

        self.assertEqual(response.status_code, 200)
        self.assertIn('id="memoryPanel"', response.text)
        self.assertIn("/api/memories?session_id=", response.text)
        self.assertIn("content.textContent = fact.text", response.text)
        self.assertNotIn("content.innerHTML = fact.text", response.text)

    def test_documents_api_groups_rag_chunks_by_source(self):
        memories = [
            {
                "id": f"chunk-{index}",
                "text": f"Chunk {index}",
                "metadata": {
                    "category": "document",
                    "source_id": "source-1",
                    "file_id": "drive-1",
                    "file_name": "assignment.pptx",
                    "content_hash": "hash-1",
                    "chunk_index": index,
                    "chunk_count": 2,
                    "created_at": "2026-09-05T12:00:00+00:00",
                },
            }
            for index in range(2)
        ]
        fake_agent = SimpleNamespace(service_api_key="sk-admin-001")
        with (
            patch.object(server, "get_agent", return_value=fake_agent),
            patch.object(server, "list_all_memories", return_value=memories),
        ):
            response = self.client.get("/api/documents?session_id=test")

        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body["total"], 1)
        self.assertEqual(body["documents"][0]["file_name"], "assignment.pptx")
        self.assertEqual(body["documents"][0]["stored_chunks"], 2)

    def test_ui_persists_session_and_exposes_documents_panel(self):
        response = self.client.get("/")

        self.assertEqual(response.status_code, 200)
        self.assertIn("localStorage.getItem(SESSION_STORAGE_KEY)", response.text)
        self.assertIn('id="documentPanel"', response.text)
        self.assertIn("/api/documents?session_id=", response.text)
        self.assertIn("title.textContent = documentMemory.file_name", response.text)

    def test_audit_api_reads_persisted_conversation_entries(self):
        conversation_id = "72f0ae68-c8bb-4c56-a12c-7811199f87ca"
        persisted = [{"tool": "echo", "status": "success", "steps": []}]
        repository = unittest.mock.MagicMock()
        repository.list_entries.return_value = persisted

        with (
            patch.object(server, "audit_log_repository", repository),
            patch.object(server.conversation_repository, "get_conversation"),
        ):
            response = self.client.get(
                f"/api/audit?session_id={conversation_id}"
            )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["audit_log"], persisted)
        repository.list_entries.assert_called_once_with(
            f"conversation:{conversation_id}",
            "user_admin",
        )


if __name__ == "__main__":
    unittest.main()
