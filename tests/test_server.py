"""Tests for protected APIs and the JWT-aware bundled UI."""

import unittest
import uuid
from unittest.mock import MagicMock, patch

from fastapi.testclient import TestClient

import server


TEST_USER = {
    "user_id": "user-1", "username": "tester", "display_name": "Tester",
    "role": "user", "is_active": True, "must_change_password": False,
    "permissions": [
        "chat:use", "conversation:read", "conversation:write",
        "audit:read", "memory:read", "memory:write", "drive:read",
    ],
}


class _FakeAgent:
    last_artifact = None

    def run(self, _message):
        return "### Result\n\nRead $\\rightarrow$ display"

    def get_audit_log(self):
        return []


class ServerRenderingTests(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(server.app)
        server.app.dependency_overrides[server.current_principal] = lambda: TEST_USER

    def tearDown(self):
        server.app.dependency_overrides.clear()
        server.sessions.clear()

    def test_protected_endpoint_requires_bearer_without_override(self):
        server.app.dependency_overrides.clear()
        response = self.client.get("/api/conversations")
        self.assertEqual(response.status_code, 401)

    def test_standard_user_cannot_call_admin_api(self):
        response = self.client.get("/api/admin/users")
        self.assertEqual(response.status_code, 403)

    def test_clear_checks_conversation_ownership(self):
        with patch.object(
            server.conversation_repository,
            "get_conversation",
            side_effect=server.ConversationNotFoundError("Conversation not found"),
        ):
            response = self.client.post(
                "/api/clear", json={"conversation_id": str(uuid.uuid4())}
            )
        self.assertEqual(response.status_code, 404)

    def test_chat_returns_plain_text_and_sanitized_html(self):
        conversation_id = str(uuid.uuid4())
        repository = MagicMock()
        repository.find_user_message.return_value = None
        repository.create_conversation_with_message.return_value = (
            {"id": conversation_id, "title": "Read file"},
            {"id": str(uuid.uuid4()), "ordinal": 1},
        )
        repository.append_assistant_message.return_value = {
            "id": str(uuid.uuid4()), "created_at": "2026-09-05T12:00:00+00:00",
        }
        with (
            patch.object(server, "conversation_repository", repository),
            patch.object(server, "_get_conversation_agent", return_value=_FakeAgent()),
        ):
            response = self.client.post(
                "/api/chat",
                json={"client_message_id": "message-1", "message": "Read file"},
            )
        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body["response"], "### Result\n\nRead $\\rightarrow$ display")
        self.assertIn("<h3>Result</h3>", body["response_html"])

    def test_memory_api_uses_authenticated_user(self):
        memories = [
            {"id": "fact-1", "text": "User likes Python", "metadata": {"category": "fact", "created_at": "2026-09-05T11:00:00+00:00"}},
            {"id": "document-1", "text": "Document", "metadata": {"category": "document"}},
        ]
        with patch.object(server, "list_all_memories", return_value=memories) as listing:
            response = self.client.get("/api/memories")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["total"], 1)
        listing.assert_called_once_with(
            limit=100, user_id="user-1", categories={"fact", "user_preference"}
        )

    def test_documents_api_groups_chunks_by_source(self):
        memories = [{
            "id": f"chunk-{index}", "text": f"Chunk {index}",
            "metadata": {"category": "document", "source_id": "source-1", "file_name": "assignment.pptx", "chunk_index": index, "chunk_count": 2},
        } for index in range(2)]
        with patch.object(server, "list_all_memories", return_value=memories):
            response = self.client.get("/api/documents")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["documents"][0]["stored_chunks"], 2)

    def test_audit_api_validates_conversation_ownership(self):
        conversation_id = str(uuid.uuid4())
        with (
            patch.object(server.conversation_repository, "get_conversation"),
            patch.object(server.audit_log_repository, "list_entries", return_value=[]) as listing,
        ):
            response = self.client.get(f"/api/audit?session_id={conversation_id}")
        self.assertEqual(response.status_code, 200)
        listing.assert_called_once_with(f"conversation:{conversation_id}", "user-1")

    def test_ui_contains_login_refresh_admin_and_safe_rendering(self):
        body = self.client.get("/").text
        self.assertIn('id="loginForm"', body)
        self.assertIn("/api/auth/refresh", body)
        self.assertIn('id="adminPanel"', body)
        self.assertIn("node.innerHTML = safeHtml", body)
        self.assertIn("node.textContent = text", body)


if __name__ == "__main__":
    unittest.main()
