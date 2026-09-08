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

    def __init__(self):
        self.principal = {}

    def run(self, _message):
        return "### Result\n\nRead $\\rightarrow$ display"

    def get_audit_log(self):
        return []


class ServerRenderingTests(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(server.app)
        server.app.dependency_overrides[server.current_principal] = lambda: TEST_USER
        server.chat_usage_limiter.clear()

    def tearDown(self):
        server.app.dependency_overrides.clear()
        server.sessions.clear()
        server.session_access.clear()
        server.conversation_locks.clear()
        server.conversation_lock_states.clear()
        server.chat_usage_limiter.clear()

    def test_protected_endpoint_requires_bearer_without_override(self):
        server.app.dependency_overrides.clear()
        response = self.client.get("/api/conversations")
        self.assertEqual(response.status_code, 401)

    def test_standard_user_cannot_call_admin_api(self):
        response = self.client.get("/api/admin/users")
        self.assertEqual(response.status_code, 403)

    def test_google_login_start_sets_browser_binding_cookie(self):
        authorization = {
            "authorization_url": "https://accounts.google.test/authorize",
            "browser_binding": "browser-secret",
        }
        with patch.object(
            server.google_oauth_service, "begin", return_value=authorization
        ) as begin:
            response = self.client.get(
                "/api/auth/google/start", follow_redirects=False
            )
        self.assertEqual(response.status_code, 302)
        self.assertEqual(response.headers["location"], authorization["authorization_url"])
        self.assertIn(server.GOOGLE_OAUTH_BINDING_COOKIE, response.cookies)
        begin.assert_called_once_with(mode="login")

    def test_drive_authorization_is_bound_to_authenticated_user(self):
        authorization = {
            "authorization_url": "https://accounts.google.test/drive",
            "browser_binding": "browser-secret",
        }
        with patch.object(
            server.google_oauth_service, "begin", return_value=authorization
        ) as begin:
            response = self.client.post(
                "/api/integrations/google-drive/authorize"
            )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(
            response.json()["authorization_url"],
            authorization["authorization_url"],
        )
        self.assertIn(server.GOOGLE_OAUTH_BINDING_COOKIE, response.cookies)
        begin.assert_called_once_with(mode="drive", user_id="user-1")

    def test_google_callback_logs_oauth_failure_without_callback_url(self):
        with (
            patch.object(
                server.google_oauth_service,
                "complete",
                side_effect=server.GoogleOAuthError("token exchange failed"),
            ),
            patch.object(server.logger, "exception") as log_exception,
        ):
            response = self.client.get(
                "/api/auth/google/callback?state=test-state&code=secret-code",
                follow_redirects=False,
            )

        self.assertEqual(response.status_code, 303)
        self.assertEqual(response.headers["location"], "/?oauth=error")
        log_exception.assert_called_once()
        rendered_log_arguments = " ".join(
            str(value) for value in log_exception.call_args.args
        )
        self.assertIn("token exchange failed", rendered_log_arguments)
        self.assertNotIn("secret-code", rendered_log_arguments)

    def test_cached_agent_receives_current_principal(self):
        conversation_id = str(uuid.uuid4())
        key = server._agent_key("user-1", conversation_id)
        cached = _FakeAgent()
        cached.principal = {"user_id": "user-1", "permissions": ["memory:write"]}
        server.sessions[key] = cached
        downgraded = {**TEST_USER, "permissions": ["chat:use"]}

        result = server._get_conversation_agent(conversation_id, downgraded)

        self.assertIs(result, cached)
        self.assertEqual(result.principal["permissions"], ["chat:use"])

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
        self.assertIn("x-ratelimit-remaining", response.headers)
        self.assertIn("x-tokenquota-remaining", response.headers)

    def test_rejects_oversized_request_body_before_json_parsing(self):
        response = self.client.post(
            "/api/chat",
            content=b"x" * (server.MAX_REQUEST_BODY_BYTES + 1),
            headers={"Content-Type": "application/json"},
        )
        self.assertEqual(response.status_code, 413)

    def test_rejects_chat_message_above_configured_character_limit(self):
        response = self.client.post(
            "/api/chat",
            json={"message": "x" * (server.CHAT_MESSAGE_MAX_CHARS + 1)},
        )
        self.assertEqual(response.status_code, 422)

    def test_chat_rate_limit_returns_429_and_retry_after(self):
        conversation_id = str(uuid.uuid4())
        repository = MagicMock()
        repository.find_user_message.return_value = None
        repository.create_conversation_with_message.return_value = (
            {"id": conversation_id, "title": "Hello"},
            {"id": str(uuid.uuid4()), "ordinal": 1},
        )
        repository.append_assistant_message.return_value = {
            "id": str(uuid.uuid4()), "created_at": "2026-09-05T12:00:00+00:00",
        }
        limiter = server.ChatUsageLimiter(
            max_requests=1,
            window_seconds=60,
            daily_token_quota=100_000,
            base_token_charge=1,
        )
        with (
            patch.object(server, "chat_usage_limiter", limiter),
            patch.object(server, "conversation_repository", repository),
            patch.object(server, "_get_conversation_agent", return_value=_FakeAgent()),
        ):
            first = self.client.post("/api/chat", json={"message": "Hello"})
            second = self.client.post("/api/chat", json={"message": "Again"})
        self.assertEqual(first.status_code, 200)
        self.assertEqual(second.status_code, 429)
        self.assertIn("Retry-After", second.headers)

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
        self.assertIn('id="loginError" role="alert" aria-live="polite"', body)
        self.assertIn("/api/auth/refresh", body)
        self.assertIn("Phiên đăng nhập đã hết hạn", body)
        self.assertIn("Tên đăng nhập hoặc mật khẩu không đúng", body)
        self.assertIn('id="adminPanel"', body)
        self.assertIn('id="profileButton" onclick="openProfile()">Settings</button>', body)
        self.assertIn('id="googleLoginButton"', body)
        self.assertIn("/api/integrations/google-drive/authorize", body)
        self.assertIn("AUDIT_STEP_ORDER", body)
        self.assertIn("renderAuditLog", body)
        self.assertIn("Authenticate caller", body)
        self.assertIn("Write audit log", body)
        self.assertIn("node.innerHTML = safeHtml", body)
        self.assertIn("node.textContent = text", body)


if __name__ == "__main__":
    unittest.main()
