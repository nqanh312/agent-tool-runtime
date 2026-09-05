"""Tests for durable conversation storage and chat-history APIs."""

import unittest
import uuid
from unittest.mock import patch

from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

import server
from services.conversations import (
    Base,
    ConversationNotFoundError,
    ConversationRepository,
    make_title,
)

TEST_USER = {
    "user_id": "user-1", "username": "test", "display_name": "Test",
    "role": "user", "is_active": True, "must_change_password": False,
    "permissions": ["chat:use", "conversation:read", "conversation:write"],
}


class ConversationRepositoryTests(unittest.TestCase):
    def setUp(self):
        self.engine = create_engine(
            "sqlite+pysqlite:///:memory:",
            connect_args={"check_same_thread": False},
            poolclass=StaticPool,
        )
        Base.metadata.create_all(self.engine)
        self.repository = ConversationRepository(
            sessionmaker(bind=self.engine, expire_on_commit=False)
        )

    def tearDown(self):
        self.engine.dispose()

    def test_title_is_normalized_and_truncated_without_an_llm(self):
        title = make_title("  Đây   là " + "một tin nhắn rất dài " * 6)
        self.assertLessEqual(len(title), 60)
        self.assertTrue(title.endswith("…"))
        self.assertNotIn("  ", title)

    def test_conversation_messages_are_durable_and_idempotent(self):
        conversation = self.repository.create_conversation(
            "user-1", "Tôi thích Python"
        )
        user_message, created = self.repository.append_user_message(
            conversation["id"], "user-1", "Tôi thích Python", "client-1"
        )
        duplicate, duplicate_created = self.repository.append_user_message(
            conversation["id"], "user-1", "Tôi thích Python", "client-1"
        )
        assistant = self.repository.append_assistant_message(
            conversation["id"],
            "user-1",
            "Tôi đã ghi nhớ.",
            ["upsert_user_memory"],
            user_message["id"],
        )

        self.assertTrue(created)
        self.assertFalse(duplicate_created)
        self.assertEqual(duplicate["id"], user_message["id"])
        self.assertEqual(
            self.repository.get_reply(user_message["id"])["id"], assistant["id"]
        )
        messages, before = self.repository.list_messages(
            conversation["id"], "user-1"
        )
        self.assertIsNone(before)
        self.assertEqual([message["role"] for message in messages], ["user", "assistant"])
        self.assertEqual(messages[1]["tools_used"], ["upsert_user_memory"])

    def test_first_message_is_created_atomically_with_conversation(self):
        conversation, message = self.repository.create_conversation_with_message(
            "user-1", "First durable message", "atomic-client-1"
        )
        messages, _ = self.repository.list_messages(
            conversation["id"], "user-1"
        )
        self.assertEqual(message["ordinal"], 1)
        self.assertEqual([item["content"] for item in messages], ["First durable message"])

    def test_conversations_are_isolated_by_authenticated_user(self):
        conversation = self.repository.create_conversation("user-1", "Private")
        with self.assertRaises(ConversationNotFoundError):
            self.repository.get_conversation(conversation["id"], "user-2")
        items, _ = self.repository.list_conversations("user-2")
        self.assertEqual(items, [])

    def test_message_pagination_and_artifact_state(self):
        conversation = self.repository.create_conversation("user-1", "Files")
        for index in range(3):
            user_message, _ = self.repository.append_user_message(
                conversation["id"], "user-1", f"Message {index}", f"client-{index}"
            )
            self.repository.append_assistant_message(
                conversation["id"], "user-1", f"Reply {index}", [], user_message["id"]
            )
        latest, before = self.repository.list_messages(
            conversation["id"], "user-1", limit=2
        )
        older, _ = self.repository.list_messages(
            conversation["id"], "user-1", limit=10, before=before
        )
        self.assertEqual([item["ordinal"] for item in latest], [5, 6])
        self.assertEqual([item["ordinal"] for item in older], [1, 2, 3, 4])

        artifact = {"file_id": "drive-1", "content": "Document"}
        self.repository.save_last_artifact(
            conversation["id"], "user-1", artifact
        )
        self.assertEqual(
            self.repository.get_last_artifact(conversation["id"], "user-1"),
            artifact,
        )


class _PersistentFakeAgent:
    last_artifact = None

    def __init__(self):
        self.audit = []

    def get_audit_log(self):
        return self.audit

    def run(self, message):
        self.audit.append({"tool": "search_memory"})
        return f"Reply to {message}"


class PersistentChatApiTests(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(server.app)
        server.sessions.clear()
        server.app.dependency_overrides[server.current_principal] = lambda: TEST_USER

    def tearDown(self):
        server.sessions.clear()
        server.app.dependency_overrides.clear()

    def test_agent_is_rehydrated_from_messages_and_artifact_after_restart(self):
        repository = unittest.mock.MagicMock()
        repository.list_context_messages.return_value = [
            {"role": "user", "content": "old question"},
            {"role": "assistant", "content": "old answer"},
        ]
        repository.get_last_artifact.return_value = {"file_id": "drive-1"}
        with (
            patch.object(server, "conversation_repository", repository),
            patch.object(server, "Agent") as agent_class,
        ):
            server._get_conversation_agent("conversation-1", TEST_USER)

        agent_class.assert_called_once_with(
            principal=TEST_USER,
            conversation_history=[
                {"role": "user", "content": "old question"},
                {"role": "assistant", "content": "old answer"},
            ],
            last_artifact={"file_id": "drive-1"},
            audit_sink=unittest.mock.ANY,
        )

    def test_first_message_returns_persistent_conversation_metadata(self):
        conversation_id = str(uuid.uuid4())
        user_message_id = str(uuid.uuid4())
        assistant_message_id = str(uuid.uuid4())
        repository = unittest.mock.MagicMock()
        repository.find_user_message.return_value = None
        repository.create_conversation_with_message.return_value = (
            {"id": conversation_id, "title": "Hello history"},
            {"id": user_message_id, "ordinal": 1},
        )
        repository.append_assistant_message.return_value = {
            "id": assistant_message_id,
            "created_at": "2026-09-05T12:00:00+00:00",
        }

        with (
            patch.object(server, "conversation_repository", repository),
            patch.object(
                server,
                "_get_conversation_agent",
                return_value=_PersistentFakeAgent(),
            ),
        ):
            response = self.client.post(
                "/api/chat",
                json={
                    "message": "Hello history",
                    "client_message_id": "client-api-1",
                },
            )

        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body["conversation_id"], conversation_id)
        self.assertEqual(body["title"], "Hello history")
        self.assertEqual(body["tools_used"], ["search_memory"])
        repository.create_conversation_with_message.assert_called_once()
        repository.append_user_message.assert_not_called()
        repository.append_assistant_message.assert_called_once()

    def test_history_endpoint_sanitizes_stored_assistant_markdown(self):
        repository = unittest.mock.MagicMock()
        repository.list_messages.return_value = (
            [
                {
                    "id": str(uuid.uuid4()),
                    "conversation_id": str(uuid.uuid4()),
                    "ordinal": 1,
                    "role": "assistant",
                    "content": "<script>alert(1)</script> **safe**",
                    "tools_used": [],
                    "created_at": "2026-09-05T12:00:00+00:00",
                }
            ],
            None,
        )
        with patch.object(server, "conversation_repository", repository):
            response = self.client.get(
                f"/api/conversations/{uuid.uuid4()}/messages"
            )
        self.assertEqual(response.status_code, 200)
        html = response.json()["messages"][0]["response_html"]
        self.assertNotIn("<script", html)
        self.assertIn("<strong>safe</strong>", html)

    def test_ui_contains_left_history_sidebar_and_lazy_new_chat(self):
        body = self.client.get("/").text
        self.assertIn('id="sidebar"', body)
        self.assertIn("/api/conversations", body)
        self.assertIn("function newChat()", body)
        self.assertIn("localStorage.removeItem(ACTIVE_KEY)", body)
        self.assertNotIn("apiFetch('/api/clear'", body)


if __name__ == "__main__":
    unittest.main()
