"""Optional PostgreSQL integration test.

Run with RUN_POSTGRES_TESTS=1 after `docker compose up -d postgres` and
`alembic upgrade head`.
"""

import os
import unittest
import uuid

from services.conversations import (
    Conversation,
    ConversationRepository,
    SessionLocal,
)
from services.auth import User


@unittest.skipUnless(
    os.getenv("RUN_POSTGRES_TESTS") == "1",
    "set RUN_POSTGRES_TESTS=1 to run PostgreSQL integration tests",
)
class PostgreSQLConversationTests(unittest.TestCase):
    def setUp(self):
        self.repository = ConversationRepository(SessionLocal)
        self.user_id = f"integration-{uuid.uuid4()}"
        self.conversation_ids: list[str] = []
        with SessionLocal.begin() as session:
            session.add(User(
                id=self.user_id,
                username=f"integration_{uuid.uuid4().hex}",
                display_name="Integration test",
                password_hash=None,
                role_name="guest",
                is_active=False,
                must_change_password=True,
            ))

    def tearDown(self):
        with SessionLocal.begin() as session:
            for conversation_id in self.conversation_ids:
                row = session.get(Conversation, uuid.UUID(conversation_id))
                if row is not None:
                    session.delete(row)
            user = session.get(User, self.user_id)
            if user is not None:
                session.delete(user)

    def test_round_trip_survives_a_new_repository_instance(self):
        conversation = self.repository.create_conversation(
            self.user_id, "Persistent PostgreSQL chat"
        )
        self.conversation_ids.append(conversation["id"])
        user_message, _ = self.repository.append_user_message(
            conversation["id"], self.user_id, "hello", str(uuid.uuid4())
        )
        self.repository.append_assistant_message(
            conversation["id"],
            self.user_id,
            "world",
            ["search_memory"],
            user_message["id"],
        )
        self.repository.save_last_artifact(
            conversation["id"], self.user_id, {"file_id": "drive-test"}
        )

        after_restart = ConversationRepository(SessionLocal)
        messages, _ = after_restart.list_messages(
            conversation["id"], self.user_id
        )
        self.assertEqual([item["content"] for item in messages], ["hello", "world"])
        self.assertEqual(
            after_restart.get_last_artifact(conversation["id"], self.user_id),
            {"file_id": "drive-test"},
        )


if __name__ == "__main__":
    unittest.main()
