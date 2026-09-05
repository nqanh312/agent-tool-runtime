"""Tests for durable audit-log persistence."""

import unittest

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from services.audit_logs import AuditBase, AuditLogRepository


class AuditLogRepositoryTests(unittest.TestCase):
    def setUp(self):
        self.engine = create_engine("sqlite+pysqlite:///:memory:")
        AuditBase.metadata.create_all(self.engine)
        self.repository = AuditLogRepository(
            sessionmaker(bind=self.engine, expire_on_commit=False)
        )

    def tearDown(self):
        self.engine.dispose()

    def test_entry_survives_a_new_repository_instance(self):
        entry = {
            "timestamp": "2026-09-05T12:00:00+00:00",
            "user_id": "user-1",
            "role": "admin",
            "tool": "echo",
            "arguments": {"message": "hello"},
            "result": {"message": "hello"},
            "error": None,
            "status": "success",
            "steps": [
                {"step": number, "status": "success"}
                for number in range(1, 7)
            ],
        }
        self.repository.append("conversation:test", entry)

        after_restart = AuditLogRepository(
            sessionmaker(bind=self.engine, expire_on_commit=False)
        )
        rows = after_restart.list_entries("conversation:test", "user-1")

        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["tool"], "echo")
        self.assertEqual(len(rows[0]["steps"]), 6)

    def test_entries_are_isolated_by_user_and_context(self):
        base_entry = {
            "timestamp": "2026-09-05T12:00:00+00:00",
            "role": "user",
            "tool": "echo",
            "arguments": {},
            "result": None,
            "error": None,
            "status": "success",
            "steps": [],
        }
        self.repository.append("conversation:one", {**base_entry, "user_id": "a"})
        self.repository.append("conversation:two", {**base_entry, "user_id": "a"})
        self.repository.append("conversation:one", {**base_entry, "user_id": "b"})

        rows = self.repository.list_entries("conversation:one", "a")

        self.assertEqual(len(rows), 1)


if __name__ == "__main__":
    unittest.main()
