"""Tests for chat admission limits and bounded in-process state."""

from datetime import datetime, timezone
import unittest
from unittest.mock import patch

import server
from services.request_limits import ChatLimitExceeded, ChatUsageLimiter


class ChatUsageLimiterTests(unittest.TestCase):
    def test_enforces_sliding_request_limit_and_retry_time(self):
        monotonic_now = [100.0]
        wall_now = [datetime(2026, 9, 8, tzinfo=timezone.utc).timestamp()]
        limiter = ChatUsageLimiter(
            max_requests=2,
            window_seconds=60,
            daily_token_quota=100_000,
            base_token_charge=10,
            monotonic=lambda: monotonic_now[0],
            wall_time=lambda: wall_now[0],
        )

        limiter.check("user-1", "hello")
        limiter.check("user-1", "hello again")
        with self.assertRaises(ChatLimitExceeded) as raised:
            limiter.check("user-1", "third request")
        self.assertEqual(raised.exception.retry_after, 60)

        monotonic_now[0] += 61
        limiter.check("user-1", "allowed after the window")

    def test_enforces_daily_estimated_token_quota(self):
        wall_now = [datetime(2026, 9, 8, 12, tzinfo=timezone.utc).timestamp()]
        limiter = ChatUsageLimiter(
            max_requests=100,
            window_seconds=60,
            daily_token_quota=23,
            base_token_charge=10,
            monotonic=lambda: 100.0,
            wall_time=lambda: wall_now[0],
        )

        limiter.check("user-1", "a")
        limiter.check("user-1", "b")
        with self.assertRaises(ChatLimitExceeded) as raised:
            limiter.check("user-1", "c")
        self.assertIn("quota", str(raised.exception).lower())


class BoundedStateTests(unittest.TestCase):
    def tearDown(self):
        server.sessions.clear()
        server.session_access.clear()
        server.conversation_locks.clear()
        server.conversation_lock_states.clear()

    def test_session_cache_evicts_least_recently_used_agent(self):
        server.sessions.update({"old": object(), "new": object()})
        server.session_access.update({"old": 1.0, "new": 2.0})
        with (
            patch.object(server, "AGENT_SESSION_CACHE_MAX", 1),
            patch.object(server, "AGENT_SESSION_TTL_SECONDS", 1_000),
        ):
            server._prune_sessions(10.0, protected_key="new")
        self.assertNotIn("old", server.sessions)
        self.assertIn("new", server.sessions)

    def test_idle_conversation_lock_is_removed(self):
        key = "user-1:conversation-1"
        lock = server._try_acquire_conversation_lock(key)
        self.assertIsNotNone(lock)
        with patch.object(server, "CONVERSATION_LOCK_TTL_SECONDS", 0):
            server._release_conversation_lock(key, lock)
        self.assertNotIn(key, server.conversation_locks)


if __name__ == "__main__":
    unittest.main()
