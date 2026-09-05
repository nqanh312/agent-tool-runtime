"""Tests for registry policy and the list_drive_files execution path."""

import unittest
from unittest.mock import patch

from registry.models import ToolDefinition
from registry.registry import (
    AUDIT_LOG,
    RateLimiter,
    ToolRegistry,
    rate_limiter,
    validate_schema,
)
from tools import google_drive


def _echo_tool(required_scopes=None):
    return ToolDefinition(
        name="echo",
        description="Echo a message.",
        input_schema={
            "type": "object",
            "properties": {
                "message": {"type": "string"},
                "count": {"type": "integer", "enum": [1, 2]},
            },
            "required": ["message"],
        },
        required_scopes=required_scopes or [],
        handler=lambda message, count=1: message * count,
    )


class SchemaValidationTests(unittest.TestCase):
    def test_validates_required_type_enum_and_unknown_fields(self):
        tool = _echo_tool()

        self.assertEqual(
            validate_schema(tool, {"message": "hi", "count": 2}),
            {"message": "hi", "count": 2},
        )

        invalid_arguments = (
            ({}, "Missing required fields"),
            ({"message": 1}, "must be of type string"),
            ({"message": "hi", "count": 3}, "must be one of"),
            ({"message": "hi", "extra": True}, "Unexpected fields"),
        )
        for arguments, message in invalid_arguments:
            with self.subTest(arguments=arguments):
                with self.assertRaisesRegex(ValueError, message):
                    validate_schema(tool, arguments)


class RegistryTests(unittest.TestCase):
    def setUp(self):
        AUDIT_LOG.clear()
        rate_limiter._buckets.clear()

    def test_executes_list_drive_files_through_registry(self):
        registry = ToolRegistry()
        registry.register(google_drive.list_files_tool)
        expected_files = [{"id": "file-1", "name": "Document"}]

        with patch.object(
            google_drive.drive_service,
            "list_files",
            return_value=expected_files,
        ):
            response = registry.call(
                tool_name="list_drive_files",
                arguments={},
                api_key="sk-admin-001",
            )

        self.assertEqual(
            response,
            {"result": {"total_files": 1, "files": expected_files}},
        )
        self.assertEqual(AUDIT_LOG[-1]["status"], "success")
        self.assertEqual(AUDIT_LOG[-1]["tool"], "list_drive_files")

    def test_rejects_invalid_authentication(self):
        registry = ToolRegistry()
        registry.register(_echo_tool())

        response = registry.call("echo", {"message": "hi"}, "wrong-key")

        self.assertEqual(response["error_type"], "PermissionError")
        self.assertEqual(AUDIT_LOG[-1]["user_id"], "anonymous")
        self.assertNotIn("wrong-key", str(AUDIT_LOG[-1]))

    def test_rejects_missing_scope(self):
        registry = ToolRegistry()
        registry.register(_echo_tool(required_scopes=["memory:write"]))

        response = registry.call("echo", {"message": "hi"}, "sk-guest-003")

        self.assertEqual(response["error_type"], "PermissionError")
        self.assertIn("memory:write", response["error"])

    def test_rate_limiter_uses_a_sliding_window(self):
        limiter = RateLimiter(max_calls=2, window_seconds=60)

        with patch("registry.registry.time.monotonic", side_effect=[0, 1, 30]):
            self.assertTrue(limiter.check("user-1"))
            self.assertTrue(limiter.check("user-1"))
            with self.assertRaisesRegex(RuntimeError, "Rate limit exceeded"):
                limiter.check("user-1")

        with patch("registry.registry.time.monotonic", return_value=61):
            self.assertTrue(limiter.check("user-1"))


if __name__ == "__main__":
    unittest.main()
