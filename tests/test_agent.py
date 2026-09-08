"""Test the model -> registry -> Drive tool -> model loop."""

import unittest
from unittest.mock import patch

import agent as agent_module
from agent import Agent
from registry.registry import AUDIT_LOG, rate_limiter
from services.llm import ModelResponse
from services.memory_extractor import TurnPlan
from tools import google_drive

ADMIN = {
    "user_id": "user_admin", "role": "admin", "is_active": True,
    "permissions": ["drive:read", "memory:read", "memory:write"],
}


class _FakeLLMClient:
    provider = "test"
    model = "test-model"

    def __init__(self, responses):
        self.responses = iter(responses)
        self.requests = []

    def complete(self, **request):
        self.requests.append(request)
        return next(self.responses)


class AgentToolLoopTests(unittest.TestCase):
    def setUp(self):
        AUDIT_LOG.clear()
        rate_limiter._buckets.clear()

    def test_list_drive_files_tool_result_is_returned_to_model(self):
        llm = _FakeLLMClient(
            [
                ModelResponse(text="Drive has 1 file.", stop_reason="stop"),
            ]
        )

        with (
            patch.object(
                agent_module,
                "plan_user_turn",
                return_value=TurnPlan(intent="browse_drive"),
            ),
            patch.object(
                google_drive.drive_service,
                "list_files",
                return_value=[{"id": "file-1", "name": "Document"}],
            ),
        ):
            tested_agent = Agent(principal=ADMIN, llm_client=llm)
            response = tested_agent.run("List files in Drive")

        self.assertEqual(response, "Drive has 1 file.")
        self.assertEqual(len(llm.requests), 1)
        self.assertIn("Document", llm.requests[0]["system_prompt"])
        self.assertEqual(llm.requests[0]["tools"], [])
        self.assertEqual(AUDIT_LOG[-1]["tool"], "list_drive_files")

    def test_agent_lists_then_reads_the_selected_drive_file(self):
        llm = _FakeLLMClient(
            [
                ModelResponse(
                    text="File content",
                    stop_reason="stop",
                ),
            ]
        )
        download = {
            "file_id": "file-1",
            "file_name": "Document.txt",
            "mime_type": "text/plain",
            "temp_path": "temporary-file.txt",
        }

        with (
            patch.object(
                agent_module,
                "plan_user_turn",
                return_value=TurnPlan(
                    intent="read_drive_file",
                    file_query="Document.txt",
                ),
            ),
            patch.object(
                google_drive.drive_service,
                "search_files",
                return_value=[{"id": "file-1", "name": "Document.txt"}],
            ),
            patch.object(
                google_drive.drive_service,
                "download_file",
                return_value=download,
            ),
            patch.object(
                google_drive,
                "read_file",
                return_value={
                    "file_name": "temporary-file.txt",
                    "content": "File content",
                    "truncated": False,
                    "total_characters": 12,
                },
            ),
            patch.object(google_drive.os, "unlink"),
        ):
            tested_agent = Agent(principal=ADMIN, llm_client=llm)
            response = tested_agent.run("Read Document.txt")

        self.assertEqual(response, "File content")
        self.assertEqual(
            [entry["tool"] for entry in AUDIT_LOG],
            ["search_drive_files", "get_drive_file"],
        )
        self.assertEqual(len(llm.requests), 1)
        self.assertEqual(llm.requests[0]["tools"], [])
        self.assertEqual(tested_agent.last_artifact["file_id"], "file-1")


if __name__ == "__main__":
    unittest.main()
