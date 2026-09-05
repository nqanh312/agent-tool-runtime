"""Test the model -> registry -> Drive tool -> model loop."""

import json
import unittest
from unittest.mock import patch

from agent import Agent
from registry.registry import AUDIT_LOG, rate_limiter
from services.llm import ModelResponse, ModelToolCall
from tools import google_drive


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
                ModelResponse(
                    tool_calls=[
                        ModelToolCall(
                            id="tool-use-1",
                            name="list_drive_files",
                            arguments={},
                        )
                    ],
                    stop_reason="tool_calls",
                ),
                ModelResponse(text="Drive has 1 file.", stop_reason="stop"),
            ]
        )

        with patch.object(
            google_drive.drive_service,
            "list_files",
            return_value=[{"id": "file-1", "name": "Document"}],
        ):
            tested_agent = Agent(llm_client=llm)
            response = tested_agent.run("List files in Drive")

        self.assertEqual(response, "Drive has 1 file.")
        self.assertEqual(len(llm.requests), 2)

        tool_result_message = next(
            message
            for message in tested_agent.conversation_history
            if message["role"] == "tool"
        )
        tool_result = tool_result_message["results"][0]
        payload = json.loads(tool_result["content"])
        self.assertFalse(tool_result["is_error"])
        self.assertEqual(payload["result"]["total_files"], 1)
        self.assertEqual(AUDIT_LOG[-1]["tool"], "list_drive_files")

    def test_agent_lists_then_reads_the_selected_drive_file(self):
        llm = _FakeLLMClient(
            [
                ModelResponse(
                    tool_calls=[
                        ModelToolCall("list-1", "list_drive_files", {})
                    ],
                    stop_reason="tool_calls",
                ),
                ModelResponse(
                    tool_calls=[
                        ModelToolCall(
                            "read-1",
                            "get_drive_file",
                            {"file_id": "file-1"},
                        )
                    ],
                    stop_reason="tool_calls",
                ),
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
                google_drive.drive_service,
                "list_files",
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
            response = Agent(llm_client=llm).run("Read Document.txt")

        self.assertEqual(response, "File content")
        self.assertEqual(
            [entry["tool"] for entry in AUDIT_LOG],
            ["list_drive_files", "get_drive_file"],
        )


if __name__ == "__main__":
    unittest.main()
