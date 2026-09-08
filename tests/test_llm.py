"""Tests for provider-independent LLM adapters."""

from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from services.llm import (
    AnthropicLLMClient,
    ModelToolCall,
    OpenAICompatibleLLMClient,
    _anthropic_messages,
    _openai_messages,
    create_llm_client,
)


TOOLS = [
    {
        "name": "list_drive_files",
        "description": "List Drive files.",
        "input_schema": {
            "type": "object",
            "properties": {},
            "required": [],
        },
    }
]


HISTORY = [
    {"role": "user", "content": "List Drive files"},
    {
        "role": "assistant",
        "content": "",
        "tool_calls": [
            {
                "id": "call-1",
                "name": "list_drive_files",
                "arguments": {},
            }
        ],
    },
    {
        "role": "tool",
        "results": [
            {
                "tool_call_id": "call-1",
                "name": "list_drive_files",
                "content": '{"result":{"total_files":1}}',
                "is_error": False,
            }
        ],
    },
]


class MessageConversionTests(unittest.TestCase):
    def test_converts_history_for_anthropic(self):
        messages = _anthropic_messages(HISTORY)

        self.assertEqual(messages[1]["content"][0]["type"], "tool_use")
        self.assertEqual(messages[2]["role"], "user")
        self.assertEqual(messages[2]["content"][0]["type"], "tool_result")

    def test_converts_history_for_openai_compatible_apis(self):
        messages = _openai_messages("System prompt", HISTORY)

        self.assertEqual(messages[0], {"role": "system", "content": "System prompt"})
        self.assertEqual(messages[2]["tool_calls"][0]["type"], "function")
        self.assertEqual(messages[3]["role"], "tool")
        self.assertEqual(messages[3]["tool_call_id"], "call-1")


class ProviderAdapterTests(unittest.TestCase):
    def test_anthropic_adapter_normalizes_tool_calls(self):
        response = SimpleNamespace(
            stop_reason="tool_use",
            content=[
                SimpleNamespace(type="text", text="Checking Drive"),
                SimpleNamespace(
                    type="tool_use",
                    id="call-1",
                    name="list_drive_files",
                    input={},
                ),
            ],
        )
        messages_api = Mock()
        messages_api.create.return_value = response
        sdk_client = SimpleNamespace(messages=messages_api)
        adapter = AnthropicLLMClient(
            api_key="test-key",
            model="test-model",
            client=sdk_client,
        )

        result = adapter.complete(
            system_prompt="System",
            tools=TOOLS,
            messages=[HISTORY[0]],
        )

        self.assertEqual(result.text, "Checking Drive")
        self.assertEqual(
            result.tool_calls,
            [ModelToolCall("call-1", "list_drive_files", {})],
        )

    def test_openai_compatible_adapter_normalizes_tool_calls(self):
        message = SimpleNamespace(
            content=None,
            tool_calls=[
                SimpleNamespace(
                    id="call-1",
                    function=SimpleNamespace(
                        name="list_drive_files",
                        arguments="{}",
                    ),
                )
            ],
        )
        completion = SimpleNamespace(
            choices=[SimpleNamespace(message=message, finish_reason="tool_calls")]
        )
        completions_api = Mock()
        completions_api.create.return_value = completion
        sdk_client = SimpleNamespace(
            chat=SimpleNamespace(completions=completions_api)
        )
        adapter = OpenAICompatibleLLMClient(
            provider="fci",
            api_key="test-key",
            model="Llama-test",
            base_url="https://example.test/v1",
            client=sdk_client,
        )

        result = adapter.complete(
            system_prompt="System",
            tools=TOOLS,
            messages=[HISTORY[0]],
        )

        self.assertEqual(
            result.tool_calls,
            [ModelToolCall("call-1", "list_drive_files", {})],
        )
        request = completions_api.create.call_args.kwargs
        self.assertEqual(request["tools"][0]["type"], "function")
        self.assertEqual(
            request["tools"][0]["function"]["name"],
            "list_drive_files",
        )

    def test_rejects_missing_provider_key(self):
        with self.assertRaisesRegex(ValueError, "FCI_API_KEY"):
            OpenAICompatibleLLMClient(
                provider="fci",
                api_key="",
                model="test-model",
            )

    @patch("services.llm.OpenAICompatibleLLMClient")
    def test_factory_selects_fci_configuration(self, adapter):
        with (
            patch("services.llm.FCI_API_KEY", "fci-key"),
            patch("services.llm.FCI_MODEL", "fci-model"),
            patch("services.llm.FCI_BASE_URL", "https://fci.example/v1"),
            patch("services.llm.LLM_MODEL", ""),
        ):
            create_llm_client("fci")

        adapter.assert_called_once_with(
            provider="fci",
            api_key="fci-key",
            model="fci-model",
            base_url="https://fci.example/v1",
        )

    def test_factory_rejects_unknown_provider(self):
        with self.assertRaisesRegex(ValueError, "Choose anthropic, openai, or fci"):
            create_llm_client("unknown")


if __name__ == "__main__":
    unittest.main()
