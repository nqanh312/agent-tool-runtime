"""Provider adapters for Anthropic and OpenAI-compatible chat models."""

from dataclasses import dataclass, field
import json
from typing import Protocol

import anthropic
from openai import OpenAI

from config import (
    ANTHROPIC_API_KEY,
    ANTHROPIC_MODEL,
    FCI_API_KEY,
    FCI_BASE_URL,
    FCI_MODEL,
    LLM_MODEL,
    LLM_PROVIDER,
    OPENAI_API_KEY,
    OPENAI_BASE_URL,
    OPENAI_LLM_MODEL,
)


@dataclass(frozen=True)
class ModelToolCall:
    """A provider-independent request to execute one registered tool."""

    id: str
    name: str
    arguments: dict


@dataclass(frozen=True)
class ModelResponse:
    """A provider-independent model response."""

    text: str = ""
    tool_calls: list[ModelToolCall] = field(default_factory=list)
    stop_reason: str = "stop"


class LLMClient(Protocol):
    """Interface consumed by the Agent conversation loop."""

    provider: str
    model: str

    def complete(
        self,
        *,
        system_prompt: str,
        tools: list[dict],
        messages: list[dict],
    ) -> ModelResponse:
        """Generate the next assistant response or set of tool calls."""


def _anthropic_messages(messages: list[dict]) -> list[dict]:
    """Convert provider-independent history to Anthropic messages."""
    converted = []

    for message in messages:
        role = message["role"]
        if role == "user":
            converted.append({"role": "user", "content": message["content"]})
            continue

        if role == "assistant":
            tool_calls = message.get("tool_calls", [])
            if not tool_calls:
                converted.append(
                    {"role": "assistant", "content": message.get("content", "")}
                )
                continue

            content = []
            if message.get("content"):
                content.append({"type": "text", "text": message["content"]})
            content.extend(
                {
                    "type": "tool_use",
                    "id": call["id"],
                    "name": call["name"],
                    "input": call["arguments"],
                }
                for call in tool_calls
            )
            converted.append({"role": "assistant", "content": content})
            continue

        if role == "tool":
            converted.append(
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": result["tool_call_id"],
                            "content": result["content"],
                            "is_error": result.get("is_error", False),
                        }
                        for result in message["results"]
                    ],
                }
            )
            continue

        raise ValueError(f"Unsupported conversation role: {role}")

    return converted


def _openai_messages(system_prompt: str, messages: list[dict]) -> list[dict]:
    """Convert provider-independent history to Chat Completions messages."""
    converted = [{"role": "system", "content": system_prompt}]

    for message in messages:
        role = message["role"]
        if role == "user":
            converted.append({"role": "user", "content": message["content"]})
            continue

        if role == "assistant":
            tool_calls = message.get("tool_calls", [])
            assistant_message = {
                "role": "assistant",
                "content": message.get("content") or None,
            }
            if tool_calls:
                assistant_message["tool_calls"] = [
                    {
                        "id": call["id"],
                        "type": "function",
                        "function": {
                            "name": call["name"],
                            "arguments": json.dumps(
                                call["arguments"],
                                ensure_ascii=False,
                            ),
                        },
                    }
                    for call in tool_calls
                ]
            converted.append(assistant_message)
            continue

        if role == "tool":
            converted.extend(
                {
                    "role": "tool",
                    "tool_call_id": result["tool_call_id"],
                    "content": result["content"],
                }
                for result in message["results"]
            )
            continue

        raise ValueError(f"Unsupported conversation role: {role}")

    return converted


def _openai_tools(tools: list[dict]) -> list[dict]:
    """Wrap internal tool definitions as OpenAI function tools."""
    return [
        {
            "type": "function",
            "function": {
                "name": tool["name"],
                "description": tool["description"],
                "parameters": tool["input_schema"],
            },
        }
        for tool in tools
    ]


class AnthropicLLMClient:
    """Native Anthropic Messages API adapter."""

    provider = "anthropic"

    def __init__(self, api_key: str, model: str, client=None):
        if not api_key:
            raise ValueError("ANTHROPIC_API_KEY is required for LLM_PROVIDER=anthropic")
        self.model = model
        self.client = client or anthropic.Anthropic(api_key=api_key)

    def complete(self, *, system_prompt, tools, messages) -> ModelResponse:
        response = self.client.messages.create(
            model=self.model,
            max_tokens=4096,
            system=system_prompt,
            tools=tools,
            messages=_anthropic_messages(messages),
        )

        text = "\n".join(
            block.text
            for block in response.content
            if getattr(block, "type", None) == "text"
        )
        tool_calls = [
            ModelToolCall(
                id=block.id,
                name=block.name,
                arguments=dict(block.input),
            )
            for block in response.content
            if getattr(block, "type", None) == "tool_use"
        ]
        return ModelResponse(
            text=text,
            tool_calls=tool_calls,
            stop_reason=response.stop_reason or "stop",
        )


class OpenAICompatibleLLMClient:
    """Chat Completions adapter used by OpenAI and FCI."""

    def __init__(
        self,
        *,
        provider: str,
        api_key: str,
        model: str,
        base_url: str | None = None,
        client=None,
    ):
        if not api_key:
            variable = "FCI_API_KEY" if provider == "fci" else "OPENAI_API_KEY"
            raise ValueError(f"{variable} is required for LLM_PROVIDER={provider}")
        self.provider = provider
        self.model = model
        self.base_url = base_url
        self.client = client or OpenAI(api_key=api_key, base_url=base_url)

    def complete(self, *, system_prompt, tools, messages) -> ModelResponse:
        request = {
            "model": self.model,
            "messages": _openai_messages(system_prompt, messages),
            "max_tokens": 4096,
        }
        if tools:
            request["tools"] = _openai_tools(tools)

        response = self.client.chat.completions.create(**request)
        choice = response.choices[0]
        message = choice.message
        tool_calls = []

        for call in message.tool_calls or []:
            try:
                arguments = json.loads(call.function.arguments or "{}")
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Model returned invalid JSON arguments for tool "
                    f"'{call.function.name}'"
                ) from exc
            if not isinstance(arguments, dict):
                raise ValueError(
                    f"Model arguments for tool '{call.function.name}' must be an object"
                )
            tool_calls.append(
                ModelToolCall(
                    id=call.id,
                    name=call.function.name,
                    arguments=arguments,
                )
            )

        return ModelResponse(
            text=message.content or "",
            tool_calls=tool_calls,
            stop_reason=choice.finish_reason or "stop",
        )


def create_llm_client(provider: str | None = None) -> LLMClient:
    """Build the configured provider adapter."""
    selected = (provider or LLM_PROVIDER).strip().lower()

    if selected == "anthropic":
        return AnthropicLLMClient(
            api_key=ANTHROPIC_API_KEY,
            model=LLM_MODEL or ANTHROPIC_MODEL,
        )

    if selected == "openai":
        return OpenAICompatibleLLMClient(
            provider="openai",
            api_key=OPENAI_API_KEY,
            model=LLM_MODEL or OPENAI_LLM_MODEL,
            base_url=OPENAI_BASE_URL or None,
        )

    if selected in {"fci", "fpt"}:
        return OpenAICompatibleLLMClient(
            provider="fci",
            api_key=FCI_API_KEY,
            model=LLM_MODEL or FCI_MODEL,
            base_url=FCI_BASE_URL,
        )

    raise ValueError(
        f"Unsupported LLM_PROVIDER '{selected}'. "
        "Choose anthropic, openai, or fci."
    )
