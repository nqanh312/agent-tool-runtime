"""Connect Claude to registered tools and manage conversation state."""

import json
from registry import ToolRegistry, ToolDefinition
from services.llm import LLMClient, create_llm_client

from tools.google_drive import ALL_DRIVE_TOOLS
from tools.read_file import ALL_READ_FILE_TOOLS
from tools.memory import ALL_MEMORY_TOOLS

ALL_TOOLS: list[ToolDefinition] = ALL_DRIVE_TOOLS + ALL_READ_FILE_TOOLS + ALL_MEMORY_TOOLS

SYSTEM_PROMPT = """\
You are a powerful AI assistant with access to the following capabilities:

1. **Google Drive**: You can list all files and read their contents (supports many formats: PDF, DOCX, XLSX, PPTX, images, etc.).
2. **Long-term Memory**: You can save and search information across conversations using semantic search (RAG).

Guidelines:
- When the user asks to see files from Google Drive, use list_drive_files.
- When the user asks to read a specific Drive file, use list_drive_files first to identify the exact file, then call get_drive_file with its file ID.
- When asked to display a file, reproduce the returned content faithfully. Do not summarize unless the user asks for a summary. If the tool reports truncated=true, clearly tell the user that only part of the file was returned.
- Proactively save important information to memory (user preferences, key facts, task results).
- Before answering questions about past interactions, search_memory first.
- Always respond in the same language as the user's message.
- Be concise and helpful.
"""


class Agent:
    """Run a provider-independent conversation with registered tools."""

    def __init__(
        self,
        service_api_key: str = "sk-admin-001",
        llm_client: LLMClient | None = None,
    ):
        self.llm = llm_client or create_llm_client()
        self.model = self.llm.model
        self.service_api_key = service_api_key
        self.conversation_history: list[dict] = []

        # Register tools once so every model request uses the same catalog.
        self.registry = ToolRegistry()
        for tool in ALL_TOOLS:
            self.registry.register(tool)

    def get_tools_for_claude(self) -> list[dict]:
        """Return tool schemas in the format expected by Claude."""
        return self.registry.list_tools()

    def run(self, user_message: str) -> str:
        """Run the agent loop until Claude returns a final text response."""
        print(f"\n{'#'*60}")
        print(f"  USER: {user_message}")
        print(f"{'#'*60}")

        self.conversation_history.append({
            "role": "user",
            "content": user_message,
        })

        tools = self.get_tools_for_claude()

        while True:
            print(
                f"\n>>> Calling {self.llm.provider} model "
                f"'{self.llm.model}'..."
            )
            response = self.llm.complete(
                system_prompt=SYSTEM_PROMPT,
                tools=tools,
                messages=self.conversation_history,
            )

            print(f">>> Model stop_reason: {response.stop_reason}")

            # End the loop when the model no longer requests a tool.
            if not response.tool_calls:
                if not response.text:
                    raise RuntimeError(
                        f"Model stopped without text or tool calls: "
                        f"{response.stop_reason}"
                    )
                self.conversation_history.append({
                    "role": "assistant",
                    "content": response.text,
                })
                final_response = response.text

                print(f"\n{'#'*60}")
                print(f"  ASSISTANT: {final_response[:500]}")
                if len(final_response) > 500:
                    print(f"  ... [truncated, total {len(final_response)} chars]")
                print(f"{'#'*60}\n")

                return final_response

            self.conversation_history.append({
                "role": "assistant",
                "content": response.text,
                "tool_calls": [
                    {
                        "id": call.id,
                        "name": call.name,
                        "arguments": call.arguments,
                    }
                    for call in response.tool_calls
                ],
            })

            tool_results = []
            for call in response.tool_calls:
                result = self.registry.call(
                    tool_name=call.name,
                    arguments=call.arguments,
                    api_key=self.service_api_key,
                )
                tool_results.append({
                    "tool_call_id": call.id,
                    "name": call.name,
                    "content": json.dumps(result, ensure_ascii=False),
                    "is_error": "error" in result,
                })

            self.conversation_history.append({
                "role": "tool",
                "results": tool_results,
            })

    def clear_history(self):
        """Clear messages while preserving registered tools and audit records."""
        self.conversation_history = []
        print("[Agent] Conversation history cleared.")

    def get_audit_log(self) -> list[dict]:
        """Return tool-call audit entries recorded by the registry."""
        return self.registry.get_audit_log()
