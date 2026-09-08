"""Provide an interactive command-line interface for the agent."""

import json
from agent import Agent
from config import CLI_USER_ID
from services.auth import AuthenticationError, auth_repository
from services.audit_logs import audit_log_repository


def print_help():
    """Print the commands supported by the interactive shell."""
    print("""
=== Agent CLI Commands ===
  /quit     - Exit the application
  /clear    - Clear conversation history
  /audit    - Show audit log (all tool calls)
  /memory   - Show all stored memories
  /help     - Show this help message
==============================
""")


def main():
    """Run the interactive chat loop."""
    print("\n" + "="*60)
    print("  ASSIGNMENT 1 - AI Agent with Tool Registry")
    print("  Capabilities: Google Drive | Read File | RAG Memory")
    print("="*60)

    context_id = "cli:default"
    try:
        principal = auth_repository.principal(CLI_USER_ID)
    except AuthenticationError as exc:
        raise SystemExit(str(exc)) from exc
    user_id = principal["user_id"]
    agent = Agent(
        principal=principal,
        audit_sink=lambda entry: audit_log_repository.append(context_id, entry),
    )

    print_help()

    while True:
        try:
            user_input = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not user_input:
            continue

        # Handle local commands before sending input to the model.
        if user_input.lower() == "/quit":
            print("Goodbye!")
            break

        if user_input.lower() == "/help":
            print_help()
            continue

        if user_input.lower() == "/clear":
            agent.clear_history()
            continue

        if user_input.lower() == "/audit":
            logs = audit_log_repository.list_entries(context_id, user_id)
            if not logs:
                print("\n[No audit logs yet]")
            else:
                print(f"\n--- AUDIT LOG ({len(logs)} entries) ---")
                for entry in logs:
                    print(json.dumps(entry, indent=2, ensure_ascii=False))
                print("--- END AUDIT LOG ---")
            continue

        if user_input.lower() == "/memory":
            try:
                from services.vectorstore import list_all_memories
                memories = list_all_memories(user_id=user_id)
                if not memories:
                    print("\n[No memories stored yet]")
                else:
                    print(f"\n--- MEMORIES ({len(memories)} entries) ---")
                    for m in memories:
                        print(f"  [{m.get('metadata', {}).get('category', 'general')}] {m['text'][:100]}")
                    print("--- END MEMORIES ---")
            except Exception as e:
                print(f"\n[Error accessing memory: {e}]")
            continue

        try:
            response = agent.run(user_input)
            print(f"\nAssistant: {response}")
        except Exception as e:
            print(f"\n[Error: {e}]")


if __name__ == "__main__":
    main()
