"""Register tools and enforce policy before each tool execution."""

from copy import deepcopy
from datetime import datetime, timezone
import json
import threading
import time
from typing import Any

from .models import ToolDefinition


# Step 1: validate arguments.

def validate_schema(tool: ToolDefinition, arguments: dict) -> dict:
    """Validate arguments against the tool's JSON schema."""
    if not isinstance(arguments, dict):
        raise ValueError("Tool arguments must be an object")

    schema = tool.input_schema
    if schema.get("type", "object") != "object":
        raise ValueError(f"Tool '{tool.name}' must declare an object input schema")

    properties = schema.get("properties", {})
    required = schema.get("required", [])

    missing = [field for field in required if field not in arguments]
    if missing:
        raise ValueError(f"Missing required fields: {', '.join(missing)}")

    unknown = [field for field in arguments if field not in properties]
    if unknown:
        raise ValueError(f"Unexpected fields: {', '.join(unknown)}")

    type_checks = {
        "string": lambda value: isinstance(value, str),
        "integer": lambda value: isinstance(value, int) and not isinstance(value, bool),
        "number": lambda value: isinstance(value, (int, float)) and not isinstance(value, bool),
        "boolean": lambda value: isinstance(value, bool),
        "object": lambda value: isinstance(value, dict),
        "array": lambda value: isinstance(value, list),
        "null": lambda value: value is None,
    }

    for field, value in arguments.items():
        field_schema = properties[field]
        expected_type = field_schema.get("type")
        if expected_type:
            checker = type_checks.get(expected_type)
            if checker is None:
                raise ValueError(
                    f"Unsupported schema type '{expected_type}' for field '{field}'"
                )
            if not checker(value):
                raise ValueError(
                    f"Field '{field}' must be of type {expected_type}"
                )

        allowed_values = field_schema.get("enum")
        if allowed_values is not None and value not in allowed_values:
            raise ValueError(
                f"Field '{field}' must be one of: {allowed_values}"
            )

    return dict(arguments)


# Step 2: authenticate the caller.

USER_DB: dict[str, dict] = {
    "sk-admin-001": {
        "user_id": "user_admin",
        "role": "admin",
        "scopes": [
            "drive:read",
            "memory:read", "memory:write",
        ],
    },
    "sk-user-002": {
        "user_id": "user_standard",
        "role": "user",
        "scopes": [
            "drive:read",
            "memory:read", "memory:write",
        ],
    },
    "sk-guest-003": {
        "user_id": "user_guest",
        "role": "guest",
        "scopes": ["drive:read", "memory:read"],
    },
}


def check_authentication(api_key: str) -> dict:
    """Return the user associated with an API key."""
    if not api_key:
        raise PermissionError("Authentication required")

    user = USER_DB.get(api_key)
    if user is None:
        raise PermissionError("Invalid API key")

    return deepcopy(user)


# Step 3: authorize access to the tool.

def check_scopes(user: dict, tool: ToolDefinition) -> bool:
    """Verify that the user has every scope required by the tool."""
    user_scopes = set(user.get("scopes", []))
    missing_scopes = [
        scope for scope in tool.required_scopes
        if scope not in user_scopes
    ]
    if missing_scopes:
        raise PermissionError(
            f"Missing required scopes: {', '.join(missing_scopes)}"
        )
    return True


# Step 4: enforce a per-user rate limit.

class RateLimiter:
    """Track tool calls within a sliding time window."""

    def __init__(self, max_calls: int = 10, window_seconds: int = 60):
        if max_calls <= 0 or window_seconds <= 0:
            raise ValueError("Rate-limit values must be positive")
        self.max_calls = max_calls
        self.window_seconds = window_seconds
        self._buckets: dict[str, list[float]] = {}
        self._lock = threading.Lock()

    def check(self, user_id: str) -> bool:
        """Record a call or raise RuntimeError when the limit is exceeded."""
        if not user_id:
            raise ValueError("user_id is required for rate limiting")

        now = time.monotonic()
        cutoff = now - self.window_seconds

        with self._lock:
            recent_calls = [
                timestamp
                for timestamp in self._buckets.get(user_id, [])
                if timestamp > cutoff
            ]
            if len(recent_calls) >= self.max_calls:
                self._buckets[user_id] = recent_calls
                raise RuntimeError("Rate limit exceeded; please try again later")

            recent_calls.append(now)
            self._buckets[user_id] = recent_calls

        return True


rate_limiter = RateLimiter(max_calls=20, window_seconds=60)


# Step 5: record the outcome.

AUDIT_LOG: list[dict] = []


def audit_log(user: dict, tool_name: str, arguments: dict, result: Any = None, error: str = None):
    """Append a structured tool-call outcome to the audit log."""
    entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "user_id": user.get("user_id", "anonymous"),
        "role": user.get("role", "unknown"),
        "tool": tool_name,
        "arguments": deepcopy(arguments),
        "result": deepcopy(result),
        "error": error,
        "status": "error" if error else "success",
    }
    AUDIT_LOG.append(entry)
    return entry


# Step 6: invoke the handler.

def execute_tool(tool: ToolDefinition, arguments: dict) -> Any:
    """Invoke a tool handler with validated keyword arguments."""
    return tool.handler(**arguments)


class ToolRegistry:
    """Store tool definitions and mediate access to their handlers."""

    def __init__(self):
        self._tools: dict[str, ToolDefinition] = {}

    def register(self, tool: ToolDefinition):
        """Add or replace a tool definition by name."""
        self._tools[tool.name] = tool
        print(f"[Registry] Registered tool: '{tool.name}'")

    def get_tool(self, name: str) -> ToolDefinition:
        """Return a registered tool or raise KeyError."""
        tool = self._tools.get(name)
        if not tool:
            raise KeyError(f"[Registry] Tool '{name}' not found. Available: {list(self._tools.keys())}")
        return tool

    def list_tools(self) -> list[dict]:
        """Return model-facing schemas for all registered tools."""
        return [
            {
                "name": t.name,
                "description": t.description,
                "input_schema": t.input_schema,
            }
            for t in self._tools.values()
        ]

    def call(self, tool_name: str, arguments: dict, api_key: str) -> dict:
        """Apply registry policy, execute a tool, and audit the outcome."""
        print(f"\n{'='*60}")
        print(f"  TOOL CALL: {tool_name}")
        print(f"  Arguments: {json.dumps(arguments, ensure_ascii=False)}")
        print(f"{'='*60}")

        user = {"user_id": "anonymous", "role": "unknown", "scopes": []}
        audit_arguments = arguments if isinstance(arguments, dict) else {}

        try:
            tool = self.get_tool(tool_name)
            validated_arguments = validate_schema(tool, arguments)
            audit_arguments = validated_arguments
            user = check_authentication(api_key)
            check_scopes(user, tool)
            rate_limiter.check(user["user_id"])

            result = execute_tool(tool, validated_arguments)
            audit_log(user, tool_name, validated_arguments, result=result)
            return {"result": result}
        except Exception as exc:
            error_message = str(exc) or exc.__class__.__name__
            audit_log(
                user,
                tool_name,
                audit_arguments,
                error=error_message,
            )
            return {
                "error": error_message,
                "error_type": exc.__class__.__name__,
            }

    def get_audit_log(self) -> list[dict]:
        """Return all audit entries recorded by this process."""
        return deepcopy(AUDIT_LOG)
