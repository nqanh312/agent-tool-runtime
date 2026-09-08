"""Register tools and enforce policy before each tool execution."""

from copy import deepcopy
from contextvars import ContextVar
from datetime import datetime, timezone
import json
import sys
import threading
import time
from typing import Any, Callable

from .models import ToolDefinition


_current_user: ContextVar[dict | None] = ContextVar(
    "current_tool_user",
    default=None,
)


def get_current_user() -> dict:
    """Return the authenticated user for the tool currently being executed."""
    return deepcopy(_current_user.get() or {
        "user_id": "anonymous",
        "role": "unknown",
        "permissions": [],
    })


def _console_safe(value: str) -> str:
    """Make diagnostic output safe for Windows consoles with legacy encodings."""
    encoding = getattr(sys.stdout, "encoding", None) or "utf-8"
    return value.encode(encoding, errors="backslashreplace").decode(encoding)


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

def check_authentication(principal: dict | None) -> dict:
    """Validate an identity produced by the HTTP or CLI authenticator."""
    if not isinstance(principal, dict):
        raise PermissionError("Authentication required")
    if not principal.get("user_id") or not principal.get("role"):
        raise PermissionError("Invalid authenticated principal")
    if principal.get("is_active") is False:
        raise PermissionError("Account is inactive")
    permissions = principal.get("permissions")
    if not isinstance(permissions, (list, tuple, set)):
        raise PermissionError("Invalid authenticated principal")
    authenticated = deepcopy(principal)
    authenticated["permissions"] = list(permissions)
    return authenticated


# Step 3: authorize access to the tool.

def check_permissions(user: dict, tool: ToolDefinition) -> bool:
    """Verify that the principal has every permission required by the tool."""
    user_permissions = set(user.get("permissions", []))
    missing_permissions = [
        permission for permission in tool.required_permissions
        if permission not in user_permissions
    ]
    if missing_permissions:
        raise PermissionError(
            f"Missing required permissions: {', '.join(missing_permissions)}"
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


class AuditPersistenceError(RuntimeError):
    """Raised when a configured durable audit sink cannot store an entry."""


AUDIT_STEP_NAMES = (
    "validate_schema",
    "check_authentication",
    "check_permissions",
    "check_rate_limit",
    "audit_log",
    "execute_tool",
)

AUDIT_STEP_LABELS = (
    "Validate Schema",
    "Check Authentication",
    "Check Permissions",
    "Check Rate Limit",
    "Audit Log",
    "Execute Tool",
)


def _new_audit_steps() -> list[dict]:
    """Create the six pipeline steps shown in agent_tool_flow.png."""
    return [
        {
            "step": number,
            "name": name,
            "label": label,
            "status": "skipped",
            "timestamp": None,
        }
        for number, (name, label) in enumerate(
            zip(AUDIT_STEP_NAMES, AUDIT_STEP_LABELS),
            start=1,
        )
    ]


def _mark_audit_step(
    steps: list[dict],
    step_number: int,
    status: str,
    *,
    error: str | None = None,
) -> None:
    """Record the outcome of one pipeline step without storing secrets."""
    step = steps[step_number - 1]
    step["status"] = status
    step["timestamp"] = datetime.now(timezone.utc).isoformat()
    if error:
        step["error"] = error


def audit_log(
    user: dict,
    tool_name: str,
    arguments: dict,
    result: Any = None,
    error: str = None,
    steps: list[dict] | None = None,
):
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
        "steps": deepcopy(steps) if steps is not None else _new_audit_steps(),
    }
    AUDIT_LOG.append(entry)
    return entry


# Step 6: invoke the handler.

def execute_tool(tool: ToolDefinition, arguments: dict) -> Any:
    """Invoke a tool handler with validated keyword arguments."""
    return tool.handler(**arguments)


class ToolRegistry:
    """Store tool definitions and mediate access to their handlers."""

    def __init__(self, audit_sink: Callable[[dict], Any] | None = None):
        self._tools: dict[str, ToolDefinition] = {}
        self._audit_entries: list[dict] = []
        self._audit_sink = audit_sink

    def _save_audit_entry(self, entry: dict) -> None:
        """Keep a local copy and synchronously persist it when configured."""
        self._audit_entries.append(entry)
        if self._audit_sink is not None:
            try:
                self._audit_sink(deepcopy(entry))
            except Exception as exc:
                raise AuditPersistenceError(
                    f"Could not persist audit log: {exc}"
                ) from exc

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

    def list_tools(self, principal: dict | None = None) -> list[dict]:
        """Return model-facing schemas allowed for the authenticated principal."""
        permissions = (
            set(principal.get("permissions", []))
            if isinstance(principal, dict)
            else set()
        )
        return [
            {
                "name": t.name,
                "description": t.description,
                "input_schema": t.input_schema,
            }
            for t in self._tools.values()
            if t.model_visible and set(t.required_permissions).issubset(permissions)
        ]

    def call(
        self,
        tool_name: str,
        arguments: dict,
        principal: dict,
        *,
        model_initiated: bool = False,
    ) -> dict:
        """Apply registry policy, execute a tool, and audit the outcome."""
        print(f"\n{'='*60}")
        print(f"  TOOL CALL: {tool_name}")
        arguments_json = json.dumps(arguments, ensure_ascii=False)
        print(f"  Arguments: {_console_safe(arguments_json)}")
        print(f"{'='*60}")

        user = {"user_id": "anonymous", "role": "unknown", "permissions": []}
        audit_arguments = arguments if isinstance(arguments, dict) else {}
        steps = _new_audit_steps()
        current_step = 1

        try:
            tool = self.get_tool(tool_name)
            if model_initiated and not tool.model_visible:
                raise PermissionError(
                    f"Tool '{tool_name}' is internal and cannot be called by the model"
                )
            validated_arguments = validate_schema(tool, arguments)
            _mark_audit_step(steps, 1, "success")
            audit_arguments = validated_arguments

            current_step = 2
            user = check_authentication(principal)
            _mark_audit_step(steps, 2, "success")

            current_step = 3
            check_permissions(user, tool)
            _mark_audit_step(steps, 3, "success")

            current_step = 4
            rate_limiter.check(user["user_id"])
            _mark_audit_step(steps, 4, "success")

            current_step = 5
            _mark_audit_step(steps, 5, "success")

            current_step = 6
            user_token = _current_user.set(user)
            try:
                result = execute_tool(tool, validated_arguments)
            finally:
                _current_user.reset(user_token)
            _mark_audit_step(steps, 6, "success")
            entry = audit_log(
                user,
                tool_name,
                validated_arguments,
                result=result,
                steps=steps,
            )
            self._save_audit_entry(entry)
            return {"result": result}
        except Exception as exc:
            if isinstance(exc, AuditPersistenceError):
                raise
            error_message = str(exc) or exc.__class__.__name__
            _mark_audit_step(steps, current_step, "error", error=error_message)
            if current_step < 5:
                _mark_audit_step(steps, 5, "success")
            entry = audit_log(
                user,
                tool_name,
                audit_arguments,
                error=error_message,
                steps=steps,
            )
            self._save_audit_entry(entry)
            return {
                "error": error_message,
                "error_type": exc.__class__.__name__,
            }

    def get_audit_log(self) -> list[dict]:
        """Return audit entries produced by this registry instance."""
        return deepcopy(self._audit_entries)
