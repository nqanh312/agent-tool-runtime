"""Expose local file-to-Markdown conversion as an agent tool."""

from fnmatch import fnmatch
from pathlib import Path

from config import LOCAL_FILE_ALLOWED_ROOTS
from registry.models import ToolDefinition
from services.file_reader import read_file


_BLOCKED_FILE_PATTERNS = (
    ".env",
    ".env.*",
    "credentials*.json",
    "client_secret*.json",
    "service-account*.json",
    "service_account*.json",
    "token.json",
    "*.key",
    "*.pem",
    "*.p12",
    "*.pfx",
    "id_rsa",
    "id_ed25519",
    "authorized_keys",
)
_BLOCKED_DIRECTORY_NAMES = {".git", ".ssh", ".aws", ".azure", ".kube"}


def _allowed_local_path(file_path: str) -> Path:
    """Resolve a CLI path inside an explicit root and reject secret material."""
    if not isinstance(file_path, str) or not file_path.strip():
        raise ValueError("file_path is required")
    if not LOCAL_FILE_ALLOWED_ROOTS:
        raise PermissionError(
            "Local file access is disabled; configure LOCAL_FILE_ALLOWED_ROOTS"
        )

    candidate = Path(file_path).expanduser()
    try:
        resolved = candidate.resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        raise FileNotFoundError(f"File not found: '{file_path}'") from exc

    matching_root = None
    for configured_root in LOCAL_FILE_ALLOWED_ROOTS:
        try:
            root = Path(configured_root).expanduser().resolve(strict=True)
        except (OSError, RuntimeError):
            continue
        if resolved.is_relative_to(root):
            matching_root = root
            break
    if matching_root is None:
        raise PermissionError("Local file path is outside the configured roots")

    relative_parts = resolved.relative_to(matching_root).parts
    if any(part.casefold() in _BLOCKED_DIRECTORY_NAMES for part in relative_parts[:-1]):
        raise PermissionError("Local file path is in a sensitive directory")
    lowered_name = resolved.name.casefold()
    if any(fnmatch(lowered_name, pattern) for pattern in _BLOCKED_FILE_PATTERNS):
        raise PermissionError("Sensitive local files cannot be read")
    return resolved


def read_local_file(file_path: str) -> dict:
    """Read an allowlisted local file and convert it to Markdown."""
    return read_file(str(_allowed_local_path(file_path)))


read_file_tool = ToolDefinition(
    name="read_file",
    description=(
        "Read an allowlisted local file and convert its content to Markdown. "
        "Supports many formats: PDF, DOCX, XLSX, PPTX, images, text files, and more. "
        "This capability is available only in the explicitly configured CLI workspace."
    ),
    input_schema={
        "type": "object",
        "properties": {
            "file_path": {
                "type": "string",
                "description": "A path inside an administrator-configured local root.",
            },
        },
        "required": ["file_path"],
    },
    required_permissions=["local_file:read"],
    handler=read_local_file,
)


ALL_READ_FILE_TOOLS = [read_file_tool]
