"""Convert supported local files to Markdown with MarkItDown."""

import os
from pathlib import Path

from markitdown import MarkItDown

MAX_CHARS = 15000
MAX_FILE_SIZE_BYTES = 25 * 1024 * 1024

_converter = MarkItDown(enable_plugins=False)


def read_file(file_path: str) -> dict:
    """Read a local file and convert its content to Markdown using MarkItDown."""
    if not isinstance(file_path, str) or not file_path.strip():
        raise ValueError("file_path is required")

    path = Path(file_path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"File not found: '{file_path}'")
    if not path.is_file():
        raise ValueError(f"Path is not a file: '{file_path}'")

    file_size = os.path.getsize(path)
    if file_size > MAX_FILE_SIZE_BYTES:
        raise ValueError(
            f"File is too large ({file_size} bytes); "
            f"maximum is {MAX_FILE_SIZE_BYTES} bytes"
        )

    conversion = _converter.convert_local(str(path))
    content = getattr(conversion, "markdown", None)
    if content is None:
        content = getattr(conversion, "text_content", None)
    if content is None:
        raise ValueError(f"MarkItDown returned no content for '{path.name}'")

    total_characters = len(content)
    truncated = total_characters > MAX_CHARS

    return {
        "file_name": path.name,
        "content": content[:MAX_CHARS],
        "truncated": truncated,
        "total_characters": total_characters,
    }
