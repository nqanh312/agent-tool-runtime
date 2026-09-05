"""Expose Google Drive operations as agent tools."""

import os

from registry.models import ToolDefinition
from services import drive_service
from services.file_reader import read_file


# List files visible to the configured service account.

def list_drive_files(folder_id: str = "") -> dict:
    """List all files in Google Drive."""
    fid = folder_id if folder_id else None
    files = drive_service.list_files(folder_id=fid)
    return {
        "total_files": len(files),
        "files": files,
    }


list_files_tool = ToolDefinition(
    name="list_drive_files",
    description=(
        "List all files in Google Drive. "
        "Returns file names, IDs, types, sizes, and modification times. "
        "Optionally provide a folder_id to list files in a specific folder."
    ),
    input_schema={
        "type": "object",
        "properties": {
            "folder_id": {
                "type": "string",
                "description": "Optional Google Drive folder ID to list files from. Leave empty for default folder.",
            },
        },
        "required": [],
    },
    required_scopes=["drive:read"],
    handler=list_drive_files,
)


# Download and convert one Drive file.

def get_drive_file(file_id: str) -> dict:
    """Download one Drive file and return its Markdown content."""
    download = drive_service.download_file(file_id=file_id)
    temp_path = download["temp_path"]
    try:
        result = read_file(temp_path)
    finally:
        os.unlink(temp_path)
    return {
        "file_id": download["file_id"],
        "file_name": download["file_name"],
        "mime_type": download["mime_type"],
        "content": result["content"],
        "truncated": result["truncated"],
        "total_characters": result["total_characters"],
    }


read_file_tool = ToolDefinition(
    name="get_drive_file",
    description=(
        "Download and read one Google Drive file by its file ID. "
        "The result contains Markdown content suitable for displaying in chat. "
        "Supports PDF, DOCX, XLS/XLSX, PPTX, Google Docs/Sheets/Slides, "
        "HTML, CSV, JSON, XML, and text files. "
        "Always use list_drive_files first to find the correct file ID."
    ),
    input_schema={
        "type": "object",
        "properties": {
            "file_id": {
                "type": "string",
                "description": "The Google Drive file ID to read.",
            },
        },
        "required": ["file_id"],
    },
    required_scopes=["drive:read"],
    handler=get_drive_file,
)

# Backward-compatible Python alias. The model-facing tool name is get_drive_file.
read_drive_file = get_drive_file


ALL_DRIVE_TOOLS = [list_files_tool, read_file_tool]
