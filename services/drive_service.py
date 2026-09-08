"""List and download files using the authenticated user's Drive grant."""

import os
import re
import tempfile
import threading
import time
from urllib.parse import parse_qs, urlparse
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload

from services.file_reader import MAX_FILE_SIZE_BYTES
from services.google_oauth import google_oauth_service

_CACHE_TTL_SECONDS = 60
_cache_lock = threading.Lock()
_file_cache: dict[tuple, tuple[float, list[dict]]] = {}

# Prevent folder IDs from altering the Drive query expression.
_DRIVE_ID_PATTERN = re.compile(r"^[A-Za-z0-9_-]+$")


def clear_drive_cache(user_id: str | None = None):
    """Clear cached file metadata, primarily for explicit refreshes and tests."""
    with _cache_lock:
        if user_id is None:
            _file_cache.clear()
        else:
            for key in [key for key in _file_cache if key[0] == user_id]:
                _file_cache.pop(key, None)


def _cached(key: tuple) -> list[dict] | None:
    now = time.monotonic()
    with _cache_lock:
        cached = _file_cache.get(key)
        if cached is None or cached[0] <= now:
            _file_cache.pop(key, None)
            return None
        return [dict(item) for item in cached[1]]


def _store_cache(key: tuple, files: list[dict]) -> list[dict]:
    with _cache_lock:
        _file_cache[key] = (
            time.monotonic() + _CACHE_TTL_SECONDS,
            [dict(item) for item in files],
        )
    return files


def _normalize_files(files: list[dict]) -> list[dict]:
    return [
        {
            "id": item["id"],
            "name": item["name"],
            "mimeType": item.get("mimeType", ""),
            "size": item.get("size", "unknown"),
            "modifiedTime": item.get("modifiedTime", ""),
        }
        for item in files
    ]


def _normalize_folder_id(folder_reference: str | None) -> str | None:
    """Return a Drive folder ID from either a raw ID or a Drive URL."""
    if not folder_reference:
        return None

    value = folder_reference.strip()
    if _DRIVE_ID_PATTERN.fullmatch(value):
        return value

    parsed = urlparse(value)
    if parsed.scheme in {"http", "https"} and parsed.hostname == "drive.google.com":
        folder_match = re.search(r"/folders/([A-Za-z0-9_-]+)", parsed.path)
        if folder_match:
            return folder_match.group(1)

        query_id = parse_qs(parsed.query).get("id", [None])[0]
        if query_id and _DRIVE_ID_PATTERN.fullmatch(query_id):
            return query_id

    raise ValueError(
        "folder_id must be a Drive folder ID or a drive.google.com folder URL"
    )


def _get_service(user_id: str):
    """Build a Drive client from the requesting user's encrypted OAuth grant."""
    if not user_id:
        raise ValueError("Authenticated user_id is required for Google Drive")
    credentials = google_oauth_service.credentials_for_user(user_id)
    return build(
        "drive", "v3", credentials=credentials, cache_discovery=False
    )


def list_files(
    user_id: str,
    folder_id: str | None = None,
    page_size: int = 100,
) -> list[dict]:
    """List every accessible Drive file, optionally within a folder.

    ``page_size`` controls the number of items requested per API call, not the
    total number returned. All pages are followed until ``nextPageToken`` is
    absent.
    """
    if isinstance(page_size, bool) or not isinstance(page_size, int):
        raise TypeError("page_size must be an integer")
    if not 1 <= page_size <= 1000:
        raise ValueError("page_size must be between 1 and 1000")

    effective_folder_id = _normalize_folder_id(folder_id)
    cache_key = (user_id, "list", effective_folder_id, page_size)
    cached = _cached(cache_key)
    if cached is not None:
        return cached

    service = _get_service(user_id)

    query_parts = ["trashed = false"]
    if effective_folder_id:
        query_parts.append(f"'{effective_folder_id}' in parents")

    query = " and ".join(query_parts)
    files: list[dict] = []
    page_token = None

    while True:
        results = service.files().list(
            q=query,
            spaces="drive",
            pageSize=page_size,
            pageToken=page_token,
            fields="nextPageToken, files(id, name, mimeType, size, modifiedTime)",
            orderBy="modifiedTime desc",
            includeItemsFromAllDrives=True,
            supportsAllDrives=True,
        ).execute()

        files.extend(results.get("files", []))
        page_token = results.get("nextPageToken")
        if not page_token:
            break

    return _store_cache(cache_key, _normalize_files(files))


def search_files(user_id: str, query: str, page_size: int = 100) -> list[dict]:
    """Search accessible Drive file names without recursively listing folders."""
    if isinstance(page_size, bool) or not isinstance(page_size, int):
        raise TypeError("page_size must be an integer")
    if not 1 <= page_size <= 1000:
        raise ValueError("page_size must be between 1 and 1000")
    terms = list(dict.fromkeys(re.findall(r"[\w.-]+", query.strip(), re.UNICODE)))
    terms = [term for term in terms if len(term) > 1][:6]
    if not terms:
        raise ValueError("A non-empty Drive file search query is required")

    normalized_query = " ".join(terms).casefold()
    cache_key = (user_id, "search", normalized_query, page_size)
    cached = _cached(cache_key)
    if cached is not None:
        return cached

    def escaped(value: str) -> str:
        return value.replace("\\", "\\\\").replace("'", "\\'")

    name_conditions = " or ".join(
        f"name contains '{escaped(term)}'" for term in terms
    )
    drive_query = f"trashed = false and ({name_conditions})"
    service = _get_service(user_id)
    files: list[dict] = []
    page_token = None
    while True:
        results = service.files().list(
            q=drive_query,
            spaces="drive",
            pageSize=page_size,
            pageToken=page_token,
            fields="nextPageToken, files(id, name, mimeType, size, modifiedTime)",
            orderBy="modifiedTime desc",
            includeItemsFromAllDrives=True,
            supportsAllDrives=True,
        ).execute()
        files.extend(results.get("files", []))
        page_token = results.get("nextPageToken")
        if not page_token:
            break
    return _store_cache(cache_key, _normalize_files(files))


def download_file(user_id: str, file_id: str) -> dict:
    """Download a file from Google Drive to a temp file. Returns metadata and temp path."""
    if not isinstance(file_id, str) or not _DRIVE_ID_PATTERN.fullmatch(file_id.strip()):
        raise ValueError("Invalid Google Drive file ID")
    service = _get_service(user_id)

    file_meta = service.files().get(
        fileId=file_id,
        fields="name,mimeType,size,capabilities(canDownload)",
        supportsAllDrives=True,
    ).execute()
    mime_type = file_meta.get("mimeType", "")
    file_name = file_meta.get("name", "")
    if file_meta.get("capabilities", {}).get("canDownload") is False:
        raise PermissionError("Google Drive does not allow downloading this file")
    declared_size = int(file_meta.get("size") or 0)
    if declared_size > MAX_FILE_SIZE_BYTES:
        raise ValueError(
            f"File is too large ({declared_size} bytes); "
            f"maximum is {MAX_FILE_SIZE_BYTES} bytes"
        )

    # Export Google-native files to formats supported by the file reader.
    export_map = {
        "application/vnd.google-apps.document": ("application/vnd.openxmlformats-officedocument.wordprocessingml.document", ".docx"),
        "application/vnd.google-apps.spreadsheet": ("application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", ".xlsx"),
        "application/vnd.google-apps.presentation": ("application/vnd.openxmlformats-officedocument.presentationml.presentation", ".pptx"),
    }

    if mime_type in export_map:
        export_mime, ext = export_map[mime_type]
        request = service.files().export_media(fileId=file_id, mimeType=export_mime)
    else:
        request = service.files().get_media(
            fileId=file_id, supportsAllDrives=True
        )
        ext = os.path.splitext(file_name)[1] or ".bin"

    tmp_path = ""
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
            tmp_path = tmp.name
            downloader = MediaIoBaseDownload(tmp, request)
            done = False
            while not done:
                _, done = downloader.next_chunk()
                if tmp.tell() > MAX_FILE_SIZE_BYTES:
                    raise ValueError(
                        f"Downloaded file exceeds the {MAX_FILE_SIZE_BYTES}-byte limit"
                    )
    except Exception:
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)
        raise

    return {
        "file_id": file_id,
        "file_name": file_name,
        "mime_type": mime_type,
        "temp_path": tmp_path,
    }
