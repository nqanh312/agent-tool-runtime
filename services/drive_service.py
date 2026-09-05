"""List and download files through the Google Drive API."""

import io
import os
import re
import tempfile
import threading
import time
from urllib.parse import parse_qs, urlparse
from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload

from config import GOOGLE_SERVICE_ACCOUNT_FILE, GOOGLE_DRIVE_FOLDER_ID

SCOPES = ["https://www.googleapis.com/auth/drive.readonly"]

_service = None
_CACHE_TTL_SECONDS = 60
_cache_lock = threading.Lock()
_file_cache: dict[tuple, tuple[float, list[dict]]] = {}

# Prevent folder IDs from altering the Drive query expression.
_DRIVE_ID_PATTERN = re.compile(r"^[A-Za-z0-9_-]+$")


def clear_drive_cache():
    """Clear cached file metadata, primarily for explicit refreshes and tests."""
    with _cache_lock:
        _file_cache.clear()


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


def _get_service():
    """Create the Drive client lazily and reuse it for later requests."""
    global _service
    if _service is None:
        creds_path = GOOGLE_SERVICE_ACCOUNT_FILE
        if not os.path.exists(creds_path):
            raise FileNotFoundError(
                f"Google Service Account file not found: '{creds_path}'. "
                f"Download it from Google Cloud Console and place it in the project root."
            )
        creds = service_account.Credentials.from_service_account_file(creds_path, scopes=SCOPES)
        _service = build("drive", "v3", credentials=creds)
    return _service


def list_files(folder_id: str | None = None, page_size: int = 100) -> list[dict]:
    """List every accessible Drive file, optionally within a folder.

    ``page_size`` controls the number of items requested per API call, not the
    total number returned. All pages are followed until ``nextPageToken`` is
    absent.
    """
    if isinstance(page_size, bool) or not isinstance(page_size, int):
        raise TypeError("page_size must be an integer")
    if not 1 <= page_size <= 1000:
        raise ValueError("page_size must be between 1 and 1000")

    effective_folder_id = _normalize_folder_id(
        folder_id or GOOGLE_DRIVE_FOLDER_ID or None
    )
    cache_key = ("list", effective_folder_id, page_size)
    cached = _cached(cache_key)
    if cached is not None:
        return cached

    service = _get_service()

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


def search_files(query: str, page_size: int = 100) -> list[dict]:
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
    cache_key = ("search", normalized_query, page_size)
    cached = _cached(cache_key)
    if cached is not None:
        return cached

    def escaped(value: str) -> str:
        return value.replace("\\", "\\\\").replace("'", "\\'")

    name_conditions = " or ".join(
        f"name contains '{escaped(term)}'" for term in terms
    )
    drive_query = f"trashed = false and ({name_conditions})"
    service = _get_service()
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


def download_file(file_id: str) -> dict:
    """Download a file from Google Drive to a temp file. Returns metadata and temp path."""
    service = _get_service()

    file_meta = service.files().get(fileId=file_id, fields="name, mimeType").execute()
    mime_type = file_meta.get("mimeType", "")
    file_name = file_meta.get("name", "")

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
        request = service.files().get_media(fileId=file_id)
        ext = os.path.splitext(file_name)[1] or ".bin"

    buffer = io.BytesIO()
    downloader = MediaIoBaseDownload(buffer, request)
    done = False
    while not done:
        _, done = downloader.next_chunk()

    with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
        tmp.write(buffer.getvalue())
        tmp_path = tmp.name

    return {
        "file_id": file_id,
        "file_name": file_name,
        "mime_type": mime_type,
        "temp_path": tmp_path,
    }
