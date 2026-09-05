"""Tests for Google Drive listing and tool response formatting."""

import unittest
from unittest.mock import patch

from services import drive_service
from tools import google_drive


class _FakeRequest:
    def __init__(self, response):
        self._response = response

    def execute(self):
        return self._response


class _FakeFilesResource:
    def __init__(self):
        self.calls = []

    def list(self, **kwargs):
        self.calls.append(kwargs)
        if kwargs["pageToken"] is None:
            return _FakeRequest(
                {
                    "files": [
                        {
                            "id": "file-1",
                            "name": "First document",
                            "mimeType": "text/plain",
                            "size": "10",
                            "modifiedTime": "2026-01-02T00:00:00Z",
                        }
                    ],
                    "nextPageToken": "page-2",
                }
            )

        return _FakeRequest(
            {
                "files": [
                    {
                        "id": "file-2",
                        "name": "Second document",
                        "mimeType": "application/pdf",
                    }
                ]
            }
        )


class _FakeDriveService:
    def __init__(self):
        self.files_resource = _FakeFilesResource()

    def files(self):
        return self.files_resource


class ListFilesTests(unittest.TestCase):
    def setUp(self):
        drive_service.clear_drive_cache()

    def test_lists_every_page_in_a_folder(self):
        fake_service = _FakeDriveService()

        with patch.object(drive_service, "_get_service", return_value=fake_service):
            files = drive_service.list_files(folder_id="folder_123", page_size=50)

        self.assertEqual([item["id"] for item in files], ["file-1", "file-2"])
        self.assertEqual(files[1]["size"], "unknown")
        self.assertEqual(files[1]["modifiedTime"], "")

        calls = fake_service.files_resource.calls
        self.assertEqual(len(calls), 2)
        self.assertIsNone(calls[0]["pageToken"])
        self.assertEqual(calls[1]["pageToken"], "page-2")
        self.assertEqual(
            calls[0]["q"],
            "trashed = false and 'folder_123' in parents",
        )
        self.assertTrue(calls[0]["includeItemsFromAllDrives"])
        self.assertIn("nextPageToken", calls[0]["fields"])

    def test_rejects_invalid_folder_id(self):
        with self.assertRaisesRegex(ValueError, "Drive folder ID"):
            drive_service.list_files(folder_id="folder' or trashed = true")

    def test_accepts_drive_folder_url(self):
        fake_service = _FakeDriveService()

        with patch.object(drive_service, "_get_service", return_value=fake_service):
            drive_service.list_files(
                folder_id="https://drive.google.com/drive/folders/folder_123?usp=sharing"
            )

        self.assertEqual(
            fake_service.files_resource.calls[0]["q"],
            "trashed = false and 'folder_123' in parents",
        )

    def test_rejects_page_size_outside_drive_limits(self):
        with self.assertRaisesRegex(ValueError, "between 1 and 1000"):
            drive_service.list_files(page_size=1001)

    def test_reuses_cached_folder_listing(self):
        fake_service = _FakeDriveService()
        with patch.object(drive_service, "_get_service", return_value=fake_service):
            first = drive_service.list_files(folder_id="folder_cached")
            second = drive_service.list_files(folder_id="folder_cached")

        self.assertEqual(first, second)
        self.assertEqual(len(fake_service.files_resource.calls), 2)

    def test_searches_file_names_without_listing_folders(self):
        fake_service = _FakeDriveService()
        with patch.object(drive_service, "_get_service", return_value=fake_service):
            files = drive_service.search_files("Drive Agent")

        self.assertEqual(len(files), 2)
        query = fake_service.files_resource.calls[0]["q"]
        self.assertIn("name contains 'Drive'", query)
        self.assertIn("name contains 'Agent'", query)
        self.assertNotIn("in parents", query)

    def test_tool_returns_count_and_files(self):
        expected_files = [{"id": "file-1", "name": "Document"}]

        with patch.object(
            google_drive.drive_service,
            "list_files",
            return_value=expected_files,
        ) as list_files:
            result = google_drive.list_drive_files(folder_id="folder_123")

        list_files.assert_called_once_with(folder_id="folder_123")
        self.assertEqual(
            result,
            {"total_files": 1, "files": expected_files},
        )

    def test_search_tool_returns_query_and_files(self):
        expected_files = [{"id": "file-1", "name": "Assignment.pptx"}]
        with patch.object(
            google_drive.drive_service,
            "search_files",
            return_value=expected_files,
        ) as search_files:
            result = google_drive.search_drive_files("Assignment")

        search_files.assert_called_once_with(query="Assignment")
        self.assertEqual(
            result,
            {
                "query": "Assignment",
                "total_files": 1,
                "files": expected_files,
            },
        )

    def test_get_drive_file_converts_content_and_removes_temp_file(self):
        download = {
            "file_id": "file-1",
            "file_name": "Document.txt",
            "mime_type": "text/plain",
            "temp_path": "temporary-file.txt",
        }
        converted = {
            "file_name": "temporary-file.txt",
            "content": "File content",
            "truncated": False,
            "total_characters": 12,
        }

        with (
            patch.object(
                google_drive.drive_service,
                "download_file",
                return_value=download,
            ),
            patch.object(google_drive, "read_file", return_value=converted),
            patch.object(google_drive.os, "unlink") as unlink,
        ):
            result = google_drive.get_drive_file("file-1")

        unlink.assert_called_once_with("temporary-file.txt")
        self.assertEqual(result["file_name"], "Document.txt")
        self.assertEqual(result["content"], "File content")
        self.assertFalse(result["truncated"])
        self.assertEqual(google_drive.read_file_tool.name, "get_drive_file")


if __name__ == "__main__":
    unittest.main()
