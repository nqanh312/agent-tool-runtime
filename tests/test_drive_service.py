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
        with self.assertRaisesRegex(ValueError, "invalid characters"):
            drive_service.list_files(folder_id="folder' or trashed = true")

    def test_rejects_page_size_outside_drive_limits(self):
        with self.assertRaisesRegex(ValueError, "between 1 and 1000"):
            drive_service.list_files(page_size=1001)

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


if __name__ == "__main__":
    unittest.main()
