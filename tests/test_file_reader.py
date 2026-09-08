"""Tests for the filesystem conversion used by Google Drive downloads."""

from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from services import file_reader


class _FakeConversion:
    def __init__(self, markdown):
        self.markdown = markdown


class _FakeConverter:
    def __init__(self, markdown):
        self.markdown = markdown

    def convert_local(self, _file_path):
        return _FakeConversion(self.markdown)


class FileReaderTests(unittest.TestCase):
    def test_converts_a_text_file_to_markdown(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "example.txt"
            path.write_text("Hello from Drive", encoding="utf-8")

            result = file_reader.read_file(str(path))

        self.assertEqual(result["file_name"], "example.txt")
        self.assertIn("Hello from Drive", result["content"])
        self.assertFalse(result["truncated"])
        self.assertEqual(result["total_characters"], len(result["content"]))

    def test_reports_when_converted_content_is_truncated(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "long.txt"
            path.write_text("source", encoding="utf-8")

            with (
                patch.object(file_reader, "MAX_CHARS", 5),
                patch.object(file_reader, "_converter", _FakeConverter("123456789")),
            ):
                result = file_reader.read_file(str(path))

        self.assertEqual(result["content"], "12345")
        self.assertTrue(result["truncated"])
        self.assertEqual(result["total_characters"], 9)

    def test_rejects_files_over_the_size_limit(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "large.txt"
            path.write_text("too large", encoding="utf-8")

            with patch.object(file_reader, "MAX_FILE_SIZE_BYTES", 1):
                with self.assertRaisesRegex(ValueError, "File is too large"):
                    file_reader.read_file(str(path))


if __name__ == "__main__":
    unittest.main()
