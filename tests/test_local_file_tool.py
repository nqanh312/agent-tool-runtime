"""Security boundary tests for the CLI-only local file tool."""

from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from tools import read_file as local_file_tool


class LocalFileToolTests(unittest.TestCase):
    def test_fails_closed_without_configured_roots(self):
        with patch.object(local_file_tool, "LOCAL_FILE_ALLOWED_ROOTS", ()):
            with self.assertRaisesRegex(PermissionError, "disabled"):
                local_file_tool.read_local_file("anything.txt")

    def test_reads_only_inside_configured_root(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "uploads"
            root.mkdir()
            document = root / "notes.txt"
            document.write_text("safe", encoding="utf-8")
            expected = {"content": "safe"}
            with (
                patch.object(local_file_tool, "LOCAL_FILE_ALLOWED_ROOTS", (str(root),)),
                patch.object(local_file_tool, "read_file", return_value=expected) as reader,
            ):
                result = local_file_tool.read_local_file(str(document))

            self.assertEqual(result, expected)
            reader.assert_called_once_with(str(document.resolve()))

    def test_rejects_path_outside_configured_root(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            root = base / "uploads"
            root.mkdir()
            outside = base / "secret.txt"
            outside.write_text("secret", encoding="utf-8")
            with patch.object(
                local_file_tool, "LOCAL_FILE_ALLOWED_ROOTS", (str(root),)
            ):
                with self.assertRaisesRegex(PermissionError, "outside"):
                    local_file_tool.read_local_file(str(root / ".." / "secret.txt"))

    def test_rejects_sensitive_file_inside_root(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            secret = root / ".env"
            secret.write_text("TOKEN=secret", encoding="utf-8")
            with patch.object(
                local_file_tool, "LOCAL_FILE_ALLOWED_ROOTS", (str(root),)
            ):
                with self.assertRaisesRegex(PermissionError, "Sensitive"):
                    local_file_tool.read_local_file(str(secret))


if __name__ == "__main__":
    unittest.main()
