"""Tests for safe Markdown rendering in the chat UI."""

import unittest

from services.chat_renderer import render_chat_markdown


class ChatRendererTests(unittest.TestCase):
    def test_renders_headings_tables_entities_and_arrows(self):
        source = r"""### Demo

Save $\rightarrow$ search&#x20;

| Tool | Result |
| --- | --- |
| get_drive_file | Success |
"""

        rendered = render_chat_markdown(source)

        self.assertIn("<h3>Demo</h3>", rendered)
        self.assertIn("Save → search", rendered)
        self.assertNotIn("&#x20;", rendered)
        self.assertIn("<table>", rendered)
        self.assertIn("<td>get_drive_file</td>", rendered)

    def test_removes_unsafe_html_and_javascript_links(self):
        rendered = render_chat_markdown(
            '<script>alert("xss")</script> [bad](javascript:alert(1))'
        )

        self.assertNotIn("<script", rendered)
        self.assertNotIn("javascript:", rendered)


if __name__ == "__main__":
    unittest.main()
