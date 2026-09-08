"""Render model Markdown as sanitized HTML for the web chat."""

import html
import re

import bleach
import markdown


ALLOWED_TAGS = {
    "a", "blockquote", "br", "code", "del", "em", "h1", "h2", "h3",
    "h4", "h5", "h6", "hr", "li", "ol", "p", "pre", "strong",
    "table", "tbody", "td", "th", "thead", "tr", "ul",
}
ALLOWED_ATTRIBUTES = {
    "a": ["href", "title"],
    "code": ["class"],
    "th": ["align"],
    "td": ["align"],
}

_SIMPLE_LATEX = {
    r"$\rightarrow$": "→",
    r"$\Rightarrow$": "⇒",
    r"$\leftarrow$": "←",
    r"$\Leftarrow$": "⇐",
    r"$\leftrightarrow$": "↔",
}


def normalize_chat_markdown(text: str) -> str:
    """Normalize common conversion artifacts without changing Markdown syntax."""
    normalized = text or ""
    normalized = re.sub(
        r"&+(?:amp;)?(?:#x20|#32|nbsp);",
        " ",
        normalized,
        flags=re.IGNORECASE,
    )
    # Decode nested entities such as &amp;#x20; while sanitizing after rendering.
    normalized = html.unescape(html.unescape(normalized))
    for expression, symbol in _SIMPLE_LATEX.items():
        normalized = normalized.replace(expression, symbol)
    return normalized


def render_chat_markdown(text: str) -> str:
    """Convert Markdown to an allowlisted HTML fragment for the browser."""
    rendered = markdown.markdown(
        normalize_chat_markdown(text),
        extensions=["extra", "sane_lists", "nl2br"],
        output_format="html",
    )
    return bleach.clean(
        rendered,
        tags=ALLOWED_TAGS,
        attributes=ALLOWED_ATTRIBUTES,
        protocols={"http", "https", "mailto"},
        strip=True,
        strip_comments=True,
    )
