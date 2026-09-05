"""Expose long-term memory operations as agent tools."""

from dataclasses import dataclass
from datetime import datetime, timezone
from functools import lru_cache
import hashlib
import re
import uuid

import tiktoken

from config import (
    EMBEDDING_MODEL,
    MEMORY_CHUNK_OVERLAP_TOKENS,
    MEMORY_CHUNK_TOKENS,
)
from registry import get_current_user
from registry.models import ToolDefinition
from services import embedding, vectorstore


FACT_CATEGORIES = {"fact", "user_preference"}
_HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*$")
_LIST_RE = re.compile(r"^\s*(?:[-+*]|\d+[.)])\s+")
_FENCE_RE = re.compile(r"^\s*(```|~~~)")


@dataclass(frozen=True)
class _Block:
    text: str
    heading_path: tuple[tuple[int, str], ...] = ()


@dataclass(frozen=True)
class _MemoryChunk:
    text: str
    heading_paths: tuple[str, ...]
    token_count: int


@lru_cache(maxsize=1)
def _tokenizer():
    """Return the tokenizer used to enforce embedding-friendly chunk sizes."""
    try:
        return tiktoken.encoding_for_model(EMBEDDING_MODEL)
    except KeyError:
        return tiktoken.get_encoding("cl100k_base")


def _encode(text: str) -> list[int]:
    return _tokenizer().encode(text)


def _decode(tokens: list[int]) -> str:
    return _tokenizer().decode(tokens)


def _heading_label(path: tuple[tuple[int, str], ...]) -> str:
    return " > ".join(title for _, title in path)


def _heading_prefix(path: tuple[tuple[int, str], ...]) -> str:
    return "\n".join(f"{'#' * level} {title}" for level, title in path)


def _line_kind(line: str) -> str:
    stripped = line.strip()
    if _LIST_RE.match(line):
        return "list"
    if "|" in stripped:
        return "table"
    return "paragraph"


def _parse_markdown_blocks(text: str) -> list[_Block]:
    """Parse headings, paragraphs, lists, tables, and fenced code blocks."""
    lines = text.splitlines()
    heading_stack: list[tuple[int, str]] = []
    blocks: list[_Block] = []
    buffered: list[str] = []
    buffered_kind = ""
    buffered_path: tuple[tuple[int, str], ...] = ()
    fence_marker = ""

    def flush():
        nonlocal buffered, buffered_kind
        block_text = "\n".join(buffered).strip()
        if block_text:
            blocks.append(_Block(block_text, buffered_path))
        buffered = []
        buffered_kind = ""

    for line in lines:
        fence_match = _FENCE_RE.match(line)
        if fence_marker:
            buffered.append(line)
            if line.strip().startswith(fence_marker):
                flush()
                fence_marker = ""
            continue

        if fence_match:
            flush()
            fence_marker = fence_match.group(1)
            buffered_path = tuple(heading_stack)
            buffered_kind = "code"
            buffered = [line]
            continue

        heading_match = _HEADING_RE.match(line)
        if heading_match:
            flush()
            level = len(heading_match.group(1))
            title = heading_match.group(2).strip()
            heading_stack = [item for item in heading_stack if item[0] < level]
            heading_stack.append((level, title))
            continue

        if not line.strip():
            flush()
            continue

        kind = _line_kind(line)
        current_path = tuple(heading_stack)
        # Indented lines directly following a list remain part of that list.
        if buffered_kind == "list" and kind == "paragraph" and line[:1].isspace():
            kind = "list"
        if buffered and (kind != buffered_kind or current_path != buffered_path):
            flush()
        if not buffered:
            buffered_kind = kind
            buffered_path = current_path
        buffered.append(line)

    flush()
    return blocks


def _render_block(block: _Block) -> str:
    prefix = _heading_prefix(block.heading_path)
    return f"{prefix}\n\n{block.text}" if prefix else block.text


def _hard_split_block(block: _Block) -> list[_MemoryChunk]:
    """Split only an oversized block while repeating its heading context."""
    prefix = _heading_prefix(block.heading_path)
    prefix_tokens = _encode(f"{prefix}\n\n") if prefix else []
    capacity = MEMORY_CHUNK_TOKENS - len(prefix_tokens)
    if capacity <= 0:
        prefix_tokens = prefix_tokens[: MEMORY_CHUNK_TOKENS // 3]
        capacity = MEMORY_CHUNK_TOKENS - len(prefix_tokens)

    content_tokens = _encode(block.text)
    overlap = min(MEMORY_CHUNK_OVERLAP_TOKENS, max(0, capacity - 1))
    chunks = []
    start = 0
    while start < len(content_tokens):
        piece = content_tokens[start:start + capacity]
        tokens = prefix_tokens + piece
        text = _decode(tokens).strip()
        chunks.append(
            _MemoryChunk(
                text=text,
                heading_paths=(_heading_label(block.heading_path),)
                if block.heading_path else (),
                token_count=len(_encode(text)),
            )
        )
        if start + capacity >= len(content_tokens):
            break
        start += capacity - overlap
    return chunks


def _build_chunks(content: str, category: str = "general") -> list[_MemoryChunk]:
    """Build structure-aware chunks capped by tokens instead of characters."""
    text = content.strip()
    if not text:
        return []

    token_count = len(_encode(text))
    normalized_category = (category or "general").strip().lower()
    if normalized_category in FACT_CATEGORIES and token_count <= MEMORY_CHUNK_TOKENS:
        return [_MemoryChunk(text=text, heading_paths=(), token_count=token_count)]

    blocks = _parse_markdown_blocks(text)
    if not blocks:
        blocks = [_Block(text)]

    chunks: list[_MemoryChunk] = []
    current_parts: list[str] = []
    current_paths: list[str] = []

    def joined_current() -> str:
        return "\n\n".join(current_parts).strip()

    def flush_current():
        nonlocal current_parts, current_paths
        chunk_text = joined_current()
        if chunk_text:
            chunks.append(
                _MemoryChunk(
                    text=chunk_text,
                    heading_paths=tuple(dict.fromkeys(filter(None, current_paths))),
                    token_count=len(_encode(chunk_text)),
                )
            )
        current_parts = []
        current_paths = []

    for block in blocks:
        rendered = _render_block(block)
        rendered_tokens = _encode(rendered)
        if len(rendered_tokens) > MEMORY_CHUNK_TOKENS:
            flush_current()
            chunks.extend(_hard_split_block(block))
            continue

        candidate = "\n\n".join([*current_parts, rendered])
        if current_parts and len(_encode(candidate)) > MEMORY_CHUNK_TOKENS:
            previous_text = joined_current()
            previous_paths = list(current_paths)
            flush_current()

            available = MEMORY_CHUNK_TOKENS - len(rendered_tokens) - 2
            overlap_size = min(MEMORY_CHUNK_OVERLAP_TOKENS, max(0, available))
            while overlap_size:
                overlap_text = _decode(_encode(previous_text)[-overlap_size:]).strip()
                with_overlap = f"{overlap_text}\n\n{rendered}" if overlap_text else rendered
                if len(_encode(with_overlap)) <= MEMORY_CHUNK_TOKENS:
                    break
                overlap_size -= 1
            if overlap_size and overlap_text:
                current_parts.append(overlap_text)
                current_paths.extend(previous_paths)

        current_parts.append(rendered)
        current_paths.append(_heading_label(block.heading_path))

    flush_current()
    return chunks


def _chunk_content(content: str, category: str = "general") -> list[str]:
    """Return chunk texts; retained as a small compatibility helper."""
    return [chunk.text for chunk in _build_chunks(content, category)]


# Shared persistence path used by the two deliberately separate write APIs.

def _persist_memory(
    content: str,
    category: str = "general",
    topic: str = "",
    value: str = "",
    canonical_value: str = "",
    polarity: str = "",
    confidence: float = 0.0,
    source_id: str = "",
    source_type: str = "",
    file_id: str = "",
    file_name: str = "",
    content_hash: str = "",
) -> dict:
    """Persist already-classified content; callers own classification policy."""
    chunks = _build_chunks(content, category)
    if not chunks:
        raise ValueError("Memory content must be a non-empty string")
    user_id = get_current_user()["user_id"]
    normalized_category = (category or "general").strip().lower()
    normalized_topic = "_".join(topic.strip().casefold().split())
    normalized_value = " ".join(canonical_value.strip().casefold().split())
    normalized_hash = content_hash.strip().lower() or hashlib.sha256(
        content.encode("utf-8")
    ).hexdigest()
    if normalized_topic and normalized_value:
        identity = f"{normalized_category}:{normalized_topic}:{normalized_value}"
    else:
        identity = f"{normalized_category}:{normalized_hash}"
    memory_key = hashlib.sha256(identity.encode("utf-8")).hexdigest()
    resolved_source_id = source_id.strip() or str(uuid.uuid4())
    created_at = datetime.now(timezone.utc).isoformat()
    vectors = embedding.embed_texts([chunk.text for chunk in chunks])
    records = []
    for index, (chunk, vector) in enumerate(zip(chunks, vectors)):
        metadata = {
            "user_id": user_id,
            "category": category or "general",
            "created_at": created_at,
            "source_id": resolved_source_id,
            "source_type": source_type.strip(),
            "file_id": file_id.strip(),
            "file_name": file_name.strip(),
            "content_hash": normalized_hash,
            "chunk_index": index,
            "chunk_count": len(chunks),
            "token_count": chunk.token_count,
            "heading_paths": list(chunk.heading_paths),
            "heading_path": chunk.heading_paths[-1] if chunk.heading_paths else "",
            "chunk_strategy": "markdown_token",
            "memory_key": memory_key,
            "memory_type": "preference" if normalized_category == "user_preference" else normalized_category,
            "topic": normalized_topic,
            "value": value.strip(),
            "canonical_value": normalized_value,
            "polarity": polarity.strip().lower(),
            "confidence": confidence,
            "extraction_source": "fci" if normalized_topic else "tool",
            "active": True,
        }
        records.append((chunk.text, vector, metadata))
    if len(records) == 1:
        text, vector, metadata = records[0]
        saved = [vectorstore.save_memory(text, vector, metadata)]
    else:
        saved = vectorstore.save_memories(records)
    return {
        "status": "saved",
        "source_id": resolved_source_id,
        "category": category or "general",
        "chunks_saved": len(saved),
    }


def upsert_user_memory(
    content: str,
    category: str,
    topic: str,
    value: str,
    canonical_value: str,
    polarity: str,
    confidence: float,
) -> dict:
    """Internal structured upsert used only by the FCI extraction pipeline."""
    normalized_category = category.strip().lower()
    if normalized_category not in FACT_CATEGORIES:
        raise ValueError("Structured user memory category must be fact or user_preference")
    if not topic.strip() or not canonical_value.strip():
        raise ValueError("Structured user memory requires topic and canonical_value")
    if polarity not in {"like", "dislike", "neutral"}:
        raise ValueError("Invalid structured user memory polarity")
    if normalized_category == "fact" and polarity != "neutral":
        raise ValueError("Facts must use neutral polarity")
    if normalized_category == "user_preference" and polarity == "neutral":
        raise ValueError("Preferences must use like or dislike polarity")
    if not 0 <= confidence <= 1:
        raise ValueError("Confidence must be between 0 and 1")
    return _persist_memory(
        content=content,
        category=normalized_category,
        topic=topic,
        value=value,
        canonical_value=canonical_value,
        polarity=polarity,
        confidence=confidence,
    )


upsert_user_memory_tool = ToolDefinition(
    name="upsert_user_memory",
    description=(
        "Internal structured upsert for facts and preferences extracted by FCI. "
        "This operation is not available to the chat model."
    ),
    input_schema={
        "type": "object",
        "properties": {
            "content": {
                "type": "string",
                "description": "The information to remember.",
            },
            "category": {
                "type": "string",
                "enum": ["fact", "user_preference"],
                "description": "Structured user-memory category.",
            },
            "topic": {
                "type": "string",
                "description": "Normalized semantic topic for a structured fact or preference.",
            },
            "value": {
                "type": "string",
                "description": "Human-readable structured value.",
            },
            "canonical_value": {
                "type": "string",
                "description": "Normalized value used to update the same memory slot.",
            },
            "polarity": {
                "type": "string",
                "enum": ["like", "dislike", "neutral"],
                "description": "Preference direction, or neutral for a fact.",
            },
            "confidence": {
                "type": "number",
                "description": "Extraction confidence between 0 and 1.",
            },
        },
        "required": [
            "content",
            "category",
            "topic",
            "value",
            "canonical_value",
            "polarity",
            "confidence",
        ],
    },
    required_permissions=["memory:write"],
    handler=upsert_user_memory,
    model_visible=False,
)


DOCUMENT_CATEGORIES = {"document", "note", "task"}


def save_document_memory(
    content: str,
    category: str = "document",
    *,
    source_id: str = "",
    source_type: str = "",
    file_id: str = "",
    file_name: str = "",
    content_hash: str = "",
) -> dict:
    """Save explicit user-requested content without classifying personal facts."""
    normalized_category = category.strip().lower()
    if normalized_category not in DOCUMENT_CATEGORIES:
        raise ValueError("Document memory category must be document, note, or task")
    return _persist_memory(
        content=content,
        category=normalized_category,
        source_id=source_id,
        source_type=source_type,
        file_id=file_id,
        file_name=file_name,
        content_hash=content_hash,
    )


save_document_memory_tool = ToolDefinition(
    name="save_document_memory",
    description=(
        "Save file contents, displayed text, notes, or reusable task results when "
        "the user explicitly asks to retain them. Do not use this for personal "
        "facts or preferences; those are extracted automatically. Long content is "
        "split into Markdown-aware, token-limited RAG chunks."
    ),
    input_schema={
        "type": "object",
        "properties": {
            "content": {
                "type": "string",
                "description": "Complete content that the user explicitly asked to save.",
            },
            "category": {
                "type": "string",
                "enum": ["document", "note", "task"],
                "description": "Kind of non-personal memory; defaults to document.",
            },
        },
        "required": ["content"],
    },
    required_permissions=["memory:write"],
    handler=save_document_memory,
    model_visible=False,
)


# Retrieve memories by semantic similarity.

def search_memory(query: str, top_k: int = 5) -> dict:
    """Search long-term memory for relevant information."""
    if top_k < 1 or top_k > 20:
        raise ValueError("top_k must be between 1 and 20")
    query_vector = embedding.embed_query(query)
    user_id = get_current_user()["user_id"]
    memories = vectorstore.search_memory(
        query_vector,
        top_k=top_k,
        query_text=query,
        user_id=user_id,
    )
    return {
        "query": query,
        "total_results": len(memories),
        "memories": memories,
    }


search_memory_tool = ToolDefinition(
    name="search_memory",
    description=(
        "Search long-term memory for previously saved information. "
        "Uses semantic search to find the most relevant memories. "
        "Use this to recall facts, user preferences, or past interactions."
    ),
    input_schema={
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "Search query to find relevant memories.",
            },
            "top_k": {
                "type": "integer",
                "description": "Number of results to return (default: 5).",
            },
        },
        "required": ["query"],
    },
    required_permissions=["memory:read"],
    handler=search_memory,
)


ALL_MEMORY_TOOLS = [
    upsert_user_memory_tool,
    save_document_memory_tool,
    search_memory_tool,
]
