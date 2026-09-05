"""Expose the agent through a FastAPI service and a simple web UI."""

import json
import threading
import traceback
import uuid
import uvicorn
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from agent import Agent
from config import SERVICE_API_KEY
from registry.registry import check_authentication
from services.chat_renderer import render_chat_markdown
from services.conversations import (
    ConversationNotFoundError,
    conversation_repository,
)
from services.vectorstore import list_all_memories

app = FastAPI(title="AI Agent - Assignment 1")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Agent instances are an optimization. PostgreSQL remains the source of truth.
sessions: dict[str, Agent] = {}
conversation_locks: dict[str, threading.Lock] = {}
conversation_locks_guard = threading.Lock()


def get_agent(session_id: str) -> Agent:
    """Return a legacy in-memory session for backward-compatible clients."""
    key = f"legacy:{session_id}"
    if key not in sessions:
        sessions[key] = Agent(service_api_key=SERVICE_API_KEY)
    return sessions[key]


def _conversation_lock(key: str) -> threading.Lock:
    with conversation_locks_guard:
        return conversation_locks.setdefault(key, threading.Lock())


def _current_user() -> dict:
    return check_authentication(SERVICE_API_KEY)


def _get_conversation_agent(
    conversation_id: str,
    user_id: str,
    *,
    before_ordinal: int | None = None,
    force_reload: bool = False,
) -> Agent:
    key = f"conversation:{conversation_id}"
    if force_reload:
        sessions.pop(key, None)
    if key not in sessions:
        stored = conversation_repository.list_context_messages(
            conversation_id,
            user_id,
            before_ordinal=before_ordinal,
        )
        history = [
            {"role": item["role"], "content": item["content"]}
            for item in stored
            if item["role"] in {"user", "assistant"}
        ]
        artifact = conversation_repository.get_last_artifact(
            conversation_id, user_id
        )
        sessions[key] = Agent(
            service_api_key=SERVICE_API_KEY,
            conversation_history=history,
            last_artifact=artifact,
        )
    return sessions[key]


# Request and response models

class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=100_000)
    conversation_id: str | None = None
    client_message_id: str | None = None
    session_id: str | None = None


class ClearRequest(BaseModel):
    session_id: str | None = None
    conversation_id: str | None = None


class ChatResponse(BaseModel):
    response: str
    response_html: str = ""
    tools_used: list[str] = Field(default_factory=list)
    conversation_id: str | None = None
    title: str = ""
    message_id: str | None = None
    created_at: str = ""


class ConversationSummary(BaseModel):
    id: str
    title: str
    created_at: str
    updated_at: str


class ConversationListResponse(BaseModel):
    conversations: list[ConversationSummary]
    next_cursor: str | None = None


class StoredMessage(BaseModel):
    id: str
    conversation_id: str
    ordinal: int
    role: str
    content: str
    response_html: str = ""
    tools_used: list[str] = Field(default_factory=list)
    created_at: str


class MessageListResponse(BaseModel):
    messages: list[StoredMessage]
    next_before: int | None = None


class MemoryFact(BaseModel):
    id: str
    text: str
    category: str
    created_at: str = ""
    topic: str = ""
    value: str = ""
    polarity: str = ""
    confidence: float = 0.0


class MemoryFactsResponse(BaseModel):
    total: int
    facts: list[MemoryFact]


class MemoryDocument(BaseModel):
    source_id: str
    file_id: str = ""
    file_name: str = ""
    category: str
    created_at: str = ""
    content_hash: str = ""
    chunk_count: int = 0
    stored_chunks: int = 0
    preview: str = ""


class MemoryDocumentsResponse(BaseModel):
    total: int
    documents: list[MemoryDocument]


# API endpoints

@app.post("/api/chat")
def chat(req: ChatRequest):
    """Process one message and report tools used during the turn."""
    # Compatibility for the original client and external assignment tests.
    if req.session_id and not req.conversation_id and not req.client_message_id:
        try:
            agent = get_agent(req.session_id)
            audit_before = len(agent.get_audit_log())
            response = agent.run(req.message)
            tools_used = [
                log["tool"] for log in agent.get_audit_log()[audit_before:]
            ]
            return ChatResponse(
                response=response,
                response_html=render_chat_markdown(response),
                tools_used=tools_used,
            )
        except Exception as exc:
            traceback.print_exc()
            return JSONResponse(
                status_code=500,
                content={
                    "response": f"Error: {exc}",
                    "response_html": "",
                    "tools_used": [],
                },
            )

    user = _current_user()
    client_message_id = req.client_message_id or str(uuid.uuid4())
    lock_key = req.conversation_id or f"new:{client_message_id}"
    lock = _conversation_lock(lock_key)
    if not lock.acquire(blocking=False):
        raise HTTPException(
            status_code=409,
            detail="Another message is already running for this conversation",
        )
    try:
        conversation = None
        user_message = None
        new_conversation = False

        # A retry can arrive before the browser received the newly-created ID.
        retried = conversation_repository.find_user_message(
            user["user_id"], client_message_id
        )
        if retried is not None:
            user_message, conversation = retried
        elif req.conversation_id:
            conversation = conversation_repository.get_conversation(
                req.conversation_id, user["user_id"]
            )
        else:
            conversation, user_message = (
                conversation_repository.create_conversation_with_message(
                    user["user_id"], req.message, client_message_id
                )
            )
            new_conversation = True

        conversation_id = conversation["id"]
        if new_conversation:
            agent = _get_conversation_agent(
                conversation_id,
                user["user_id"],
                before_ordinal=user_message["ordinal"],
            )
        elif user_message is None:
            # Hydrate before the new user message is stored to avoid duplication.
            agent = _get_conversation_agent(
                conversation_id, user["user_id"]
            )
            user_message, _ = conversation_repository.append_user_message(
                conversation_id,
                user["user_id"],
                req.message,
                client_message_id,
            )
        else:
            existing_reply = conversation_repository.get_reply(user_message["id"])
            if existing_reply is not None:
                return ChatResponse(
                    response=existing_reply["content"],
                    response_html=render_chat_markdown(existing_reply["content"]),
                    tools_used=existing_reply["tools_used"],
                    conversation_id=conversation_id,
                    title=conversation["title"],
                    message_id=existing_reply["id"],
                    created_at=existing_reply["created_at"],
                )
            agent = _get_conversation_agent(
                conversation_id,
                user["user_id"],
                before_ordinal=user_message["ordinal"],
                force_reload=True,
            )

        audit_before = len(agent.get_audit_log())
        response = agent.run(req.message)
        new_logs = agent.get_audit_log()[audit_before:]
        tools_used = [log["tool"] for log in new_logs]
        assistant_message = conversation_repository.append_assistant_message(
            conversation_id,
            user["user_id"],
            response,
            tools_used,
            user_message["id"],
        )
        if agent.last_artifact is not None:
            conversation_repository.save_last_artifact(
                conversation_id, user["user_id"], agent.last_artifact
            )
        return ChatResponse(
            response=response,
            response_html=render_chat_markdown(response),
            tools_used=tools_used,
            conversation_id=conversation_id,
            title=conversation["title"],
            message_id=assistant_message["id"],
            created_at=assistant_message["created_at"],
        )
    except ConversationNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:
        if "conversation_id" in locals():
            sessions.pop(f"conversation:{conversation_id}", None)
        traceback.print_exc()
        return JSONResponse(
            status_code=500,
            content={
                "response": f"Error: {exc}",
                "response_html": "",
                "tools_used": [],
            },
        )
    finally:
        lock.release()


@app.get("/api/conversations", response_model=ConversationListResponse)
def list_conversations(
    limit: int = Query(default=30, ge=1, le=100),
    cursor: str | None = None,
):
    """List the authenticated user's conversations, newest first."""
    try:
        user = _current_user()
        items, next_cursor = conversation_repository.list_conversations(
            user["user_id"], limit=limit, cursor=cursor
        )
        return ConversationListResponse(
            conversations=items, next_cursor=next_cursor
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get(
    "/api/conversations/{conversation_id}/messages",
    response_model=MessageListResponse,
)
def list_conversation_messages(
    conversation_id: str,
    limit: int = Query(default=50, ge=1, le=200),
    before: int | None = Query(default=None, ge=1),
):
    """Return display-safe message history for one owned conversation."""
    try:
        user = _current_user()
        messages, next_before = conversation_repository.list_messages(
            conversation_id,
            user["user_id"],
            limit=limit,
            before=before,
        )
        return MessageListResponse(
            messages=[
                StoredMessage(
                    **item,
                    response_html=(
                        render_chat_markdown(item["content"])
                        if item["role"] == "assistant"
                        else ""
                    ),
                )
                for item in messages
            ],
            next_before=next_before,
        )
    except ConversationNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.post("/api/clear")
def clear_session(req: ClearRequest):
    """Evict legacy transient state without deleting durable conversations."""
    session_id = req.session_id or "default"
    sessions.pop(f"legacy:{session_id}", None)
    if req.conversation_id:
        sessions.pop(f"conversation:{req.conversation_id}", None)
    return {
        "status": "cache_cleared",
        "session_id": session_id,
        "conversation_id": req.conversation_id,
    }


@app.get("/api/audit")
def get_audit(session_id: str = "default"):
    """Return audit entries for one session."""
    agent = sessions.get(f"conversation:{session_id}") or get_agent(session_id)
    return {"audit_log": agent.get_audit_log()}


@app.get("/api/memories", response_model=MemoryFactsResponse)
def get_memory_facts(session_id: str = "default", limit: int = 100):
    """Return durable facts and preferences for the authenticated session user."""
    try:
        safe_limit = min(max(limit, 1), 500)
        agent = get_agent(session_id)
        user = check_authentication(agent.service_api_key)
        if "memory:read" not in user.get("scopes", []):
            raise PermissionError("Missing required scope: memory:read")
        memories = list_all_memories(
            limit=safe_limit,
            user_id=user["user_id"],
            categories={"fact", "user_preference"},
        )

        unique = {}
        for memory in memories:
            metadata = memory.get("metadata", {})
            category = metadata.get("category", "general")
            if category not in {"fact", "user_preference"}:
                continue
            text = memory.get("text", "").strip()
            if not text:
                continue
            key = (category, " ".join(text.casefold().split()))
            candidate = MemoryFact(
                id=str(memory.get("id", "")),
                text=text,
                category=category,
                created_at=metadata.get("created_at", ""),
                topic=metadata.get("topic", ""),
                value=metadata.get("value", ""),
                polarity=metadata.get("polarity", ""),
                confidence=metadata.get("confidence", 0.0),
            )
            previous = unique.get(key)
            if previous is None or candidate.created_at > previous.created_at:
                unique[key] = candidate

        facts = sorted(
            unique.values(),
            key=lambda item: item.created_at,
            reverse=True,
        )
        return MemoryFactsResponse(total=len(facts), facts=facts)
    except Exception as exc:
        traceback.print_exc()
        return JSONResponse(
            status_code=500,
            content={"detail": f"Could not load memory facts: {exc}"},
        )


@app.get("/api/documents", response_model=MemoryDocumentsResponse)
def get_memory_documents(session_id: str = "default", limit: int = 500):
    """Return saved document sources grouped across their RAG chunks."""
    try:
        safe_limit = min(max(limit, 1), 2_000)
        agent = get_agent(session_id)
        user = check_authentication(agent.service_api_key)
        if "memory:read" not in user.get("scopes", []):
            raise PermissionError("Missing required scope: memory:read")
        memories = list_all_memories(
            limit=safe_limit,
            user_id=user["user_id"],
            categories={"document", "note", "task"},
        )

        grouped: dict[str, list[dict]] = {}
        for memory in memories:
            metadata = memory.get("metadata", {})
            group_key = (
                metadata.get("source_id")
                or metadata.get("content_hash")
                or str(memory.get("id", ""))
            )
            grouped.setdefault(str(group_key), []).append(memory)

        documents = []
        for source_id, chunks in grouped.items():
            chunks.sort(key=lambda item: item.get("metadata", {}).get("chunk_index", 0))
            metadata = chunks[0].get("metadata", {})
            documents.append(
                MemoryDocument(
                    source_id=source_id,
                    file_id=metadata.get("file_id", ""),
                    file_name=metadata.get("file_name", "") or "Saved document",
                    category=metadata.get("category", "document"),
                    created_at=metadata.get("created_at", ""),
                    content_hash=metadata.get("content_hash", ""),
                    chunk_count=metadata.get("chunk_count", len(chunks)),
                    stored_chunks=len(chunks),
                    preview=chunks[0].get("text", "")[:240],
                )
            )
        documents.sort(key=lambda item: item.created_at, reverse=True)
        return MemoryDocumentsResponse(total=len(documents), documents=documents)
    except Exception as exc:
        traceback.print_exc()
        return JSONResponse(
            status_code=500,
            content={"detail": f"Could not load memory documents: {exc}"},
        )


@app.get("/api/health")
def health():
    """Report database readiness without exposing connection details."""
    database_ok = conversation_repository.healthcheck()
    return JSONResponse(
        status_code=200 if database_ok else 503,
        content={
            "status": "ok" if database_ok else "degraded",
            "database": "ok" if database_ok else "unavailable",
            "port": 9004,
        },
    )


# Web UI

@app.get("/", response_class=HTMLResponse)
def serve_ui():
    """Serve the bundled chat interface."""
    with open("static/index.html", "r", encoding="utf-8") as f:
        return f.read()


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=9004)
