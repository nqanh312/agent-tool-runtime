"""Expose the agent through a FastAPI service and a simple web UI."""

import json
import traceback
import uvicorn
from fastapi import FastAPI
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from agent import Agent
from registry.registry import check_authentication
from services.chat_renderer import render_chat_markdown
from services.vectorstore import list_all_memories

app = FastAPI(title="AI Agent - Assignment 1")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Keep conversation history isolated by client-provided session ID.
sessions: dict[str, Agent] = {}


def get_agent(session_id: str) -> Agent:
    """Return the existing session agent or create one on demand."""
    if session_id not in sessions:
        sessions[session_id] = Agent(service_api_key="sk-admin-001")
    return sessions[session_id]


# Request and response models

class ChatRequest(BaseModel):
    session_id: str = "default"
    message: str


class ChatResponse(BaseModel):
    response: str
    response_html: str = ""
    tools_used: list[str] = []


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
    try:
        agent = get_agent(req.session_id)
        audit_before = len(agent.get_audit_log())
        response = agent.run(req.message)

        new_logs = agent.get_audit_log()[audit_before:]
        tools_used = [log["tool"] for log in new_logs]

        return ChatResponse(
            response=response,
            response_html=render_chat_markdown(response),
            tools_used=tools_used,
        )
    except Exception as e:
        traceback.print_exc()
        return JSONResponse(
            status_code=500,
            content={
                "response": f"Error: {e}",
                "response_html": "",
                "tools_used": [],
            },
        )


@app.post("/api/clear")
def clear_session(req: ChatRequest):
    """Clear conversation history for one session."""
    session_id = req.session_id or "default"
    if session_id in sessions:
        sessions[session_id].clear_history()
    return {"status": "cleared", "session_id": session_id}


@app.get("/api/audit")
def get_audit(session_id: str = "default"):
    """Return audit entries for one session."""
    agent = get_agent(session_id)
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
    """Return a lightweight service health check."""
    return {"status": "ok", "port": 9004}


# Web UI

@app.get("/", response_class=HTMLResponse)
def serve_ui():
    """Serve the bundled chat interface."""
    with open("static/index.html", "r", encoding="utf-8") as f:
        return f.read()


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=9004)
