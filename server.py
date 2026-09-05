"""Expose the agent through a FastAPI service and a simple web UI."""

import json
import traceback
import uvicorn
from fastapi import FastAPI
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from agent import Agent
from services.chat_renderer import render_chat_markdown

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
