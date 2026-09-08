"""Expose the authenticated agent API and bundled web UI."""

from __future__ import annotations

import threading
import traceback
import uuid

import uvicorn
from fastapi import Depends, FastAPI, HTTPException, Query, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
from pydantic import BaseModel, Field

from agent import Agent
from config import (
    AUTH_COOKIE_SECURE,
    CORS_ORIGINS,
    GOOGLE_OAUTH_STATE_MINUTES,
    JWT_REFRESH_DAYS,
)
from services.auth import (
    AuthenticationError,
    AuthConfigurationError,
    DuplicateUsernameError,
    LastAdminError,
    ROLE_PERMISSIONS,
    auth_repository,
    auth_service,
    generate_temporary_password,
    login_rate_limiter,
    normalize_username,
)
from services.audit_logs import audit_log_repository
from services.chat_renderer import render_chat_markdown
from services.conversations import ConversationNotFoundError, conversation_repository
from services.drive_service import clear_drive_cache
from services.google_oauth import (
    GoogleOAuthConfigurationError,
    GoogleOAuthError,
    google_oauth_service,
)
from services.vectorstore import list_all_memories


app = FastAPI(title="AI Agent - Assignment 1")
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PATCH", "OPTIONS"],
    allow_headers=["Authorization", "Content-Type"],
)

REFRESH_COOKIE = "agent_refresh"
GOOGLE_OAUTH_BINDING_COOKIE = "google_oauth_binding"
sessions: dict[str, Agent] = {}
conversation_locks: dict[str, threading.Lock] = {}
conversation_locks_guard = threading.Lock()


def _client_ip(request: Request) -> str:
    return request.client.host if request.client else ""


def _set_refresh_cookie(response: Response, token: str) -> None:
    response.set_cookie(
        REFRESH_COOKIE,
        token,
        max_age=JWT_REFRESH_DAYS * 86400,
        httponly=True,
        secure=AUTH_COOKIE_SECURE,
        samesite="strict",
        path="/api/auth",
    )


def _clear_refresh_cookie(response: Response) -> None:
    response.delete_cookie(REFRESH_COOKIE, path="/api/auth", samesite="strict")


def _set_google_oauth_binding(response: Response, value: str) -> None:
    response.set_cookie(
        GOOGLE_OAUTH_BINDING_COOKIE,
        value,
        max_age=GOOGLE_OAUTH_STATE_MINUTES * 60,
        httponly=True,
        secure=AUTH_COOKIE_SECURE,
        samesite="lax",
        path="/api/auth/google/callback",
    )


def _clear_google_oauth_binding(response: Response) -> None:
    response.delete_cookie(
        GOOGLE_OAUTH_BINDING_COOKIE,
        path="/api/auth/google/callback",
        samesite="lax",
    )


def _bearer_token(request: Request) -> str:
    header = request.headers.get("Authorization", "")
    scheme, _, token = header.partition(" ")
    if scheme.casefold() != "bearer" or not token:
        raise HTTPException(
            status_code=401,
            detail="Authentication required",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return token


def current_principal(request: Request) -> dict:
    try:
        return auth_service.principal_from_access(_bearer_token(request))
    except HTTPException:
        raise
    except AuthConfigurationError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except AuthenticationError as exc:
        raise HTTPException(
            status_code=401,
            detail=str(exc),
            headers={"WWW-Authenticate": "Bearer"},
        ) from exc


def require_permissions(*required: str):
    def dependency(user: dict = Depends(current_principal)) -> dict:
        missing = sorted(set(required) - set(user.get("permissions", [])))
        if missing:
            raise HTTPException(
                status_code=403,
                detail=f"Missing required permissions: {', '.join(missing)}",
            )
        return user
    return dependency


def _audit_sink(context_id: str):
    return lambda entry: audit_log_repository.append(context_id, entry)


def _conversation_lock(key: str) -> threading.Lock:
    with conversation_locks_guard:
        return conversation_locks.setdefault(key, threading.Lock())


def _agent_key(user_id: str, conversation_id: str) -> str:
    return f"{user_id}:conversation:{conversation_id}"


def _get_conversation_agent(
    conversation_id: str,
    user: dict,
    *,
    before_ordinal: int | None = None,
    force_reload: bool = False,
) -> Agent:
    key = _agent_key(user["user_id"], conversation_id)
    if force_reload:
        sessions.pop(key, None)
    if key not in sessions:
        stored = conversation_repository.list_context_messages(
            conversation_id, user["user_id"], before_ordinal=before_ordinal
        )
        history = [
            {"role": item["role"], "content": item["content"]}
            for item in stored if item["role"] in {"user", "assistant"}
        ]
        artifact = conversation_repository.get_last_artifact(
            conversation_id, user["user_id"]
        )
        sessions[key] = Agent(
            principal=user,
            conversation_history=history,
            last_artifact=artifact,
            audit_sink=_audit_sink(f"conversation:{conversation_id}"),
        )
    else:
        # Cached agents must never retain permissions from an older role/token.
        sessions[key].principal = dict(user)
    return sessions[key]


class LoginRequest(BaseModel):
    username: str
    password: str


class CompletePasswordChangeRequest(BaseModel):
    change_token: str
    new_password: str


class ChangePasswordRequest(BaseModel):
    current_password: str
    new_password: str


class AdminCreateUserRequest(BaseModel):
    username: str
    display_name: str = ""
    role: str = "user"


class AdminUpdateUserRequest(BaseModel):
    role: str | None = None
    is_active: bool | None = None


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=100_000)
    conversation_id: str | None = None
    client_message_id: str | None = None


class ClearRequest(BaseModel):
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


def _public_user(user: dict) -> dict:
    return {
        key: user[key]
        for key in (
            "user_id", "username", "display_name", "role", "is_active",
            "must_change_password", "has_password", "permissions",
            "created_at", "updated_at",
        )
        if key in user
    }


def _session_response(issued) -> dict:
    return {
        "access_token": issued.access_token,
        "token_type": "bearer",
        "expires_in": issued.expires_in,
        "password_change_required": False,
        "user": _public_user(issued.user),
    }


@app.get("/api/auth/google/start")
def google_login_start(request: Request):
    """Start Google OIDC login without requesting Drive access."""
    try:
        login_rate_limiter.check(f"google:{_client_ip(request)}")
        authorization = google_oauth_service.begin(mode="login")
        response = RedirectResponse(
            authorization["authorization_url"], status_code=302
        )
        _set_google_oauth_binding(response, authorization["browser_binding"])
        return response
    except GoogleOAuthConfigurationError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except AuthenticationError as exc:
        raise HTTPException(status_code=429, detail=str(exc)) from exc


@app.get("/api/auth/google/callback")
def google_oauth_callback(
    request: Request,
    state: str = "",
    code: str = "",
    error: str | None = None,
):
    """Consume one OAuth transaction and return to the SPA without URL tokens."""
    destination = "/?oauth=connected"
    try:
        if error or not state or not code:
            raise GoogleOAuthError("Google authorization was cancelled")
        result = google_oauth_service.complete(
            raw_state=state,
            code=code,
            browser_binding=request.cookies.get(GOOGLE_OAUTH_BINDING_COOKIE, ""),
        )
        user = result["user"]
        if result["mode"] == "login":
            issued = auth_service.issue_session(
                user,
                ip_address=_client_ip(request),
                user_agent=request.headers.get("User-Agent", ""),
            )
            response = RedirectResponse(destination, status_code=303)
            _set_refresh_cookie(response, issued.refresh_token)
            auth_repository.log_event(
                "google_login",
                "success",
                actor_user_id=user["user_id"],
                ip_address=_client_ip(request),
            )
        else:
            response = RedirectResponse("/?drive=connected", status_code=303)
            auth_repository.log_event(
                "google_drive_connect",
                "success",
                actor_user_id=user["user_id"],
                ip_address=_client_ip(request),
            )
    except (GoogleOAuthError, AuthenticationError, ValueError):
        response = RedirectResponse("/?oauth=error", status_code=303)
    _clear_google_oauth_binding(response)
    return response


@app.post("/api/integrations/google-drive/authorize")
def authorize_google_drive(
    response: Response,
    user: dict = Depends(require_permissions("drive:read")),
):
    """Create a user-bound Drive consent transaction."""
    try:
        authorization = google_oauth_service.begin(
            mode="drive", user_id=user["user_id"]
        )
        _set_google_oauth_binding(response, authorization["browser_binding"])
        return {"authorization_url": authorization["authorization_url"]}
    except GoogleOAuthConfigurationError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@app.get("/api/integrations/google-drive/status")
def google_drive_status(
    user: dict = Depends(require_permissions("drive:read")),
):
    return google_oauth_service.status(user["user_id"])


@app.post("/api/integrations/google-drive/disconnect")
def disconnect_google_drive(
    request: Request,
    user: dict = Depends(require_permissions("drive:read")),
):
    disconnected = google_oauth_service.disconnect(user["user_id"])
    clear_drive_cache(user["user_id"])
    if disconnected:
        auth_repository.log_event(
            "google_drive_disconnect",
            "success",
            actor_user_id=user["user_id"],
            ip_address=_client_ip(request),
        )
    return {"connected": False}


@app.post("/api/auth/login")
def login(body: LoginRequest, request: Request, response: Response):
    ip = _client_ip(request)
    try:
        try:
            normalized = normalize_username(body.username)
        except ValueError:
            normalized = "invalid"
        login_rate_limiter.check(f"{ip}:{normalized}")
        user = auth_service.authenticate(body.username, body.password, ip_address=ip)
        if user["must_change_password"]:
            return {
                "password_change_required": True,
                "change_token": auth_service.issue_password_change_token(user),
                "user": _public_user(user),
            }
        issued = auth_service.issue_session(
            user, ip_address=ip, user_agent=request.headers.get("User-Agent", "")
        )
        _set_refresh_cookie(response, issued.refresh_token)
        return _session_response(issued)
    except AuthConfigurationError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except AuthenticationError as exc:
        status = 429 if "Too many" in str(exc) else 401
        raise HTTPException(status_code=status, detail="Invalid username or password") from exc


@app.post("/api/auth/complete-password-change")
def complete_password_change(
    body: CompletePasswordChangeRequest, request: Request, response: Response
):
    try:
        user = auth_service.complete_password_change(body.change_token, body.new_password)
        issued = auth_service.issue_session(
            user, ip_address=_client_ip(request),
            user_agent=request.headers.get("User-Agent", ""),
        )
        auth_repository.log_event(
            "complete_password_change", "success", actor_user_id=user["user_id"],
            target_user_id=user["user_id"], ip_address=_client_ip(request),
        )
        _set_refresh_cookie(response, issued.refresh_token)
        return _session_response(issued)
    except (AuthenticationError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except AuthConfigurationError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@app.post("/api/auth/refresh")
def refresh(request: Request, response: Response):
    token = request.cookies.get(REFRESH_COOKIE)
    if not token:
        raise HTTPException(status_code=401, detail="Refresh token required")
    try:
        issued = auth_service.refresh(
            token, ip_address=_client_ip(request),
            user_agent=request.headers.get("User-Agent", ""),
        )
        _set_refresh_cookie(response, issued.refresh_token)
        return _session_response(issued)
    except AuthConfigurationError as exc:
        _clear_refresh_cookie(response)
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except AuthenticationError as exc:
        _clear_refresh_cookie(response)
        raise HTTPException(status_code=401, detail=str(exc)) from exc


@app.post("/api/auth/logout")
def logout(request: Request, response: Response):
    auth_service.logout(
        request.cookies.get(REFRESH_COOKIE), ip_address=_client_ip(request)
    )
    _clear_refresh_cookie(response)
    return {"status": "logged_out"}


@app.get("/api/auth/me")
def me(user: dict = Depends(current_principal)):
    return _public_user(user)


@app.post("/api/auth/change-password")
def change_password(
    body: ChangePasswordRequest, request: Request, response: Response,
    user: dict = Depends(current_principal),
):
    try:
        updated = auth_repository.set_password(
            user["user_id"], body.new_password, must_change=False,
            expected_current=body.current_password,
        )
        issued = auth_service.issue_session(
            updated, ip_address=_client_ip(request),
            user_agent=request.headers.get("User-Agent", ""),
        )
        auth_repository.log_event(
            "change_password", "success", actor_user_id=user["user_id"],
            target_user_id=user["user_id"], ip_address=_client_ip(request),
        )
        _set_refresh_cookie(response, issued.refresh_token)
        return _session_response(issued)
    except (AuthenticationError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/api/admin/users")
def list_users(
    limit: int = Query(50, ge=1, le=100), cursor: int = Query(0, ge=0),
    role: str | None = None, is_active: bool | None = None,
    _user: dict = Depends(require_permissions("users:manage")),
):
    try:
        users, total = auth_repository.list_users(
            offset=cursor, limit=limit, role=role, is_active=is_active
        )
        next_cursor = cursor + len(users) if cursor + len(users) < total else None
        return {"users": [_public_user(item) for item in users], "total": total, "next_cursor": next_cursor}
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/api/admin/users", status_code=201)
def create_user(
    body: AdminCreateUserRequest, request: Request,
    actor: dict = Depends(require_permissions("users:manage")),
):
    password = generate_temporary_password()
    try:
        user = auth_repository.create_user(
            body.username, body.display_name, body.role, password
        )
        auth_repository.log_event(
            "create_user", "success", actor_user_id=actor["user_id"],
            target_user_id=user["user_id"], detail=f"role={body.role}",
            ip_address=_client_ip(request),
        )
        return {"user": _public_user(user), "temporary_password": password}
    except DuplicateUsernameError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.patch("/api/admin/users/{user_id}")
def update_user(
    user_id: str, body: AdminUpdateUserRequest, request: Request,
    actor: dict = Depends(require_permissions("users:manage")),
):
    if body.role is None and body.is_active is None:
        raise HTTPException(status_code=400, detail="No changes supplied")
    try:
        updated = auth_repository.update_user(
            user_id, role=body.role, is_active=body.is_active
        )
        auth_repository.log_event(
            "update_user", "success", actor_user_id=actor["user_id"],
            target_user_id=user_id,
            detail=f"role={body.role};is_active={body.is_active}",
            ip_address=_client_ip(request),
        )
        sessions_to_remove = [key for key in sessions if key.startswith(f"{user_id}:")]
        for key in sessions_to_remove:
            sessions.pop(key, None)
        return {"user": _public_user(updated)}
    except LookupError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except LastAdminError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/api/admin/users/{user_id}/reset-password")
def reset_password(
    user_id: str, request: Request,
    actor: dict = Depends(require_permissions("users:manage")),
):
    if user_id == actor["user_id"]:
        raise HTTPException(status_code=400, detail="Use the self-service password change endpoint")
    password = generate_temporary_password()
    try:
        updated = auth_repository.set_password(user_id, password, must_change=True)
        auth_repository.log_event(
            "reset_password", "success", actor_user_id=actor["user_id"],
            target_user_id=user_id, ip_address=_client_ip(request),
        )
        return {"user": _public_user(updated), "temporary_password": password}
    except LookupError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.post("/api/chat", response_model=ChatResponse)
def chat(
    req: ChatRequest,
    user: dict = Depends(require_permissions("chat:use", "conversation:write")),
):
    client_message_id = req.client_message_id or str(uuid.uuid4())
    lock_key = f"{user['user_id']}:{req.conversation_id or client_message_id}"
    lock = _conversation_lock(lock_key)
    if not lock.acquire(blocking=False):
        raise HTTPException(status_code=409, detail="Another message is already running for this conversation")
    try:
        conversation = None
        user_message = None
        new_conversation = False
        retried = conversation_repository.find_user_message(user["user_id"], client_message_id)
        if retried is not None:
            user_message, conversation = retried
        elif req.conversation_id:
            conversation = conversation_repository.get_conversation(req.conversation_id, user["user_id"])
        else:
            conversation, user_message = conversation_repository.create_conversation_with_message(
                user["user_id"], req.message, client_message_id
            )
            new_conversation = True

        conversation_id = conversation["id"]
        if new_conversation:
            agent = _get_conversation_agent(
                conversation_id, user, before_ordinal=user_message["ordinal"]
            )
        elif user_message is None:
            agent = _get_conversation_agent(conversation_id, user)
            user_message, _ = conversation_repository.append_user_message(
                conversation_id, user["user_id"], req.message, client_message_id
            )
        else:
            existing_reply = conversation_repository.get_reply(user_message["id"])
            if existing_reply is not None:
                return ChatResponse(
                    response=existing_reply["content"],
                    response_html=render_chat_markdown(existing_reply["content"]),
                    tools_used=existing_reply["tools_used"], conversation_id=conversation_id,
                    title=conversation["title"], message_id=existing_reply["id"],
                    created_at=existing_reply["created_at"],
                )
            agent = _get_conversation_agent(
                conversation_id, user, before_ordinal=user_message["ordinal"], force_reload=True
            )

        audit_before = len(agent.get_audit_log())
        answer = agent.run(req.message)
        tools_used = [entry["tool"] for entry in agent.get_audit_log()[audit_before:]]
        assistant_message = conversation_repository.append_assistant_message(
            conversation_id, user["user_id"], answer, tools_used, user_message["id"]
        )
        if agent.last_artifact is not None:
            conversation_repository.save_last_artifact(
                conversation_id, user["user_id"], agent.last_artifact
            )
        return ChatResponse(
            response=answer, response_html=render_chat_markdown(answer),
            tools_used=tools_used, conversation_id=conversation_id,
            title=conversation["title"], message_id=assistant_message["id"],
            created_at=assistant_message["created_at"],
        )
    except ConversationNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:
        if "conversation_id" in locals():
            sessions.pop(_agent_key(user["user_id"], conversation_id), None)
        traceback.print_exc()
        return JSONResponse(status_code=500, content={"response": "Internal server error", "response_html": "", "tools_used": []})
    finally:
        lock.release()


@app.get("/api/conversations", response_model=ConversationListResponse)
def list_conversations(
    limit: int = Query(30, ge=1, le=100), cursor: str | None = None,
    user: dict = Depends(require_permissions("conversation:read")),
):
    try:
        items, next_cursor = conversation_repository.list_conversations(
            user["user_id"], limit=limit, cursor=cursor
        )
        return ConversationListResponse(conversations=items, next_cursor=next_cursor)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/api/conversations/{conversation_id}/messages", response_model=MessageListResponse)
def list_conversation_messages(
    conversation_id: str, limit: int = Query(50, ge=1, le=200),
    before: int | None = Query(None, ge=1),
    user: dict = Depends(require_permissions("conversation:read")),
):
    try:
        messages, next_before = conversation_repository.list_messages(
            conversation_id, user["user_id"], limit=limit, before=before
        )
        return MessageListResponse(
            messages=[StoredMessage(
                **item,
                response_html=render_chat_markdown(item["content"]) if item["role"] == "assistant" else "",
            ) for item in messages], next_before=next_before,
        )
    except ConversationNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.post("/api/clear")
def clear_session(
    req: ClearRequest,
    user: dict = Depends(require_permissions("conversation:write")),
):
    if req.conversation_id:
        try:
            conversation_repository.get_conversation(req.conversation_id, user["user_id"])
        except ConversationNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        sessions.pop(_agent_key(user["user_id"], req.conversation_id), None)
    return {"status": "cache_cleared", "conversation_id": req.conversation_id}


@app.get("/api/audit")
def get_audit(
    session_id: str = "default",
    user: dict = Depends(require_permissions("audit:read")),
):
    if session_id == "default":
        return {"audit_log": []}
    try:
        uuid.UUID(session_id)
        conversation_repository.get_conversation(session_id, user["user_id"])
    except (ValueError, ConversationNotFoundError) as exc:
        raise HTTPException(status_code=404, detail="Conversation not found") from exc
    return {"audit_log": audit_log_repository.list_entries(
        f"conversation:{session_id}", user["user_id"]
    )}


@app.get("/api/memories", response_model=MemoryFactsResponse)
def get_memory_facts(
    limit: int = 100,
    user: dict = Depends(require_permissions("memory:read")),
):
    try:
        memories = list_all_memories(
            limit=min(max(limit, 1), 500), user_id=user["user_id"],
            categories={"fact", "user_preference"},
        )
        unique = {}
        for memory in memories:
            metadata = memory.get("metadata", {})
            category = metadata.get("category", "general")
            text = memory.get("text", "").strip()
            if category not in {"fact", "user_preference"} or not text:
                continue
            key = (category, " ".join(text.casefold().split()))
            candidate = MemoryFact(
                id=str(memory.get("id", "")), text=text, category=category,
                created_at=metadata.get("created_at", ""), topic=metadata.get("topic", ""),
                value=metadata.get("value", ""), polarity=metadata.get("polarity", ""),
                confidence=metadata.get("confidence", 0.0),
            )
            if key not in unique or candidate.created_at > unique[key].created_at:
                unique[key] = candidate
        facts = sorted(unique.values(), key=lambda item: item.created_at, reverse=True)
        return MemoryFactsResponse(total=len(facts), facts=facts)
    except Exception as exc:
        traceback.print_exc()
        return JSONResponse(status_code=500, content={"detail": "Could not load memory facts"})


@app.get("/api/documents", response_model=MemoryDocumentsResponse)
def get_memory_documents(
    limit: int = 500,
    user: dict = Depends(require_permissions("memory:read")),
):
    try:
        memories = list_all_memories(
            limit=min(max(limit, 1), 2000), user_id=user["user_id"],
            categories={"document", "note", "task"},
        )
        grouped: dict[str, list[dict]] = {}
        for memory in memories:
            metadata = memory.get("metadata", {})
            key = metadata.get("source_id") or metadata.get("content_hash") or str(memory.get("id", ""))
            grouped.setdefault(str(key), []).append(memory)
        documents = []
        for source_id, chunks in grouped.items():
            chunks.sort(key=lambda item: item.get("metadata", {}).get("chunk_index", 0))
            metadata = chunks[0].get("metadata", {})
            documents.append(MemoryDocument(
                source_id=source_id, file_id=metadata.get("file_id", ""),
                file_name=metadata.get("file_name", "") or "Saved document",
                category=metadata.get("category", "document"),
                created_at=metadata.get("created_at", ""),
                content_hash=metadata.get("content_hash", ""),
                chunk_count=metadata.get("chunk_count", len(chunks)),
                stored_chunks=len(chunks), preview=chunks[0].get("text", "")[:240],
            ))
        documents.sort(key=lambda item: item.created_at, reverse=True)
        return MemoryDocumentsResponse(total=len(documents), documents=documents)
    except Exception as exc:
        traceback.print_exc()
        return JSONResponse(status_code=500, content={"detail": "Could not load memory documents"})


@app.get("/api/health")
def health():
    database_ok = conversation_repository.healthcheck()
    return JSONResponse(
        status_code=200 if database_ok else 503,
        content={"status": "ok" if database_ok else "degraded", "database": "ok" if database_ok else "unavailable", "port": 9004},
    )


@app.get("/", response_class=HTMLResponse)
def serve_ui():
    with open("static/index.html", "r", encoding="utf-8") as handle:
        return handle.read()


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=9004)
