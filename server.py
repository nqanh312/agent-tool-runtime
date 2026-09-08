"""Expose the authenticated agent API and bundled web UI."""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
import ipaddress
import logging
import threading
import time
import traceback
import uuid

import uvicorn
from fastapi import Depends, FastAPI, HTTPException, Query, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
from pydantic import BaseModel, Field

from agent import Agent
from config import (
    AGENT_SESSION_CACHE_MAX,
    AGENT_SESSION_TTL_SECONDS,
    AUTH_COOKIE_SECURE,
    CHAT_MESSAGE_MAX_CHARS,
    CHAT_RATE_LIMIT_REQUESTS,
    CHAT_RATE_LIMIT_WINDOW_SECONDS,
    CHAT_TOKEN_BASE_CHARGE,
    CHAT_TOKEN_QUOTA_PER_DAY,
    CONVERSATION_LOCK_CACHE_MAX,
    CONVERSATION_LOCK_TTL_SECONDS,
    CORS_ORIGINS,
    GOOGLE_OAUTH_STATE_MINUTES,
    JWT_REFRESH_DAYS,
    MAX_REQUEST_BODY_BYTES,
    SERVER_HOST,
    SERVER_PORT,
    TRUSTED_PROXY_IPS,
)
from services.auth import (
    AuthenticationError,
    AuthConfigurationError,
    DuplicateUsernameError,
    LastAdminError,
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
from services.request_limits import ChatLimitExceeded, ChatUsageLimiter
from services.vectorstore import list_all_memories


logger = logging.getLogger(__name__)


class RequestBodyLimitMiddleware:
    """Reject oversized HTTP bodies, including chunked requests."""

    def __init__(self, app, max_body_bytes: int):
        self.app = app
        self.max_body_bytes = max(1, max_body_bytes)

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        headers = dict(scope.get("headers") or [])
        raw_length = headers.get(b"content-length")
        if raw_length is not None:
            try:
                if int(raw_length) > self.max_body_bytes:
                    await JSONResponse(
                        status_code=413,
                        content={"detail": "Request body is too large"},
                    )(scope, receive, send)
                    return
            except ValueError:
                await JSONResponse(
                    status_code=400,
                    content={"detail": "Invalid Content-Length header"},
                )(scope, receive, send)
                return

        received = 0

        async def limited_receive():
            nonlocal received
            message = await receive()
            if message.get("type") == "http.request":
                received += len(message.get("body", b""))
                if received > self.max_body_bytes:
                    raise _RequestBodyTooLarge
            return message

        try:
            await self.app(scope, limited_receive, send)
        except _RequestBodyTooLarge:
            await JSONResponse(
                status_code=413,
                content={"detail": "Request body is too large"},
            )(scope, receive, send)


class _RequestBodyTooLarge(Exception):
    pass


# HTTP middleware and process-wide state are initialized once at import time.
app = FastAPI(title="AI Agent - Assignment 1")
app.add_middleware(
    RequestBodyLimitMiddleware,
    max_body_bytes=MAX_REQUEST_BODY_BYTES,
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PATCH", "OPTIONS"],
    allow_headers=["Authorization", "Content-Type"],
)

REFRESH_COOKIE = "agent_refresh"
GOOGLE_OAUTH_BINDING_COOKIE = "google_oauth_binding"
# Agents are expensive, stateful objects. Keep an LRU-like cache per user and
# conversation; every access to these two mappings must hold sessions_guard.
sessions: dict[str, Agent] = {}
session_access: OrderedDict[str, float] = OrderedDict()
sessions_guard = threading.Lock()
# A separate non-blocking lock prevents two turns from assigning duplicate
# ordinals or mutating the same cached Agent concurrently.
conversation_locks: dict[str, threading.Lock] = {}
conversation_locks_guard = threading.Lock()
chat_usage_limiter = ChatUsageLimiter(
    max_requests=CHAT_RATE_LIMIT_REQUESTS,
    window_seconds=CHAT_RATE_LIMIT_WINDOW_SECONDS,
    daily_token_quota=CHAT_TOKEN_QUOTA_PER_DAY,
    base_token_charge=CHAT_TOKEN_BASE_CHARGE,
)


@dataclass
class _ConversationLockState:
    references: int = 0
    last_used: float = 0.0


conversation_lock_states: dict[str, _ConversationLockState] = {}


def _client_ip(request: Request) -> str:
    """Use a forwarded address only when the direct peer is a trusted proxy."""
    peer = request.client.host if request.client else ""
    if peer not in TRUSTED_PROXY_IPS:
        return peer
    forwarded = request.headers.get("X-Forwarded-For", "").split(",", 1)[0].strip()
    try:
        return str(ipaddress.ip_address(forwarded))
    except ValueError:
        return peer


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
    """Build a FastAPI dependency that requires every named permission."""
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


def _prune_conversation_locks(now: float) -> None:
    """Discard only inactive locks, first by TTL and then by cache capacity."""
    expired = [
        key
        for key, state in conversation_lock_states.items()
        if state.references == 0
        and now - state.last_used >= CONVERSATION_LOCK_TTL_SECONDS
    ]
    for key in expired:
        conversation_locks.pop(key, None)
        conversation_lock_states.pop(key, None)

    overflow = len(conversation_locks) - max(1, CONVERSATION_LOCK_CACHE_MAX)
    if overflow <= 0:
        return
    inactive = sorted(
        (
            (state.last_used, key)
            for key, state in conversation_lock_states.items()
            if state.references == 0
        )
    )
    for _, key in inactive[:overflow]:
        conversation_locks.pop(key, None)
        conversation_lock_states.pop(key, None)


def _try_acquire_conversation_lock(key: str) -> threading.Lock | None:
    """Acquire one conversation slot without making a concurrent request wait."""
    now = time.monotonic()
    with conversation_locks_guard:
        _prune_conversation_locks(now)
        lock = conversation_locks.setdefault(key, threading.Lock())
        state = conversation_lock_states.setdefault(
            key, _ConversationLockState(last_used=now)
        )
        state.references += 1
        state.last_used = now
        if lock.acquire(blocking=False):
            return lock
        state.references -= 1
        return None


def _release_conversation_lock(key: str, lock: threading.Lock) -> None:
    """Release a conversation slot and make the entry eligible for pruning."""
    lock.release()
    now = time.monotonic()
    with conversation_locks_guard:
        state = conversation_lock_states.get(key)
        if state is not None:
            state.references = max(0, state.references - 1)
            state.last_used = now
        _prune_conversation_locks(now)


def _agent_key(user_id: str, conversation_id: str) -> str:
    return f"{user_id}:conversation:{conversation_id}"


def _prune_sessions(now: float, protected_key: str | None = None) -> None:
    """Expire idle agents and enforce the cache cap without evicting a live key."""
    for key in list(session_access):
        if key not in sessions:
            session_access.pop(key, None)
    for key in sessions:
        session_access.setdefault(key, now)

    for key, accessed_at in list(session_access.items()):
        if key != protected_key and now - accessed_at >= AGENT_SESSION_TTL_SECONDS:
            session_access.pop(key, None)
            sessions.pop(key, None)

    max_entries = max(1, AGENT_SESSION_CACHE_MAX)
    while len(sessions) > max_entries:
        oldest_key, accessed_at = session_access.popitem(last=False)
        if oldest_key == protected_key:
            session_access[oldest_key] = accessed_at
            continue
        sessions.pop(oldest_key, None)


def _drop_session(key: str) -> None:
    with sessions_guard:
        sessions.pop(key, None)
        session_access.pop(key, None)


def _drop_user_sessions(user_id: str) -> None:
    prefix = f"{user_id}:"
    with sessions_guard:
        for key in [key for key in sessions if key.startswith(prefix)]:
            sessions.pop(key, None)
            session_access.pop(key, None)


def _get_conversation_agent(
    conversation_id: str,
    user: dict,
    *,
    before_ordinal: int | None = None,
    force_reload: bool = False,
) -> Agent:
    """Load or rebuild the stateful Agent for an owned conversation."""
    key = _agent_key(user["user_id"], conversation_id)
    now = time.monotonic()
    with sessions_guard:
        _prune_sessions(now, protected_key=key)
        if force_reload:
            sessions.pop(key, None)
            session_access.pop(key, None)
        cached = sessions.get(key)
        if cached is not None:
            # Cached agents must never retain permissions from an older role/token.
            cached.principal = dict(user)
            session_access[key] = now
            session_access.move_to_end(key)
            return cached

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
    agent = Agent(
        principal=user,
        conversation_history=history,
        last_artifact=artifact,
        audit_sink=_audit_sink(f"conversation:{conversation_id}"),
    )
    with sessions_guard:
        sessions[key] = agent
        session_access[key] = time.monotonic()
        session_access.move_to_end(key)
        _prune_sessions(time.monotonic(), protected_key=key)
    return agent


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
    message: str = Field(min_length=1, max_length=CHAT_MESSAGE_MAX_CHARS)
    conversation_id: str | None = None
    client_message_id: str | None = None


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
    except (GoogleOAuthError, AuthenticationError, ValueError) as exc:
        # Do not log the callback URL because it contains a one-time OAuth code.
        logger.exception("Google OAuth callback failed: %s", exc)
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
        _drop_user_sessions(user_id)
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
    response: Response,
    user: dict = Depends(require_permissions("chat:use", "conversation:write")),
):
    """Run one idempotent, serialized turn and persist both sides of it."""
    try:
        allowance = chat_usage_limiter.check(user["user_id"], req.message)
    except ChatLimitExceeded as exc:
        raise HTTPException(
            status_code=429,
            detail=str(exc),
            headers={"Retry-After": str(exc.retry_after)},
        ) from exc
    response.headers["X-RateLimit-Remaining"] = str(
        allowance["remaining_requests"]
    )
    response.headers["X-TokenQuota-Remaining"] = str(
        allowance["remaining_tokens"]
    )

    client_message_id = req.client_message_id or str(uuid.uuid4())
    # New chats lock on the stable retry ID until a conversation ID exists;
    # existing chats lock directly on their conversation ID.
    lock_key = f"{user['user_id']}:{req.conversation_id or client_message_id}"
    lock = _try_acquire_conversation_lock(lock_key)
    if lock is None:
        raise HTTPException(status_code=409, detail="Another message is already running for this conversation")
    try:
        conversation = None
        user_message = None
        new_conversation = False
        # The client may retry after losing the HTTP response. Recover its
        # durable user message before creating any new state or calling the LLM.
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
                # Returning the stored reply makes a completed retry free of
                # duplicate model calls and duplicate assistant messages.
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

        # Diffing the audit log associates only this turn's tools with the
        # assistant message, even when the Agent came from the session cache.
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
            _drop_session(_agent_key(user["user_id"], conversation_id))
        traceback.print_exc()
        return JSONResponse(status_code=500, content={"response": "Internal server error", "response_html": "", "tools_used": []})
    finally:
        _release_conversation_lock(lock_key, lock)


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
        content={"status": "ok" if database_ok else "degraded", "database": "ok" if database_ok else "unavailable", "port": SERVER_PORT},
    )


@app.get("/", response_class=HTMLResponse)
def serve_ui():
    with open("static/index.html", "r", encoding="utf-8") as handle:
        return handle.read()


if __name__ == "__main__":
    uvicorn.run(app, host=SERVER_HOST, port=SERVER_PORT)
