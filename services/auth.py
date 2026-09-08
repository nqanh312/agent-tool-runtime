"""PostgreSQL-backed authentication, JWT sessions, and fixed-role RBAC."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
import hashlib
import re
import secrets
import threading
import time
import uuid
from typing import Callable

from argon2 import PasswordHasher
from argon2.exceptions import InvalidHashError, VerifyMismatchError
import jwt
from sqlalchemy import (
    Boolean,
    DateTime,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
    delete,
    func,
    select,
)
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column

from config import (
    JWT_ACCESS_MINUTES,
    JWT_PASSWORD_CHANGE_MINUTES,
    JWT_REFRESH_DAYS,
    JWT_SECRET,
)
from services.conversations import SessionLocal


UTC = timezone.utc
USERNAME_RE = re.compile(r"^[a-z0-9._-]{3,64}$")
ROLE_PERMISSIONS: dict[str, tuple[str, ...]] = {
    "admin": (
        "chat:use", "conversation:read", "conversation:write",
        "audit:read", "drive:read", "memory:read", "memory:write",
        "local_file:read", "users:manage",
    ),
    "user": (
        "chat:use", "conversation:read", "conversation:write",
        "audit:read", "drive:read", "memory:read", "memory:write",
    ),
    "guest": (
        "chat:use", "conversation:read", "conversation:write",
        "audit:read", "drive:read", "memory:read",
    ),
}
ALL_PERMISSIONS = tuple(sorted({p for values in ROLE_PERMISSIONS.values() for p in values}))


def utc_now() -> datetime:
    return datetime.now(UTC)


class AuthBase(DeclarativeBase):
    pass


class Role(AuthBase):
    __tablename__ = "roles"
    name: Mapped[str] = mapped_column(String(32), primary_key=True)
    description: Mapped[str] = mapped_column(String(160), nullable=False)


class Permission(AuthBase):
    __tablename__ = "permissions"
    name: Mapped[str] = mapped_column(String(64), primary_key=True)
    description: Mapped[str] = mapped_column(String(200), nullable=False)


class RolePermission(AuthBase):
    __tablename__ = "role_permissions"
    role_name: Mapped[str] = mapped_column(
        ForeignKey("roles.name", ondelete="CASCADE"), primary_key=True
    )
    permission_name: Mapped[str] = mapped_column(
        ForeignKey("permissions.name", ondelete="CASCADE"), primary_key=True
    )


class User(AuthBase):
    __tablename__ = "users"
    id: Mapped[str] = mapped_column(String(128), primary_key=True)
    username: Mapped[str] = mapped_column(String(64), unique=True, nullable=False)
    display_name: Mapped[str] = mapped_column(String(120), nullable=False, default="")
    password_hash: Mapped[str | None] = mapped_column(Text, nullable=True)
    role_name: Mapped[str] = mapped_column(
        ForeignKey("roles.name"), nullable=False, index=True
    )
    is_active: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)
    must_change_password: Mapped[bool] = mapped_column(
        Boolean, nullable=False, default=True
    )
    token_version: Mapped[int] = mapped_column(Integer, nullable=False, default=1)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=utc_now
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=utc_now
    )


class RefreshSession(AuthBase):
    __tablename__ = "refresh_sessions"
    __table_args__ = (
        Index("ix_refresh_sessions_user_family", "user_id", "family_id"),
    )
    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    family_id: Mapped[str] = mapped_column(String(36), nullable=False)
    user_id: Mapped[str] = mapped_column(
        ForeignKey("users.id", ondelete="CASCADE"), nullable=False
    )
    token_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    expires_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    revoked_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    replaced_by: Mapped[str | None] = mapped_column(String(36))
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=utc_now
    )
    last_used_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    ip_address: Mapped[str] = mapped_column(String(64), nullable=False, default="")
    user_agent: Mapped[str] = mapped_column(String(300), nullable=False, default="")


class SecurityAuditEvent(AuthBase):
    __tablename__ = "security_audit_events"
    __table_args__ = (
        Index("ix_security_events_actor_created", "actor_user_id", "created_at"),
    )
    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    actor_user_id: Mapped[str | None] = mapped_column(String(128))
    target_user_id: Mapped[str | None] = mapped_column(String(128))
    event_type: Mapped[str] = mapped_column(String(64), nullable=False)
    status: Mapped[str] = mapped_column(String(16), nullable=False)
    detail: Mapped[str] = mapped_column(String(300), nullable=False, default="")
    ip_address: Mapped[str] = mapped_column(String(64), nullable=False, default="")
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=utc_now
    )


class ExternalIdentity(AuthBase):
    """Stable third-party identity linked to exactly one application user."""

    __tablename__ = "external_identities"
    __table_args__ = (
        UniqueConstraint(
            "user_id", "provider", name="uq_external_identity_user_provider"
        ),
    )

    provider: Mapped[str] = mapped_column(String(32), primary_key=True)
    subject: Mapped[str] = mapped_column(String(255), primary_key=True)
    user_id: Mapped[str] = mapped_column(
        ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True
    )
    email: Mapped[str] = mapped_column(String(320), nullable=False, default="")
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=utc_now
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=utc_now
    )


class AuthenticationError(PermissionError):
    pass


class AuthorizationError(PermissionError):
    pass


class AuthConfigurationError(RuntimeError):
    pass


class DuplicateUsernameError(ValueError):
    pass


class LastAdminError(ValueError):
    pass


class TokenReuseError(AuthenticationError):
    pass


_password_hasher = PasswordHasher(time_cost=3, memory_cost=65536, parallelism=4)
_dummy_password_hash = _password_hasher.hash("timing-only-password-value")


def normalize_username(value: str) -> str:
    username = (value or "").strip().casefold()
    if not USERNAME_RE.fullmatch(username):
        raise ValueError(
            "Username must be 3-64 lowercase letters, numbers, dots, underscores, or hyphens"
        )
    return username


def validate_password(value: str) -> str:
    if not isinstance(value, str) or not 12 <= len(value) <= 128:
        raise ValueError("Password must be between 12 and 128 characters")
    return value


def hash_password(value: str) -> str:
    return _password_hasher.hash(validate_password(value))


def verify_password(password_hash: str | None, value: str) -> bool:
    if not password_hash:
        return False
    try:
        return _password_hasher.verify(password_hash, value)
    except (VerifyMismatchError, InvalidHashError):
        return False


def generate_temporary_password() -> str:
    return secrets.token_urlsafe(18)


def _user_to_dict(user: User, permissions: list[str] | None = None) -> dict:
    result = {
        "user_id": user.id,
        "username": user.username,
        "display_name": user.display_name,
        "role": user.role_name,
        "is_active": user.is_active,
        "must_change_password": user.must_change_password,
        "has_password": bool(user.password_hash),
        "token_version": user.token_version,
        "created_at": user.created_at.isoformat(),
        "updated_at": user.updated_at.isoformat(),
    }
    if permissions is not None:
        result["permissions"] = permissions
    return result


class AuthRepository:
    """Own authentication persistence and fixed-role assignment rules."""

    def __init__(self, session_factory: Callable[[], Session]):
        self.session_factory = session_factory

    @staticmethod
    def _permissions(session: Session, role_name: str) -> list[str]:
        return list(session.scalars(
            select(RolePermission.permission_name)
            .where(RolePermission.role_name == role_name)
            .order_by(RolePermission.permission_name)
        ))

    def get_user(self, user_id: str, *, principal: bool = False) -> dict | None:
        with self.session_factory() as session:
            user = session.get(User, user_id)
            if user is None:
                return None
            permissions = self._permissions(session, user.role_name) if principal else None
            return _user_to_dict(user, permissions)

    def get_user_by_username(self, username: str) -> tuple[dict, str | None] | None:
        normalized = normalize_username(username)
        with self.session_factory() as session:
            user = session.scalar(select(User).where(User.username == normalized))
            if user is None:
                return None
            permissions = self._permissions(session, user.role_name)
            return _user_to_dict(user, permissions), user.password_hash

    def get_or_create_external_user(
        self,
        *,
        provider: str,
        subject: str,
        email: str,
        display_name: str,
    ) -> dict:
        """Resolve an external identity or provision a least-privilege user."""
        normalized_provider = provider.strip().casefold()
        normalized_subject = subject.strip()
        if not normalized_provider or not normalized_subject:
            raise ValueError("External identity provider and subject are required")

        with self.session_factory.begin() as session:
            identity = session.get(
                ExternalIdentity,
                {"provider": normalized_provider, "subject": normalized_subject},
            )
            if identity is not None:
                user = session.get(User, identity.user_id)
                if user is None or not user.is_active:
                    raise AuthenticationError("Invalid or inactive account")
                if email and identity.email != email:
                    identity.email = email[:320]
                    identity.updated_at = utc_now()
                return _user_to_dict(
                    user, self._permissions(session, user.role_name)
                )

            user = User(
                id=str(uuid.uuid4()),
                username=f"oauth_{uuid.uuid4().hex[:32]}",
                display_name=(display_name or email or "Google user").strip()[:120],
                password_hash=None,
                role_name="user",
                is_active=True,
                must_change_password=False,
            )
            session.add(user)
            session.flush()
            session.add(ExternalIdentity(
                provider=normalized_provider,
                subject=normalized_subject,
                user_id=user.id,
                email=(email or "")[:320],
            ))
            session.flush()
            return _user_to_dict(user, self._permissions(session, "user"))

    def link_external_identity(
        self,
        user_id: str,
        *,
        provider: str,
        subject: str,
        email: str,
    ) -> None:
        """Link a verified identity without silently merging accounts by email."""
        normalized_provider = provider.strip().casefold()
        normalized_subject = subject.strip()
        with self.session_factory.begin() as session:
            user = session.get(User, user_id)
            if user is None or not user.is_active:
                raise AuthenticationError("Invalid or inactive account")
            by_subject = session.get(
                ExternalIdentity,
                {"provider": normalized_provider, "subject": normalized_subject},
            )
            if by_subject is not None and by_subject.user_id != user_id:
                raise AuthenticationError(
                    "This Google account is already linked to another user"
                )
            by_user = session.scalar(select(ExternalIdentity).where(
                ExternalIdentity.user_id == user_id,
                ExternalIdentity.provider == normalized_provider,
            ))
            if by_user is not None and by_user.subject != normalized_subject:
                raise AuthenticationError(
                    "This user is already linked to another Google account"
                )
            identity = by_subject or by_user
            if identity is None:
                identity = ExternalIdentity(
                    provider=normalized_provider,
                    subject=normalized_subject,
                    user_id=user_id,
                    email=(email or "")[:320],
                )
                session.add(identity)
            else:
                identity.email = (email or identity.email)[:320]
                identity.updated_at = utc_now()

    def external_identity_for_user(
        self, user_id: str, provider: str
    ) -> dict | None:
        with self.session_factory() as session:
            identity = session.scalar(select(ExternalIdentity).where(
                ExternalIdentity.user_id == user_id,
                ExternalIdentity.provider == provider.strip().casefold(),
            ))
            if identity is None:
                return None
            return {
                "provider": identity.provider,
                "subject": identity.subject,
                "email": identity.email,
            }

    def principal(self, user_id: str) -> dict:
        user = self.get_user(user_id, principal=True)
        if not user or not user["is_active"]:
            raise AuthenticationError("Invalid or inactive account")
        return user

    def create_user(
        self, username: str, display_name: str, role: str, password: str
    ) -> dict:
        normalized = normalize_username(username)
        if role not in ROLE_PERMISSIONS:
            raise ValueError("Invalid role")
        try:
            with self.session_factory.begin() as session:
                if session.scalar(select(User.id).where(User.username == normalized)):
                    raise DuplicateUsernameError("Username already exists")
                user = User(
                    id=str(uuid.uuid4()), username=normalized,
                    display_name=(display_name or "").strip()[:120],
                    password_hash=hash_password(password), role_name=role,
                    is_active=True, must_change_password=True,
                )
                session.add(user)
                session.flush()
                result = _user_to_dict(user, self._permissions(session, role))
        except IntegrityError as exc:
            raise DuplicateUsernameError("Username already exists") from exc
        return result

    def list_users(
        self, *, offset: int = 0, limit: int = 50,
        role: str | None = None, is_active: bool | None = None,
    ) -> tuple[list[dict], int]:
        filters = []
        if role:
            if role not in ROLE_PERMISSIONS:
                raise ValueError("Invalid role")
            filters.append(User.role_name == role)
        if is_active is not None:
            filters.append(User.is_active == is_active)
        with self.session_factory() as session:
            total = session.scalar(select(func.count()).select_from(User).where(*filters)) or 0
            rows = list(session.scalars(
                select(User).where(*filters).order_by(User.username).offset(offset).limit(limit)
            ))
            return [_user_to_dict(row) for row in rows], int(total)

    def update_user(
        self, user_id: str, *, role: str | None = None,
        is_active: bool | None = None,
    ) -> dict:
        if role is not None and role not in ROLE_PERMISSIONS:
            raise ValueError("Invalid role")
        with self.session_factory.begin() as session:
            user = session.get(User, user_id)
            if user is None:
                raise LookupError("User not found")
            removes_active_admin = (
                user.role_name == "admin" and user.is_active
                and ((role is not None and role != "admin") or is_active is False)
            )
            if removes_active_admin:
                active_admins = list(session.scalars(
                    select(User.id).where(
                        User.role_name == "admin", User.is_active.is_(True)
                    ).with_for_update()
                ))
                if len(active_admins) <= 1:
                    raise LastAdminError("At least one active admin is required")
            changed = False
            if role is not None and role != user.role_name:
                user.role_name = role
                changed = True
            if is_active is not None and is_active != user.is_active:
                user.is_active = is_active
                changed = True
            if changed:
                user.token_version += 1
                user.updated_at = utc_now()
                session.execute(delete(RefreshSession).where(RefreshSession.user_id == user.id))
            session.flush()
            return _user_to_dict(user, self._permissions(session, user.role_name))

    def set_password(
        self, user_id: str, password: str, *, must_change: bool,
        expected_current: str | None = None,
    ) -> dict:
        with self.session_factory.begin() as session:
            user = session.get(User, user_id)
            if user is None:
                raise LookupError("User not found")
            if expected_current is not None and not verify_password(
                user.password_hash, expected_current
            ):
                raise AuthenticationError("Current password is incorrect")
            user.password_hash = hash_password(password)
            user.must_change_password = must_change
            user.token_version += 1
            user.updated_at = utc_now()
            session.execute(delete(RefreshSession).where(RefreshSession.user_id == user.id))
            session.flush()
            return _user_to_dict(user, self._permissions(session, user.role_name))

    def bootstrap_admin(self, username: str, display_name: str, password: str) -> dict:
        normalized = normalize_username(username)
        with self.session_factory.begin() as session:
            existing = session.scalar(select(User).where(User.username == normalized))
            legacy = session.get(User, "user_admin")
            user = existing or legacy
            if user is None:
                user = User(id="user_admin", username=normalized, display_name="")
                session.add(user)
            elif existing is None:
                user.username = normalized
            user.display_name = (display_name or username).strip()[:120]
            user.password_hash = hash_password(password)
            user.role_name = "admin"
            user.is_active = True
            user.must_change_password = False
            user.token_version = (user.token_version or 0) + 1
            user.updated_at = utc_now()
            session.flush()
            return _user_to_dict(user, self._permissions(session, "admin"))

    def save_refresh(
        self, *, session_id: str, family_id: str, user_id: str,
        token_hash: str, expires_at: datetime, ip_address: str, user_agent: str,
    ) -> None:
        with self.session_factory.begin() as session:
            session.add(RefreshSession(
                id=session_id, family_id=family_id, user_id=user_id,
                token_hash=token_hash, expires_at=expires_at,
                ip_address=ip_address[:64], user_agent=user_agent[:300],
            ))

    def rotate_refresh(
        self, session_id: str, token_hash: str, *, new_session_id: str,
        new_token_hash: str, expires_at: datetime, ip_address: str, user_agent: str,
    ) -> tuple[dict, str]:
        now = utc_now()
        reuse_detected = False
        result: tuple[dict, str] | None = None
        with self.session_factory.begin() as session:
            old = session.get(RefreshSession, session_id)
            if old is None:
                raise AuthenticationError("Invalid refresh token")
            old_expiry = old.expires_at
            if old_expiry.tzinfo is None:
                old_expiry = old_expiry.replace(tzinfo=UTC)
            if old.revoked_at is not None or not secrets.compare_digest(old.token_hash, token_hash):
                session.query(RefreshSession).filter(
                    RefreshSession.family_id == old.family_id
                ).update({RefreshSession.revoked_at: now})
                reuse_detected = True
            elif old_expiry <= now:
                raise AuthenticationError("Refresh token expired")
            else:
                user = session.get(User, old.user_id)
                if user is None or not user.is_active:
                    raise AuthenticationError("Invalid or inactive account")
                old.revoked_at = now
                old.last_used_at = now
                old.replaced_by = new_session_id
                session.add(RefreshSession(
                    id=new_session_id, family_id=old.family_id, user_id=user.id,
                    token_hash=new_token_hash, expires_at=expires_at,
                    ip_address=ip_address[:64], user_agent=user_agent[:300],
                ))
                permissions = self._permissions(session, user.role_name)
                result = (_user_to_dict(user, permissions), old.family_id)
        if reuse_detected:
            raise TokenReuseError("Refresh token reuse detected")
        if result is None:
            raise AuthenticationError("Invalid refresh token")
        return result

    def revoke_session(self, session_id: str) -> None:
        with self.session_factory.begin() as session:
            row = session.get(RefreshSession, session_id)
            if row and row.revoked_at is None:
                row.revoked_at = utc_now()

    def session_is_active(self, session_id: str, user_id: str) -> bool:
        with self.session_factory() as session:
            row = session.get(RefreshSession, session_id)
            if row is None or row.user_id != user_id or row.revoked_at is not None:
                return False
            expires_at = row.expires_at
            if expires_at.tzinfo is None:
                expires_at = expires_at.replace(tzinfo=UTC)
            return expires_at > utc_now()

    def log_event(
        self, event_type: str, status: str, *, actor_user_id: str | None = None,
        target_user_id: str | None = None, detail: str = "", ip_address: str = "",
    ) -> None:
        with self.session_factory.begin() as session:
            session.add(SecurityAuditEvent(
                id=str(uuid.uuid4()), actor_user_id=actor_user_id,
                target_user_id=target_user_id, event_type=event_type,
                status=status, detail=detail[:300], ip_address=ip_address[:64],
            ))


def _secret() -> str:
    if len(JWT_SECRET) < 32:
        raise AuthConfigurationError("JWT_SECRET must contain at least 32 characters")
    return JWT_SECRET


def _token_hash(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def _encode_token(user: dict, token_type: str, lifetime: timedelta, sid: str) -> str:
    now = utc_now()
    return jwt.encode({
        "sub": user["user_id"], "type": token_type, "sid": sid,
        "jti": str(uuid.uuid4()), "token_version": user["token_version"],
        "iat": now, "exp": now + lifetime,
    }, _secret(), algorithm="HS256")


def _decode_token(value: str, expected_type: str) -> dict:
    try:
        claims = jwt.decode(
            value, _secret(), algorithms=["HS256"],
            options={"require": ["sub", "type", "sid", "jti", "token_version", "iat", "exp"]},
        )
    except jwt.PyJWTError as exc:
        raise AuthenticationError("Invalid or expired token") from exc
    if claims.get("type") != expected_type:
        raise AuthenticationError("Invalid token type")
    return claims


@dataclass(frozen=True)
class IssuedSession:
    access_token: str
    refresh_token: str
    expires_in: int
    user: dict


class AuthService:
    def __init__(self, repository: AuthRepository):
        self.repository = repository

    def authenticate(self, username: str, password: str, *, ip_address: str = "") -> dict:
        try:
            found = self.repository.get_user_by_username(username)
        except ValueError:
            found = None
        password_matches = verify_password(
            found[1] if found is not None else _dummy_password_hash,
            password,
        )
        if found is None or not found[0]["is_active"] or not password_matches:
            self.repository.log_event("login", "failure", detail="invalid_credentials", ip_address=ip_address)
            raise AuthenticationError("Invalid username or password")
        user = found[0]
        self.repository.log_event("login", "success", actor_user_id=user["user_id"], ip_address=ip_address)
        return user

    def issue_password_change_token(self, user: dict) -> str:
        return _encode_token(
            user, "password_change", timedelta(minutes=JWT_PASSWORD_CHANGE_MINUTES),
            str(uuid.uuid4()),
        )

    def issue_session(self, user: dict, *, ip_address: str = "", user_agent: str = "") -> IssuedSession:
        session_id = str(uuid.uuid4())
        family_id = str(uuid.uuid4())
        refresh = _encode_token(user, "refresh", timedelta(days=JWT_REFRESH_DAYS), session_id)
        self.repository.save_refresh(
            session_id=session_id, family_id=family_id, user_id=user["user_id"],
            token_hash=_token_hash(refresh), expires_at=utc_now() + timedelta(days=JWT_REFRESH_DAYS),
            ip_address=ip_address, user_agent=user_agent,
        )
        access = _encode_token(user, "access", timedelta(minutes=JWT_ACCESS_MINUTES), session_id)
        return IssuedSession(access, refresh, JWT_ACCESS_MINUTES * 60, user)

    def principal_from_access(self, value: str) -> dict:
        claims = _decode_token(value, "access")
        user = self.repository.principal(claims["sub"])
        if (
            user["token_version"] != claims["token_version"]
            or user["must_change_password"]
            or not self.repository.session_is_active(claims["sid"], user["user_id"])
        ):
            raise AuthenticationError("Session is no longer valid")
        return user

    def complete_password_change(self, value: str, new_password: str) -> dict:
        claims = _decode_token(value, "password_change")
        user = self.repository.principal(claims["sub"])
        if user["token_version"] != claims["token_version"] or not user["must_change_password"]:
            raise AuthenticationError("Password change token is no longer valid")
        return self.repository.set_password(user["user_id"], new_password, must_change=False)

    def refresh(self, value: str, *, ip_address: str = "", user_agent: str = "") -> IssuedSession:
        claims = _decode_token(value, "refresh")
        new_session_id = str(uuid.uuid4())
        provisional_user = self.repository.principal(claims["sub"])
        new_refresh = _encode_token(
            provisional_user, "refresh", timedelta(days=JWT_REFRESH_DAYS), new_session_id
        )
        try:
            user, _family_id = self.repository.rotate_refresh(
                claims["sid"], _token_hash(value), new_session_id=new_session_id,
                new_token_hash=_token_hash(new_refresh),
                expires_at=utc_now() + timedelta(days=JWT_REFRESH_DAYS),
                ip_address=ip_address, user_agent=user_agent,
            )
        except TokenReuseError:
            self.repository.log_event(
                "refresh_reuse", "failure", target_user_id=claims["sub"],
                detail="session_family_revoked", ip_address=ip_address,
            )
            raise
        if user["token_version"] != claims["token_version"] or user["must_change_password"]:
            self.repository.revoke_session(new_session_id)
            raise AuthenticationError("Session is no longer valid")
        access = _encode_token(user, "access", timedelta(minutes=JWT_ACCESS_MINUTES), new_session_id)
        self.repository.log_event(
            "refresh", "success", actor_user_id=user["user_id"], ip_address=ip_address
        )
        return IssuedSession(access, new_refresh, JWT_ACCESS_MINUTES * 60, user)

    def logout(self, refresh_token: str | None, *, ip_address: str = "") -> None:
        if not refresh_token:
            return
        try:
            claims = _decode_token(refresh_token, "refresh")
        except (AuthenticationError, AuthConfigurationError):
            return
        self.repository.revoke_session(claims["sid"])
        self.repository.log_event(
            "logout", "success", actor_user_id=claims["sub"], ip_address=ip_address
        )


class LoginRateLimiter:
    def __init__(self, max_attempts: int = 8, window_seconds: int = 300):
        self.max_attempts = max_attempts
        self.window_seconds = window_seconds
        self._buckets: dict[str, list[float]] = {}
        self._lock = threading.Lock()

    def check(self, key: str) -> None:
        now = time.monotonic()
        cutoff = now - self.window_seconds
        with self._lock:
            attempts = [stamp for stamp in self._buckets.get(key, []) if stamp > cutoff]
            if len(attempts) >= self.max_attempts:
                raise AuthenticationError("Too many login attempts; try again later")
            attempts.append(now)
            self._buckets[key] = attempts


auth_repository = AuthRepository(SessionLocal)
auth_service = AuthService(auth_repository)
login_rate_limiter = LoginRateLimiter()
