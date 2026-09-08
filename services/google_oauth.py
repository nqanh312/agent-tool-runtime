"""Per-user Google identity and Drive OAuth authorization."""

from __future__ import annotations

import base64
from datetime import datetime, timedelta, timezone
import hashlib
import secrets
from typing import Callable
import urllib.parse
import urllib.request

from cryptography.fernet import Fernet, InvalidToken
from google.auth.transport.requests import Request as GoogleAuthRequest
from google.oauth2 import credentials as oauth_credentials
from google.oauth2 import id_token as google_id_token
from google_auth_oauthlib.flow import Flow
from sqlalchemy import DateTime, ForeignKey, String, Text, delete, select
from sqlalchemy.orm import Mapped, Session, mapped_column

from config import (
    GOOGLE_OAUTH_ALLOWED_DOMAIN,
    GOOGLE_OAUTH_CLIENT_ID,
    GOOGLE_OAUTH_CLIENT_SECRET,
    GOOGLE_OAUTH_DRIVE_SCOPES,
    GOOGLE_OAUTH_REDIRECT_URI,
    GOOGLE_OAUTH_STATE_MINUTES,
    GOOGLE_TOKEN_ENCRYPTION_KEY,
)
from services.auth import AuthBase, AuthRepository, auth_repository
from services.conversations import SessionLocal


UTC = timezone.utc
LOGIN_SCOPES = (
    "openid",
    "https://www.googleapis.com/auth/userinfo.email",
    "https://www.googleapis.com/auth/userinfo.profile",
)
TOKEN_URI = "https://oauth2.googleapis.com/token"
REVOKE_URI = "https://oauth2.googleapis.com/revoke"


def utc_now() -> datetime:
    return datetime.now(UTC)


class GoogleDriveConnection(AuthBase):
    """Encrypted offline Drive grant owned by one application user."""

    __tablename__ = "google_drive_connections"

    user_id: Mapped[str] = mapped_column(
        ForeignKey("users.id", ondelete="CASCADE"), primary_key=True
    )
    google_subject: Mapped[str] = mapped_column(
        String(255), unique=True, nullable=False
    )
    email: Mapped[str] = mapped_column(String(320), nullable=False, default="")
    encrypted_refresh_token: Mapped[str] = mapped_column(Text, nullable=False)
    scopes: Mapped[str] = mapped_column(Text, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=utc_now
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=utc_now
    )


class GoogleOAuthState(AuthBase):
    """One-time server-side OAuth transaction state and PKCE verifier."""

    __tablename__ = "google_oauth_states"

    state_hash: Mapped[str] = mapped_column(String(64), primary_key=True)
    mode: Mapped[str] = mapped_column(String(16), nullable=False)
    user_id: Mapped[str | None] = mapped_column(
        ForeignKey("users.id", ondelete="CASCADE"), nullable=True
    )
    code_verifier: Mapped[str] = mapped_column(String(160), nullable=False)
    nonce: Mapped[str] = mapped_column(String(128), nullable=False)
    browser_binding_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    expires_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=utc_now
    )


class GoogleOAuthError(RuntimeError):
    pass


class GoogleOAuthConfigurationError(GoogleOAuthError):
    pass


class GoogleDriveNotConnectedError(GoogleOAuthError):
    pass


def _state_hash(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _aware(value: datetime) -> datetime:
    return value if value.tzinfo is not None else value.replace(tzinfo=UTC)


class GoogleOAuthRepository:
    """Persist single-use OAuth transactions and encrypted Drive grants."""

    def __init__(self, session_factory: Callable[[], Session]):
        self.session_factory = session_factory

    def save_state(
        self,
        raw_state: str,
        *,
        mode: str,
        user_id: str | None,
        code_verifier: str,
        nonce: str,
        browser_binding: str,
    ) -> None:
        """Store only hashes of browser-visible transaction secrets."""
        now = utc_now()
        with self.session_factory.begin() as session:
            session.execute(delete(GoogleOAuthState).where(
                GoogleOAuthState.expires_at <= now
            ))
            session.add(GoogleOAuthState(
                state_hash=_state_hash(raw_state),
                mode=mode,
                user_id=user_id,
                code_verifier=code_verifier,
                nonce=nonce,
                browser_binding_hash=_state_hash(browser_binding),
                expires_at=now + timedelta(minutes=GOOGLE_OAUTH_STATE_MINUTES),
            ))

    def consume_state(self, raw_state: str, browser_binding: str) -> dict:
        """Atomically validate and delete one OAuth callback transaction."""
        if not browser_binding:
            raise GoogleOAuthError("OAuth browser binding is missing")
        with self.session_factory.begin() as session:
            row = session.scalar(select(GoogleOAuthState).where(
                GoogleOAuthState.state_hash == _state_hash(raw_state)
            ).with_for_update())
            if row is None:
                raise GoogleOAuthError("Invalid or already-used OAuth state")
            if not secrets.compare_digest(
                row.browser_binding_hash, _state_hash(browser_binding)
            ):
                raise GoogleOAuthError("OAuth browser binding did not match")
            result = {
                "mode": row.mode,
                "user_id": row.user_id,
                "code_verifier": row.code_verifier,
                "nonce": row.nonce,
                "browser_binding_hash": row.browser_binding_hash,
                "expires_at": row.expires_at,
            }
            # Deleting inside the locked transaction prevents callback replay.
            session.delete(row)
        if _aware(result["expires_at"]) <= utc_now():
            raise GoogleOAuthError("OAuth state has expired")
        return result

    def get_connection(self, user_id: str) -> dict | None:
        with self.session_factory() as session:
            row = session.get(GoogleDriveConnection, user_id)
            if row is None:
                return None
            return {
                "user_id": row.user_id,
                "google_subject": row.google_subject,
                "email": row.email,
                "encrypted_refresh_token": row.encrypted_refresh_token,
                "scopes": tuple(filter(None, row.scopes.split(" "))),
                "created_at": row.created_at.isoformat(),
                "updated_at": row.updated_at.isoformat(),
            }

    def save_connection(
        self,
        *,
        user_id: str,
        google_subject: str,
        email: str,
        encrypted_refresh_token: str | None,
        scopes: tuple[str, ...],
    ) -> None:
        """Create or refresh a one-user-to-one-Google-account Drive grant."""
        with self.session_factory.begin() as session:
            owner = session.scalar(select(GoogleDriveConnection.user_id).where(
                GoogleDriveConnection.google_subject == google_subject,
                GoogleDriveConnection.user_id != user_id,
            ))
            if owner is not None:
                raise GoogleOAuthError(
                    "This Google Drive is already connected to another user"
                )
            row = session.get(GoogleDriveConnection, user_id)
            if row is None:
                if not encrypted_refresh_token:
                    raise GoogleOAuthError(
                        "Google did not return an offline refresh token; reconnect with consent"
                    )
                row = GoogleDriveConnection(
                    user_id=user_id,
                    google_subject=google_subject,
                    email=email[:320],
                    encrypted_refresh_token=encrypted_refresh_token,
                    scopes=" ".join(sorted(set(scopes))),
                )
                session.add(row)
            else:
                if row.google_subject != google_subject:
                    raise GoogleOAuthError(
                        "A different Google Drive is already connected to this user"
                    )
                row.email = email[:320]
                if encrypted_refresh_token:
                    row.encrypted_refresh_token = encrypted_refresh_token
                row.scopes = " ".join(sorted(set(scopes)))
                row.updated_at = utc_now()

    def delete_connection(self, user_id: str) -> dict | None:
        with self.session_factory.begin() as session:
            row = session.get(GoogleDriveConnection, user_id)
            if row is None:
                return None
            result = {
                "encrypted_refresh_token": row.encrypted_refresh_token,
                "email": row.email,
            }
            session.delete(row)
            return result


class GoogleOAuthService:
    """Run OIDC login and incremental Google Drive authorization flows."""

    def __init__(
        self,
        repository: GoogleOAuthRepository,
        users: AuthRepository,
    ):
        self.repository = repository
        self.users = users

    @staticmethod
    def configured() -> bool:
        if not (
            GOOGLE_OAUTH_CLIENT_ID
            and GOOGLE_OAUTH_CLIENT_SECRET
            and GOOGLE_OAUTH_REDIRECT_URI
            and GOOGLE_TOKEN_ENCRYPTION_KEY
        ):
            return False
        try:
            Fernet(GOOGLE_TOKEN_ENCRYPTION_KEY.encode("ascii"))
        except (ValueError, TypeError):
            return False
        return True

    @staticmethod
    def _require_configuration() -> None:
        if not GoogleOAuthService.configured():
            raise GoogleOAuthConfigurationError(
                "Google OAuth client and token encryption key are not configured"
            )

    @staticmethod
    def _client_config() -> dict:
        return {
            "web": {
                "client_id": GOOGLE_OAUTH_CLIENT_ID,
                "client_secret": GOOGLE_OAUTH_CLIENT_SECRET,
                "auth_uri": "https://accounts.google.com/o/oauth2/auth",
                "token_uri": TOKEN_URI,
                "redirect_uris": [GOOGLE_OAUTH_REDIRECT_URI],
            }
        }

    @staticmethod
    def _scopes(mode: str) -> tuple[str, ...]:
        if mode == "login":
            return LOGIN_SCOPES
        if mode == "drive":
            return tuple(dict.fromkeys((*LOGIN_SCOPES, *GOOGLE_OAUTH_DRIVE_SCOPES)))
        raise GoogleOAuthError("Invalid Google OAuth mode")

    @staticmethod
    def _fernet() -> Fernet:
        GoogleOAuthService._require_configuration()
        return Fernet(GOOGLE_TOKEN_ENCRYPTION_KEY.encode("ascii"))

    def _encrypt(self, value: str) -> str:
        return self._fernet().encrypt(value.encode("utf-8")).decode("ascii")

    def _decrypt(self, value: str) -> str:
        try:
            return self._fernet().decrypt(value.encode("ascii")).decode("utf-8")
        except InvalidToken as exc:
            raise GoogleOAuthConfigurationError(
                "Stored Google token cannot be decrypted with the configured key"
            ) from exc

    def begin(self, *, mode: str, user_id: str | None = None) -> dict:
        """Create a bound state/nonce/PKCE transaction and authorization URL."""
        self._require_configuration()
        if mode == "drive" and not user_id:
            raise GoogleOAuthError("Authentication is required to connect Drive")
        raw_state = secrets.token_urlsafe(48)
        verifier = secrets.token_urlsafe(72)
        nonce = secrets.token_urlsafe(32)
        browser_binding = secrets.token_urlsafe(48)
        challenge = base64.urlsafe_b64encode(
            hashlib.sha256(verifier.encode("ascii")).digest()
        ).rstrip(b"=").decode("ascii")
        self.repository.save_state(
            raw_state,
            mode=mode,
            user_id=user_id,
            code_verifier=verifier,
            nonce=nonce,
            browser_binding=browser_binding,
        )
        flow = Flow.from_client_config(
            self._client_config(),
            scopes=self._scopes(mode),
            redirect_uri=GOOGLE_OAUTH_REDIRECT_URI,
        )
        authorization_options = {
            "state": raw_state,
            "include_granted_scopes": "true",
            "prompt": "consent" if mode == "drive" else "select_account",
            "nonce": nonce,
            "code_challenge": challenge,
            "code_challenge_method": "S256",
        }
        if mode == "drive":
            authorization_options["access_type"] = "offline"
        authorization_url, _ = flow.authorization_url(**authorization_options)
        return {
            "authorization_url": authorization_url,
            "browser_binding": browser_binding,
        }

    def complete(
        self,
        *,
        raw_state: str,
        code: str,
        browser_binding: str,
    ) -> dict:
        """Exchange a callback, verify identity claims, and apply its mode."""
        self._require_configuration()
        transaction = self.repository.consume_state(raw_state, browser_binding)
        mode = transaction["mode"]
        flow = Flow.from_client_config(
            self._client_config(),
            scopes=self._scopes(mode),
            state=raw_state,
            redirect_uri=GOOGLE_OAUTH_REDIRECT_URI,
        )
        flow.code_verifier = transaction["code_verifier"]
        try:
            flow.fetch_token(code=code)
        except Exception as exc:
            raise GoogleOAuthError(
                "Could not exchange the Google authorization code"
            ) from exc
        credentials = flow.credentials
        if not credentials.id_token:
            raise GoogleOAuthError("Google did not return an ID token")
        try:
            claims = google_id_token.verify_oauth2_token(
                credentials.id_token,
                GoogleAuthRequest(),
                GOOGLE_OAUTH_CLIENT_ID,
                clock_skew_in_seconds=30,
            )
        except Exception as exc:
            raise GoogleOAuthError("Google ID token verification failed") from exc
        # State binds the request, PKCE binds the code exchange, and nonce binds
        # this verified ID token to the exact transaction that initiated it.
        if claims.get("nonce") != transaction["nonce"]:
            raise GoogleOAuthError("Google ID token nonce did not match")
        if claims.get("email_verified") is not True:
            raise GoogleOAuthError("A verified Google email is required")
        if (
            GOOGLE_OAUTH_ALLOWED_DOMAIN
            and str(claims.get("hd", "")).casefold() != GOOGLE_OAUTH_ALLOWED_DOMAIN
        ):
            raise GoogleOAuthError("Google Workspace domain is not allowed")

        subject = str(claims.get("sub", "")).strip()
        if not subject:
            raise GoogleOAuthError("Google ID token has no stable subject")
        email = str(claims.get("email", ""))
        display_name = str(claims.get("name", ""))
        if mode == "login":
            user = self.users.get_or_create_external_user(
                provider="google",
                subject=subject,
                email=email,
                display_name=display_name,
            )
            return {"mode": mode, "user": user}

        user_id = transaction.get("user_id")
        if not user_id:
            raise GoogleOAuthError("Drive connection lost its application user")
        self.users.link_external_identity(
            user_id,
            provider="google",
            subject=subject,
            email=email,
        )
        granted = tuple(
            credentials.granted_scopes
            or credentials.scopes
            or self._scopes(mode)
        )
        missing = set(GOOGLE_OAUTH_DRIVE_SCOPES) - set(granted)
        if missing:
            raise GoogleOAuthError(
                "Google Drive consent did not grant every required scope"
            )
        self.repository.save_connection(
            user_id=user_id,
            google_subject=subject,
            email=email,
            encrypted_refresh_token=(
                self._encrypt(credentials.refresh_token)
                if credentials.refresh_token else None
            ),
            scopes=granted,
        )
        return {"mode": mode, "user": self.users.principal(user_id)}

    def status(self, user_id: str) -> dict:
        connection = self.repository.get_connection(user_id)
        return {
            "configured": self.configured(),
            "connected": connection is not None,
            "email": connection["email"] if connection else "",
            "scopes": list(connection["scopes"]) if connection else [],
        }

    def credentials_for_user(self, user_id: str) -> oauth_credentials.Credentials:
        """Build refreshable credentials from the user's encrypted offline grant."""
        self._require_configuration()
        connection = self.repository.get_connection(user_id)
        if connection is None:
            raise GoogleDriveNotConnectedError(
                "Google Drive is not connected for this user"
            )
        return oauth_credentials.Credentials(
            token=None,
            refresh_token=self._decrypt(connection["encrypted_refresh_token"]),
            token_uri=TOKEN_URI,
            client_id=GOOGLE_OAUTH_CLIENT_ID,
            client_secret=GOOGLE_OAUTH_CLIENT_SECRET,
            scopes=list(connection["scopes"]),
        )

    def disconnect(self, user_id: str) -> bool:
        """Delete the local grant, then best-effort revoke it at Google."""
        connection = self.repository.delete_connection(user_id)
        if connection is None:
            return False
        try:
            token = self._decrypt(connection["encrypted_refresh_token"])
            body = urllib.parse.urlencode({"token": token}).encode("ascii")
            request = urllib.request.Request(REVOKE_URI, data=body, method="POST")
            with urllib.request.urlopen(request, timeout=5):
                pass
        except Exception:
            # Local deletion is authoritative; remote revocation is best-effort.
            pass
        return True


google_oauth_repository = GoogleOAuthRepository(SessionLocal)
google_oauth_service = GoogleOAuthService(google_oauth_repository, auth_repository)
