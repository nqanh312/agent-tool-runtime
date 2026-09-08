"""Tests for per-user Google identity, OAuth state, and token storage."""

import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from cryptography.fernet import Fernet
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

import services.google_oauth as oauth_module
from services.auth import (
    AuthBase,
    AuthRepository,
    AuthenticationError,
    Permission,
    ROLE_PERMISSIONS,
    Role,
    RolePermission,
)
from services.google_oauth import (
    GoogleOAuthError,
    GoogleOAuthRepository,
    GoogleOAuthService,
)


class GoogleOAuthTests(unittest.TestCase):
    def setUp(self):
        self.engine = create_engine(
            "sqlite+pysqlite:///:memory:",
            connect_args={"check_same_thread": False},
            poolclass=StaticPool,
        )
        AuthBase.metadata.create_all(self.engine)
        self.session_factory = sessionmaker(bind=self.engine, expire_on_commit=False)
        with self.session_factory.begin() as session:
            for role in ROLE_PERMISSIONS:
                session.add(Role(name=role, description=role))
            permissions = sorted({
                permission
                for values in ROLE_PERMISSIONS.values()
                for permission in values
            })
            for permission in permissions:
                session.add(Permission(name=permission, description=permission))
            for role, permissions in ROLE_PERMISSIONS.items():
                for permission in permissions:
                    session.add(RolePermission(
                        role_name=role, permission_name=permission
                    ))
        self.users = AuthRepository(self.session_factory)
        self.oauth_repository = GoogleOAuthRepository(self.session_factory)
        self.encryption_key = Fernet.generate_key().decode("ascii")
        self.config_patch = patch.multiple(
            oauth_module,
            GOOGLE_OAUTH_CLIENT_ID="client-id",
            GOOGLE_OAUTH_CLIENT_SECRET="client-secret",
            GOOGLE_OAUTH_REDIRECT_URI="https://app.example.test/api/auth/google/callback",
            GOOGLE_TOKEN_ENCRYPTION_KEY=self.encryption_key,
            GOOGLE_OAUTH_DRIVE_SCOPES=(
                "https://www.googleapis.com/auth/drive.readonly",
            ),
            GOOGLE_OAUTH_ALLOWED_DOMAIN="",
            GOOGLE_OAUTH_STATE_MINUTES=10,
        )
        self.config_patch.start()
        self.service = GoogleOAuthService(self.oauth_repository, self.users)

    def tearDown(self):
        self.config_patch.stop()
        self.engine.dispose()

    def test_external_identity_uses_subject_and_does_not_merge_by_email(self):
        first = self.users.get_or_create_external_user(
            provider="google",
            subject="subject-1",
            email="same@example.test",
            display_name="First",
        )
        repeated = self.users.get_or_create_external_user(
            provider="google",
            subject="subject-1",
            email="renamed@example.test",
            display_name="Ignored",
        )
        second = self.users.get_or_create_external_user(
            provider="google",
            subject="subject-2",
            email="same@example.test",
            display_name="Second",
        )

        self.assertEqual(first["user_id"], repeated["user_id"])
        self.assertNotEqual(first["user_id"], second["user_id"])
        self.assertEqual(first["role"], "user")
        self.assertFalse(first["has_password"])

    def test_oauth_state_is_one_time_and_browser_bound(self):
        self.oauth_repository.save_state(
            "state-value",
            mode="login",
            user_id=None,
            code_verifier="verifier",
            nonce="nonce",
            browser_binding="browser-secret",
        )
        with self.assertRaisesRegex(GoogleOAuthError, "binding"):
            self.oauth_repository.consume_state("state-value", "wrong-browser")

        # A request without the HttpOnly binding cannot burn another browser's
        # valid state. The owner can still consume it exactly once.
        state = self.oauth_repository.consume_state(
            "state-value", "browser-secret"
        )
        self.assertEqual(state["mode"], "login")
        self.assertNotEqual(state["browser_binding_hash"], "browser-secret")
        with self.assertRaises(GoogleOAuthError):
            self.oauth_repository.consume_state("state-value", "browser-secret")

    def test_refresh_token_is_encrypted_and_scoped_to_one_user(self):
        user = self.users.get_or_create_external_user(
            provider="google",
            subject="subject-1",
            email="user@example.test",
            display_name="User",
        )
        encrypted = self.service._encrypt("raw-refresh-token")
        self.oauth_repository.save_connection(
            user_id=user["user_id"],
            google_subject="subject-1",
            email="user@example.test",
            encrypted_refresh_token=encrypted,
            scopes=oauth_module.GOOGLE_OAUTH_DRIVE_SCOPES,
        )

        stored = self.oauth_repository.get_connection(user["user_id"])
        self.assertNotIn("raw-refresh-token", stored["encrypted_refresh_token"])
        credentials = self.service.credentials_for_user(user["user_id"])
        self.assertEqual(credentials.refresh_token, "raw-refresh-token")

        other = self.users.create_user(
            "other-user", "Other", "user", "temporary-pass-123"
        )
        with self.assertRaises(GoogleOAuthError):
            self.oauth_repository.save_connection(
                user_id=other["user_id"],
                google_subject="subject-1",
                email="user@example.test",
                encrypted_refresh_token=self.service._encrypt("another-token"),
                scopes=oauth_module.GOOGLE_OAUTH_DRIVE_SCOPES,
            )

    def test_linking_does_not_move_an_identity_between_users(self):
        owner = self.users.get_or_create_external_user(
            provider="google",
            subject="subject-1",
            email="owner@example.test",
            display_name="Owner",
        )
        other = self.users.create_user(
            "local-user", "Local", "user", "temporary-pass-123"
        )
        with self.assertRaises(AuthenticationError):
            self.users.link_external_identity(
                other["user_id"],
                provider="google",
                subject="subject-1",
                email="owner@example.test",
            )
        self.assertIsNotNone(owner)

    def test_drive_callback_verifies_identity_and_stores_offline_grant(self):
        user = self.users.create_user(
            "local-owner", "Local Owner", "user", "temporary-pass-123"
        )
        self.oauth_repository.save_state(
            "callback-state",
            mode="drive",
            user_id=user["user_id"],
            code_verifier="pkce-verifier",
            nonce="id-token-nonce",
            browser_binding="browser-secret",
        )
        flow = MagicMock()
        flow.credentials = SimpleNamespace(
            id_token="signed-id-token",
            refresh_token="offline-refresh-token",
            granted_scopes=[
                *oauth_module.LOGIN_SCOPES,
                *oauth_module.GOOGLE_OAUTH_DRIVE_SCOPES,
            ],
            scopes=None,
        )
        claims = {
            "sub": "google-subject",
            "email": "owner@example.test",
            "email_verified": True,
            "name": "Owner",
            "nonce": "id-token-nonce",
        }

        with (
            patch.object(
                oauth_module.Flow,
                "from_client_config",
                return_value=flow,
            ),
            patch.object(
                oauth_module.google_id_token,
                "verify_oauth2_token",
                return_value=claims,
            ),
        ):
            result = self.service.complete(
                raw_state="callback-state",
                code="authorization-code",
                browser_binding="browser-secret",
            )

        self.assertEqual(result["mode"], "drive")
        self.assertEqual(flow.code_verifier, "pkce-verifier")
        flow.fetch_token.assert_called_once_with(code="authorization-code")
        connection = self.oauth_repository.get_connection(user["user_id"])
        self.assertEqual(connection["google_subject"], "google-subject")
        self.assertNotIn(
            "offline-refresh-token", connection["encrypted_refresh_token"]
        )


if __name__ == "__main__":
    unittest.main()
