"""Tests for password authentication, JWT rotation, and fixed-role RBAC."""

import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

import server
import services.auth as auth_module
from services.auth import (
    AuthBase,
    AuthRepository,
    AuthService,
    AuthenticationError,
    LastAdminError,
    Permission,
    ROLE_PERMISSIONS,
    Role,
    RolePermission,
    TokenReuseError,
    normalize_username,
    validate_password,
)


class AuthTests(unittest.TestCase):
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
            for permission in sorted({p for values in ROLE_PERMISSIONS.values() for p in values}):
                session.add(Permission(name=permission, description=permission))
            for role, permissions in ROLE_PERMISSIONS.items():
                for permission in permissions:
                    session.add(RolePermission(role_name=role, permission_name=permission))
        self.repository = AuthRepository(self.session_factory)
        self.service = AuthService(self.repository)
        self.secret_patch = patch.object(auth_module, "JWT_SECRET", "test-secret-" * 5)
        self.secret_patch.start()

    def tearDown(self):
        server.app.dependency_overrides.clear()
        self.secret_patch.stop()
        self.engine.dispose()

    def create_user(self, username="member", role="user"):
        return self.repository.create_user(
            username, username.title(), role, "temporary-pass-123"
        )

    def activate_password(self, user):
        return self.repository.set_password(
            user["user_id"], "permanent-pass-123", must_change=False
        )

    def test_username_and_password_validation(self):
        self.assertEqual(normalize_username("  Alice.Smith "), "alice.smith")
        with self.assertRaises(ValueError):
            normalize_username("bad username")
        with self.assertRaises(ValueError):
            validate_password("too-short")

    def test_role_permissions_are_loaded_from_database(self):
        guest = self.create_user(role="guest")
        principal = self.repository.principal(guest["user_id"])
        self.assertIn("memory:read", principal["permissions"])
        self.assertNotIn("memory:write", principal["permissions"])
        self.assertNotIn("users:manage", principal["permissions"])

    def test_password_change_token_then_access_session(self):
        created = self.create_user()
        authenticated = self.service.authenticate("MEMBER", "temporary-pass-123")
        self.assertTrue(authenticated["must_change_password"])
        change_token = self.service.issue_password_change_token(authenticated)
        updated = self.service.complete_password_change(change_token, "permanent-pass-123")
        self.assertFalse(updated["must_change_password"])
        issued = self.service.issue_session(updated)
        principal = self.service.principal_from_access(issued.access_token)
        self.assertEqual(principal["user_id"], created["user_id"])

    def test_refresh_rotation_detects_reuse_and_revokes_family(self):
        user = self.activate_password(self.create_user())
        first = self.service.issue_session(user)
        second = self.service.refresh(first.refresh_token)
        with self.assertRaises(TokenReuseError):
            self.service.refresh(first.refresh_token)
        with self.assertRaises(AuthenticationError):
            self.service.principal_from_access(second.access_token)

    def test_logout_revokes_access_token_immediately(self):
        user = self.activate_password(self.create_user())
        issued = self.service.issue_session(user)
        self.service.logout(issued.refresh_token)
        with self.assertRaises(AuthenticationError):
            self.service.principal_from_access(issued.access_token)

    def test_role_change_revokes_sessions_and_preserves_last_admin(self):
        admin = self.activate_password(self.create_user("admin-one", "admin"))
        issued = self.service.issue_session(admin)
        with self.assertRaises(LastAdminError):
            self.repository.update_user(admin["user_id"], role="user")
        second = self.activate_password(self.create_user("admin-two", "admin"))
        self.repository.update_user(admin["user_id"], role="user")
        with self.assertRaises(AuthenticationError):
            self.service.principal_from_access(issued.access_token)
        self.assertEqual(self.repository.principal(second["user_id"])["role"], "admin")

    def test_http_temporary_password_flow_sets_refresh_cookie(self):
        self.create_user()
        with (
            patch.object(server, "auth_service", self.service),
            patch.object(server, "auth_repository", self.repository),
        ):
            client = TestClient(server.app)
            login = client.post("/api/auth/login", json={
                "username": "member", "password": "temporary-pass-123",
            })
            self.assertEqual(login.status_code, 200)
            self.assertTrue(login.json()["password_change_required"])
            completed = client.post("/api/auth/complete-password-change", json={
                "change_token": login.json()["change_token"],
                "new_password": "permanent-pass-123",
            })
            self.assertEqual(completed.status_code, 200)
            self.assertIn(server.REFRESH_COOKIE, client.cookies)
            me = client.get("/api/auth/me", headers={
                "Authorization": "Bearer " + completed.json()["access_token"]
            })
            self.assertEqual(me.status_code, 200)
            self.assertEqual(me.json()["username"], "member")


if __name__ == "__main__":
    unittest.main()
