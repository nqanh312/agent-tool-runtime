"""Add PostgreSQL-backed users, fixed-role RBAC, and JWT sessions."""

from alembic import op
import sqlalchemy as sa


revision = "20260905_03"
down_revision = "20260905_02"
branch_labels = None
depends_on = None


ROLE_PERMISSIONS = {
    "admin": (
        "chat:use", "conversation:read", "conversation:write", "audit:read",
        "drive:read", "memory:read", "memory:write", "users:manage",
    ),
    "user": (
        "chat:use", "conversation:read", "conversation:write", "audit:read",
        "drive:read", "memory:read", "memory:write",
    ),
    "guest": (
        "chat:use", "conversation:read", "conversation:write", "audit:read",
        "drive:read", "memory:read",
    ),
}


def upgrade() -> None:
    roles = op.create_table(
        "roles",
        sa.Column("name", sa.String(length=32), nullable=False),
        sa.Column("description", sa.String(length=160), nullable=False),
        sa.PrimaryKeyConstraint("name"),
    )
    permissions = op.create_table(
        "permissions",
        sa.Column("name", sa.String(length=64), nullable=False),
        sa.Column("description", sa.String(length=200), nullable=False),
        sa.PrimaryKeyConstraint("name"),
    )
    op.create_table(
        "role_permissions",
        sa.Column("role_name", sa.String(length=32), nullable=False),
        sa.Column("permission_name", sa.String(length=64), nullable=False),
        sa.ForeignKeyConstraint(["role_name"], ["roles.name"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(
            ["permission_name"], ["permissions.name"], ondelete="CASCADE"
        ),
        sa.PrimaryKeyConstraint("role_name", "permission_name"),
    )
    op.bulk_insert(roles, [
        {"name": "admin", "description": "Application administrator"},
        {"name": "user", "description": "Standard application user"},
        {"name": "guest", "description": "Read-only memory guest"},
    ])
    permission_names = sorted({p for values in ROLE_PERMISSIONS.values() for p in values})
    op.bulk_insert(permissions, [
        {"name": name, "description": name.replace(":", " ").title()}
        for name in permission_names
    ])
    role_permissions = sa.table(
        "role_permissions",
        sa.column("role_name", sa.String),
        sa.column("permission_name", sa.String),
    )
    op.bulk_insert(role_permissions, [
        {"role_name": role, "permission_name": permission}
        for role, values in ROLE_PERMISSIONS.items()
        for permission in values
    ])

    op.create_table(
        "users",
        sa.Column("id", sa.String(length=128), nullable=False),
        sa.Column("username", sa.String(length=64), nullable=False),
        sa.Column("display_name", sa.String(length=120), nullable=False),
        sa.Column("password_hash", sa.Text(), nullable=True),
        sa.Column("role_name", sa.String(length=32), nullable=False),
        sa.Column("is_active", sa.Boolean(), server_default=sa.false(), nullable=False),
        sa.Column("must_change_password", sa.Boolean(), server_default=sa.true(), nullable=False),
        sa.Column("token_version", sa.Integer(), server_default="1", nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.ForeignKeyConstraint(["role_name"], ["roles.name"]),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("username", name="uq_users_username"),
    )
    op.create_index("ix_users_role_name", "users", ["role_name"])

    # Reserve the legacy administrator identity so existing Qdrant ownership remains valid.
    op.execute("""
        INSERT INTO users
            (id, username, display_name, password_hash, role_name, is_active,
             must_change_password, token_version, created_at, updated_at)
        VALUES
            ('user_admin', 'admin', 'Administrator', NULL, 'admin', FALSE,
             TRUE, 1, NOW(), NOW())
        ON CONFLICT (id) DO NOTHING
    """)
    op.execute("""
        INSERT INTO users
            (id, username, display_name, password_hash, role_name, is_active,
             must_change_password, token_version, created_at, updated_at)
        SELECT DISTINCT
            c.user_id,
            'legacy_' || SUBSTRING(md5(c.user_id) FROM 1 FOR 16),
            'Legacy account ' || c.user_id,
            NULL,
            CASE WHEN c.user_id = 'user_admin' THEN 'admin' ELSE 'guest' END,
            FALSE, TRUE, 1, NOW(), NOW()
        FROM conversations c
        WHERE c.user_id <> 'user_admin'
        ON CONFLICT (id) DO NOTHING
    """)
    op.create_foreign_key(
        "fk_conversations_user_id_users",
        "conversations", "users", ["user_id"], ["id"], ondelete="RESTRICT",
    )

    op.create_table(
        "refresh_sessions",
        sa.Column("id", sa.String(length=36), nullable=False),
        sa.Column("family_id", sa.String(length=36), nullable=False),
        sa.Column("user_id", sa.String(length=128), nullable=False),
        sa.Column("token_hash", sa.String(length=64), nullable=False),
        sa.Column("expires_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("revoked_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("replaced_by", sa.String(length=36), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("last_used_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("ip_address", sa.String(length=64), server_default="", nullable=False),
        sa.Column("user_agent", sa.String(length=300), server_default="", nullable=False),
        sa.ForeignKeyConstraint(["user_id"], ["users.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        "ix_refresh_sessions_user_family", "refresh_sessions", ["user_id", "family_id"]
    )

    op.create_table(
        "security_audit_events",
        sa.Column("id", sa.String(length=36), nullable=False),
        sa.Column("actor_user_id", sa.String(length=128), nullable=True),
        sa.Column("target_user_id", sa.String(length=128), nullable=True),
        sa.Column("event_type", sa.String(length=64), nullable=False),
        sa.Column("status", sa.String(length=16), nullable=False),
        sa.Column("detail", sa.String(length=300), server_default="", nullable=False),
        sa.Column("ip_address", sa.String(length=64), server_default="", nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        "ix_security_events_actor_created",
        "security_audit_events", ["actor_user_id", "created_at"],
    )


def downgrade() -> None:
    op.drop_index("ix_security_events_actor_created", table_name="security_audit_events")
    op.drop_table("security_audit_events")
    op.drop_index("ix_refresh_sessions_user_family", table_name="refresh_sessions")
    op.drop_table("refresh_sessions")
    op.drop_constraint("fk_conversations_user_id_users", "conversations", type_="foreignkey")
    op.drop_index("ix_users_role_name", table_name="users")
    op.drop_table("users")
    op.drop_table("role_permissions")
    op.drop_table("permissions")
    op.drop_table("roles")
