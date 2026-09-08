"""Add per-user Google identity and encrypted Drive OAuth grants."""

from alembic import op
import sqlalchemy as sa


revision = "20260905_04"
down_revision = "20260905_03"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "external_identities",
        sa.Column("provider", sa.String(length=32), nullable=False),
        sa.Column("subject", sa.String(length=255), nullable=False),
        sa.Column("user_id", sa.String(length=128), nullable=False),
        sa.Column("email", sa.String(length=320), server_default="", nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.ForeignKeyConstraint(["user_id"], ["users.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("provider", "subject"),
        sa.UniqueConstraint(
            "user_id", "provider", name="uq_external_identity_user_provider"
        ),
    )
    op.create_index(
        "ix_external_identities_user_id",
        "external_identities",
        ["user_id"],
    )

    op.create_table(
        "google_drive_connections",
        sa.Column("user_id", sa.String(length=128), nullable=False),
        sa.Column("google_subject", sa.String(length=255), nullable=False),
        sa.Column("email", sa.String(length=320), server_default="", nullable=False),
        sa.Column("encrypted_refresh_token", sa.Text(), nullable=False),
        sa.Column("scopes", sa.Text(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.ForeignKeyConstraint(["user_id"], ["users.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("user_id"),
        sa.UniqueConstraint("google_subject"),
    )

    op.create_table(
        "google_oauth_states",
        sa.Column("state_hash", sa.String(length=64), nullable=False),
        sa.Column("mode", sa.String(length=16), nullable=False),
        sa.Column("user_id", sa.String(length=128), nullable=True),
        sa.Column("code_verifier", sa.String(length=160), nullable=False),
        sa.Column("nonce", sa.String(length=128), nullable=False),
        sa.Column("browser_binding_hash", sa.String(length=64), nullable=False),
        sa.Column("expires_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.ForeignKeyConstraint(["user_id"], ["users.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("state_hash"),
    )
    op.create_index(
        "ix_google_oauth_states_expires_at",
        "google_oauth_states",
        ["expires_at"],
    )

    op.execute(
        "INSERT INTO permissions (name, description) "
        "VALUES ('local_file:read', 'Local File Read') "
        "ON CONFLICT (name) DO NOTHING"
    )
    op.execute(
        "INSERT INTO role_permissions (role_name, permission_name) "
        "VALUES ('admin', 'local_file:read') "
        "ON CONFLICT (role_name, permission_name) DO NOTHING"
    )


def downgrade() -> None:
    op.execute(
        "DELETE FROM role_permissions "
        "WHERE role_name = 'admin' AND permission_name = 'local_file:read'"
    )
    op.execute(
        "DELETE FROM permissions WHERE name = 'local_file:read'"
    )
    op.drop_index("ix_google_oauth_states_expires_at", table_name="google_oauth_states")
    op.drop_table("google_oauth_states")
    op.drop_table("google_drive_connections")
    op.drop_index("ix_external_identities_user_id", table_name="external_identities")
    op.drop_table("external_identities")
