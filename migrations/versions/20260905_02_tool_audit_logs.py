"""Create durable tool audit log storage."""

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


revision = "20260905_02"
down_revision = "20260905_01"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "tool_audit_logs",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("context_id", sa.String(length=192), nullable=False),
        sa.Column("user_id", sa.String(length=128), nullable=False),
        sa.Column("role", sa.String(length=64), nullable=False),
        sa.Column("tool", sa.String(length=128), nullable=False),
        sa.Column(
            "arguments",
            postgresql.JSONB(astext_type=sa.Text()),
            nullable=False,
        ),
        sa.Column(
            "result",
            postgresql.JSONB(astext_type=sa.Text()),
            nullable=True,
        ),
        sa.Column("error", sa.Text(), nullable=True),
        sa.Column("status", sa.String(length=16), nullable=False),
        sa.Column(
            "steps",
            postgresql.JSONB(astext_type=sa.Text()),
            nullable=False,
        ),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        "ix_tool_audit_logs_context_created",
        "tool_audit_logs",
        ["user_id", "context_id", "created_at"],
    )


def downgrade() -> None:
    op.drop_index(
        "ix_tool_audit_logs_context_created",
        table_name="tool_audit_logs",
    )
    op.drop_table("tool_audit_logs")
