"""Remove the retired CLI local-file permission."""

from alembic import op


revision = "20260908_05"
down_revision = "20260905_04"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute(
        "DELETE FROM role_permissions "
        "WHERE permission_name = 'local_file:read'"
    )
    op.execute(
        "DELETE FROM permissions WHERE name = 'local_file:read'"
    )


def downgrade() -> None:
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
