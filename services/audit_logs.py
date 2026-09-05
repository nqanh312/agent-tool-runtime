"""PostgreSQL persistence for tool-registry audit entries."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Callable
import uuid

from sqlalchemy import DateTime, Index, JSON, String, Text, select
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column

from services.conversations import SessionLocal


UTC = timezone.utc
JsonType = JSON().with_variant(JSONB, "postgresql")


def utc_now() -> datetime:
    return datetime.now(UTC)


class AuditBase(DeclarativeBase):
    pass


class ToolAuditEntry(AuditBase):
    __tablename__ = "tool_audit_logs"
    __table_args__ = (
        Index(
            "ix_tool_audit_logs_context_created",
            "user_id",
            "context_id",
            "created_at",
        ),
    )

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    context_id: Mapped[str] = mapped_column(String(192), nullable=False)
    user_id: Mapped[str] = mapped_column(String(128), nullable=False)
    role: Mapped[str] = mapped_column(String(64), nullable=False)
    tool: Mapped[str] = mapped_column(String(128), nullable=False)
    arguments: Mapped[dict] = mapped_column(JsonType, nullable=False)
    result: Mapped[object | None] = mapped_column(JsonType, nullable=True)
    error: Mapped[str | None] = mapped_column(Text, nullable=True)
    status: Mapped[str] = mapped_column(String(16), nullable=False)
    steps: Mapped[list] = mapped_column(JsonType, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utc_now, nullable=False
    )


def audit_entry_to_dict(value: ToolAuditEntry) -> dict:
    return {
        "id": str(value.id),
        "timestamp": value.created_at.isoformat(),
        "user_id": value.user_id,
        "role": value.role,
        "tool": value.tool,
        "arguments": dict(value.arguments or {}),
        "result": value.result,
        "error": value.error,
        "status": value.status,
        "steps": list(value.steps or []),
    }


class AuditLogRepository:
    """Store and retrieve audit entries scoped to one user and context."""

    def __init__(self, session_factory: Callable[[], Session]):
        self.session_factory = session_factory

    def append(self, context_id: str, entry: dict) -> dict:
        if not context_id:
            raise ValueError("context_id is required for audit persistence")
        with self.session_factory.begin() as session:
            row = ToolAuditEntry(
                context_id=context_id,
                user_id=entry.get("user_id", "anonymous"),
                role=entry.get("role", "unknown"),
                tool=entry.get("tool", "unknown"),
                arguments=entry.get("arguments") or {},
                result=entry.get("result"),
                error=entry.get("error"),
                status=entry.get("status", "error"),
                steps=entry.get("steps") or [],
                created_at=datetime.fromisoformat(entry["timestamp"]),
            )
            session.add(row)
            session.flush()
            result = audit_entry_to_dict(row)
        return result

    def list_entries(
        self,
        context_id: str,
        user_id: str,
        *,
        limit: int = 500,
    ) -> list[dict]:
        safe_limit = min(max(limit, 1), 1000)
        with self.session_factory() as session:
            rows = list(
                session.scalars(
                    select(ToolAuditEntry)
                    .where(
                        ToolAuditEntry.context_id == context_id,
                        ToolAuditEntry.user_id == user_id,
                    )
                    .order_by(ToolAuditEntry.created_at.asc(), ToolAuditEntry.id.asc())
                    .limit(safe_limit)
                )
            )
            return [audit_entry_to_dict(row) for row in rows]


audit_log_repository = AuditLogRepository(SessionLocal)
