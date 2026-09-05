"""PostgreSQL persistence for durable conversations and chat messages."""

from __future__ import annotations

import base64
import binascii
from datetime import datetime, timezone
from typing import Callable
import uuid

from sqlalchemy import (
    BigInteger,
    DateTime,
    ForeignKey,
    Index,
    JSON,
    String,
    Text,
    UniqueConstraint,
    and_,
    create_engine,
    func,
    or_,
    select,
    text,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column, sessionmaker

from config import DATABASE_URL


UTC = timezone.utc
JsonType = JSON().with_variant(JSONB, "postgresql")


def utc_now() -> datetime:
    return datetime.now(UTC)


class Base(DeclarativeBase):
    pass


class Conversation(Base):
    __tablename__ = "conversations"
    __table_args__ = (
        Index("ix_conversations_user_updated", "user_id", "updated_at"),
    )

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    user_id: Mapped[str] = mapped_column(String(128), nullable=False, index=True)
    title: Mapped[str] = mapped_column(String(160), nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utc_now, nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utc_now, nullable=False
    )


class ChatMessage(Base):
    __tablename__ = "messages"
    __table_args__ = (
        UniqueConstraint("conversation_id", "ordinal", name="uq_message_ordinal"),
        UniqueConstraint("client_message_id", name="uq_client_message_id"),
        UniqueConstraint("reply_to_message_id", name="uq_reply_to_message_id"),
        Index("ix_messages_conversation_ordinal", "conversation_id", "ordinal"),
    )

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    conversation_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("conversations.id", ondelete="CASCADE"), nullable=False
    )
    ordinal: Mapped[int] = mapped_column(BigInteger, nullable=False)
    role: Mapped[str] = mapped_column(String(16), nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    tools_used: Mapped[list] = mapped_column(JsonType, default=list, nullable=False)
    client_message_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    reply_to_message_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("messages.id", ondelete="SET NULL"), nullable=True
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utc_now, nullable=False
    )


class ConversationState(Base):
    __tablename__ = "conversation_state"

    conversation_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("conversations.id", ondelete="CASCADE"), primary_key=True
    )
    last_artifact: Mapped[dict | None] = mapped_column(JsonType, nullable=True)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utc_now, nullable=False
    )


class ConversationNotFoundError(LookupError):
    """Raised when a conversation does not exist or belongs to another user."""


def make_title(message: str, limit: int = 60) -> str:
    """Create a stable title without an additional LLM request."""
    normalized = " ".join((message or "").split()) or "New chat"
    if len(normalized) <= limit:
        return normalized
    return normalized[: limit - 1].rstrip() + "…"


def _encode_cursor(updated_at: datetime, conversation_id: uuid.UUID) -> str:
    raw = f"{updated_at.isoformat()}|{conversation_id}"
    return base64.urlsafe_b64encode(raw.encode()).decode().rstrip("=")


def _decode_cursor(cursor: str) -> tuple[datetime, uuid.UUID]:
    try:
        padded = cursor + "=" * (-len(cursor) % 4)
        raw = base64.urlsafe_b64decode(padded.encode()).decode()
        timestamp, identifier = raw.rsplit("|", 1)
        parsed = datetime.fromisoformat(timestamp)
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=UTC)
        return parsed, uuid.UUID(identifier)
    except (ValueError, UnicodeError, binascii.Error) as exc:
        raise ValueError("Invalid conversation cursor") from exc


def conversation_to_dict(value: Conversation) -> dict:
    return {
        "id": str(value.id),
        "title": value.title,
        "created_at": value.created_at.isoformat(),
        "updated_at": value.updated_at.isoformat(),
    }


def message_to_dict(value: ChatMessage) -> dict:
    return {
        "id": str(value.id),
        "conversation_id": str(value.conversation_id),
        "ordinal": value.ordinal,
        "role": value.role,
        "content": value.content,
        "tools_used": list(value.tools_used or []),
        "client_message_id": value.client_message_id,
        "reply_to_message_id": (
            str(value.reply_to_message_id) if value.reply_to_message_id else None
        ),
        "created_at": value.created_at.isoformat(),
    }


class ConversationRepository:
    """Keep all persistence and ownership checks outside FastAPI handlers."""

    def __init__(self, session_factory: Callable[[], Session]):
        self.session_factory = session_factory

    def healthcheck(self) -> bool:
        try:
            with self.session_factory() as session:
                session.execute(text("SELECT 1"))
            return True
        except Exception:
            return False

    def create_conversation(self, user_id: str, first_message: str) -> dict:
        with self.session_factory.begin() as session:
            conversation = Conversation(
                user_id=user_id,
                title=make_title(first_message),
            )
            session.add(conversation)
            session.flush()
            result = conversation_to_dict(conversation)
        return result

    def create_conversation_with_message(
        self,
        user_id: str,
        first_message: str,
        client_message_id: str,
    ) -> tuple[dict, dict]:
        """Atomically create a non-empty conversation and its first user turn."""
        with self.session_factory.begin() as session:
            conversation = Conversation(
                user_id=user_id,
                title=make_title(first_message),
            )
            session.add(conversation)
            session.flush()
            message = ChatMessage(
                conversation_id=conversation.id,
                ordinal=1,
                role="user",
                content=first_message,
                client_message_id=client_message_id,
            )
            session.add(message)
            session.flush()
            conversation_result = conversation_to_dict(conversation)
            message_result = message_to_dict(message)
        return conversation_result, message_result

    def get_conversation(self, conversation_id: str, user_id: str) -> dict:
        try:
            identifier = uuid.UUID(conversation_id)
        except ValueError as exc:
            raise ConversationNotFoundError("Conversation not found") from exc
        with self.session_factory() as session:
            conversation = session.scalar(
                select(Conversation).where(
                    Conversation.id == identifier,
                    Conversation.user_id == user_id,
                )
            )
            if conversation is None:
                raise ConversationNotFoundError("Conversation not found")
            return conversation_to_dict(conversation)

    def list_conversations(
        self,
        user_id: str,
        *,
        limit: int = 30,
        cursor: str | None = None,
    ) -> tuple[list[dict], str | None]:
        safe_limit = min(max(limit, 1), 100)
        with self.session_factory() as session:
            query = select(Conversation).where(Conversation.user_id == user_id)
            if cursor:
                cursor_time, cursor_id = _decode_cursor(cursor)
                query = query.where(
                    or_(
                        Conversation.updated_at < cursor_time,
                        and_(
                            Conversation.updated_at == cursor_time,
                            Conversation.id < cursor_id,
                        ),
                    )
                )
            rows = list(
                session.scalars(
                    query.order_by(
                        Conversation.updated_at.desc(), Conversation.id.desc()
                    ).limit(safe_limit + 1)
                )
            )
            has_more = len(rows) > safe_limit
            rows = rows[:safe_limit]
            next_cursor = None
            if has_more and rows:
                next_cursor = _encode_cursor(rows[-1].updated_at, rows[-1].id)
            return [conversation_to_dict(row) for row in rows], next_cursor

    def list_messages(
        self,
        conversation_id: str,
        user_id: str,
        *,
        limit: int = 50,
        before: int | None = None,
    ) -> tuple[list[dict], int | None]:
        conversation = self.get_conversation(conversation_id, user_id)
        identifier = uuid.UUID(conversation["id"])
        safe_limit = min(max(limit, 1), 200)
        with self.session_factory() as session:
            query = select(ChatMessage).where(
                ChatMessage.conversation_id == identifier
            )
            if before is not None:
                query = query.where(ChatMessage.ordinal < before)
            rows = list(
                session.scalars(
                    query.order_by(ChatMessage.ordinal.desc()).limit(safe_limit + 1)
                )
            )
            has_more = len(rows) > safe_limit
            rows = rows[:safe_limit]
            rows.reverse()
            next_before = rows[0].ordinal if has_more and rows else None
            return [message_to_dict(row) for row in rows], next_before

    def list_context_messages(
        self,
        conversation_id: str,
        user_id: str,
        *,
        before_ordinal: int | None = None,
    ) -> list[dict]:
        messages, before = self.list_messages(
            conversation_id,
            user_id,
            limit=200,
            before=before_ordinal,
        )
        while before is not None and len(messages) < 1000:
            older, before = self.list_messages(
                conversation_id, user_id, limit=200, before=before
            )
            messages = older + messages
        return messages

    def append_user_message(
        self,
        conversation_id: str,
        user_id: str,
        content: str,
        client_message_id: str,
    ) -> tuple[dict, bool]:
        identifier = uuid.UUID(conversation_id)
        with self.session_factory.begin() as session:
            conversation = session.scalar(
                select(Conversation)
                .where(
                    Conversation.id == identifier,
                    Conversation.user_id == user_id,
                )
                .with_for_update()
            )
            if conversation is None:
                raise ConversationNotFoundError("Conversation not found")
            existing = session.scalar(
                select(ChatMessage).where(
                    ChatMessage.conversation_id == identifier,
                    ChatMessage.client_message_id == client_message_id,
                )
            )
            if existing is not None:
                return message_to_dict(existing), False
            next_ordinal = (
                session.scalar(
                    select(func.coalesce(func.max(ChatMessage.ordinal), 0)).where(
                        ChatMessage.conversation_id == identifier
                    )
                )
                + 1
            )
            message = ChatMessage(
                conversation_id=identifier,
                ordinal=next_ordinal,
                role="user",
                content=content,
                client_message_id=client_message_id,
            )
            conversation.updated_at = utc_now()
            session.add(message)
            session.flush()
            result = message_to_dict(message)
        return result, True

    def find_user_message(
        self, user_id: str, client_message_id: str
    ) -> tuple[dict, dict] | None:
        """Find a retried user turn even when the client missed its new chat ID."""
        with self.session_factory() as session:
            row = session.execute(
                select(ChatMessage, Conversation)
                .join(Conversation, Conversation.id == ChatMessage.conversation_id)
                .where(
                    Conversation.user_id == user_id,
                    ChatMessage.client_message_id == client_message_id,
                    ChatMessage.role == "user",
                )
            ).first()
            if row is None:
                return None
            message, conversation = row
            return message_to_dict(message), conversation_to_dict(conversation)

    def get_reply(self, user_message_id: str) -> dict | None:
        with self.session_factory() as session:
            reply = session.scalar(
                select(ChatMessage).where(
                    ChatMessage.reply_to_message_id == uuid.UUID(user_message_id)
                )
            )
            return message_to_dict(reply) if reply else None

    def append_assistant_message(
        self,
        conversation_id: str,
        user_id: str,
        content: str,
        tools_used: list[str],
        reply_to_message_id: str,
    ) -> dict:
        identifier = uuid.UUID(conversation_id)
        reply_identifier = uuid.UUID(reply_to_message_id)
        with self.session_factory.begin() as session:
            conversation = session.scalar(
                select(Conversation)
                .where(
                    Conversation.id == identifier,
                    Conversation.user_id == user_id,
                )
                .with_for_update()
            )
            if conversation is None:
                raise ConversationNotFoundError("Conversation not found")
            existing = session.scalar(
                select(ChatMessage).where(
                    ChatMessage.reply_to_message_id == reply_identifier
                )
            )
            if existing is not None:
                return message_to_dict(existing)
            next_ordinal = (
                session.scalar(
                    select(func.coalesce(func.max(ChatMessage.ordinal), 0)).where(
                        ChatMessage.conversation_id == identifier
                    )
                )
                + 1
            )
            message = ChatMessage(
                conversation_id=identifier,
                ordinal=next_ordinal,
                role="assistant",
                content=content,
                tools_used=tools_used,
                reply_to_message_id=reply_identifier,
            )
            conversation.updated_at = utc_now()
            session.add(message)
            session.flush()
            result = message_to_dict(message)
        return result

    def get_last_artifact(self, conversation_id: str, user_id: str) -> dict | None:
        conversation = self.get_conversation(conversation_id, user_id)
        with self.session_factory() as session:
            state = session.get(ConversationState, uuid.UUID(conversation["id"]))
            return dict(state.last_artifact) if state and state.last_artifact else None

    def save_last_artifact(
        self, conversation_id: str, user_id: str, artifact: dict | None
    ) -> None:
        conversation = self.get_conversation(conversation_id, user_id)
        identifier = uuid.UUID(conversation["id"])
        with self.session_factory.begin() as session:
            state = session.get(ConversationState, identifier)
            if state is None:
                state = ConversationState(conversation_id=identifier)
                session.add(state)
            elif state.last_artifact == artifact:
                return
            state.last_artifact = artifact
            state.updated_at = utc_now()


engine_options = {"pool_pre_ping": True}
if DATABASE_URL.startswith(("postgresql://", "postgresql+psycopg://")):
    engine_options["connect_args"] = {"connect_timeout": 5}
engine = create_engine(DATABASE_URL, **engine_options)
SessionLocal = sessionmaker(bind=engine, expire_on_commit=False)
conversation_repository = ConversationRepository(SessionLocal)
