"""Load service credentials and runtime settings from the environment."""

import os
from dotenv import load_dotenv

load_dotenv()

# LLM configuration
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "anthropic").strip().lower()
LLM_MODEL = os.getenv("LLM_MODEL", "").strip()

ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
ANTHROPIC_MODEL = os.getenv("ANTHROPIC_MODEL", "claude-sonnet-4-20250514")

OPENAI_LLM_MODEL = os.getenv("OPENAI_LLM_MODEL", "gpt-4.1-mini")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "").strip()

FCI_API_KEY = os.getenv("FCI_API_KEY") or os.getenv("FPT_API_KEY")
FCI_MODEL = os.getenv("FCI_MODEL", "gemma-4-31B-it")
TURN_PLANNER_MODEL = os.getenv(
    "TURN_PLANNER_MODEL",
    FCI_MODEL,
).strip()
MEMORY_EXTRACTION_MIN_CONFIDENCE = float(
    os.getenv("MEMORY_EXTRACTION_MIN_CONFIDENCE", "0.7")
)
MEMORY_RELEVANCE_MIN_SEMANTIC_SCORE = float(
    os.getenv("MEMORY_RELEVANCE_MIN_SEMANTIC_SCORE", "0.7")
)
FCI_BASE_URL = os.getenv(
    "FCI_BASE_URL",
    "https://mkp-api.fptcloud.com/v1",
).rstrip("/")

# Embedding configuration
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
EMBEDDING_PROVIDER = os.getenv("EMBEDDING_PROVIDER", "openai").strip().lower()
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
EMBEDDING_DIM = int(os.getenv("EMBEDDING_DIM", "1536"))
MEMORY_CHUNK_TOKENS = int(os.getenv("MEMORY_CHUNK_TOKENS", "500"))
MEMORY_CHUNK_OVERLAP_TOKENS = int(
    os.getenv("MEMORY_CHUNK_OVERLAP_TOKENS", "75")
)

# Vector store configuration
QDRANT_HOST = os.getenv("QDRANT_HOST", "127.0.0.1")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", "6333"))
QDRANT_URL = os.getenv("QDRANT_URL", "").strip()
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY", "").strip()
MEMORY_COLLECTION = os.getenv("MEMORY_COLLECTION", "agent_memory")
MEMORY_BM25_AVG_LEN = float(os.getenv("MEMORY_BM25_AVG_LEN", "256"))

# Durable conversation history
DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql+psycopg://agent:agent@localhost:5432/agent_db",
).strip()
CHAT_CONTEXT_MAX_TOKENS = int(os.getenv("CHAT_CONTEXT_MAX_TOKENS", "12000"))
CHAT_MESSAGE_MAX_CHARS = int(os.getenv("CHAT_MESSAGE_MAX_CHARS", "12000"))
MAX_REQUEST_BODY_BYTES = int(os.getenv("MAX_REQUEST_BODY_BYTES", "65536"))
CHAT_RATE_LIMIT_REQUESTS = int(os.getenv("CHAT_RATE_LIMIT_REQUESTS", "10"))
CHAT_RATE_LIMIT_WINDOW_SECONDS = int(
    os.getenv("CHAT_RATE_LIMIT_WINDOW_SECONDS", "60")
)
CHAT_TOKEN_QUOTA_PER_DAY = int(
    os.getenv("CHAT_TOKEN_QUOTA_PER_DAY", "250000")
)
# Charge a small fixed amount in addition to input tokens so repeated tiny
# prompts cannot bypass the daily anti-abuse quota.
CHAT_TOKEN_BASE_CHARGE = int(os.getenv("CHAT_TOKEN_BASE_CHARGE", "1024"))
AGENT_SESSION_TTL_SECONDS = int(
    os.getenv("AGENT_SESSION_TTL_SECONDS", "1800")
)
AGENT_SESSION_CACHE_MAX = int(os.getenv("AGENT_SESSION_CACHE_MAX", "500"))
CONVERSATION_LOCK_TTL_SECONDS = int(
    os.getenv("CONVERSATION_LOCK_TTL_SECONDS", "300")
)
CONVERSATION_LOCK_CACHE_MAX = int(
    os.getenv("CONVERSATION_LOCK_CACHE_MAX", "2000")
)
JWT_SECRET = os.getenv("JWT_SECRET", "").strip()
JWT_ACCESS_MINUTES = int(os.getenv("JWT_ACCESS_MINUTES", "15"))
JWT_REFRESH_DAYS = int(os.getenv("JWT_REFRESH_DAYS", "7"))
JWT_PASSWORD_CHANGE_MINUTES = int(
    os.getenv("JWT_PASSWORD_CHANGE_MINUTES", "10")
)
AUTH_COOKIE_SECURE = os.getenv("AUTH_COOKIE_SECURE", "false").strip().lower() in {
    "1", "true", "yes", "on",
}
CORS_ORIGINS = [
    value.strip()
    for value in os.getenv("CORS_ORIGINS", "http://localhost:9004").split(",")
    if value.strip()
]
TRUSTED_PROXY_IPS = {
    value.strip()
    for value in os.getenv("TRUSTED_PROXY_IPS", "").split(",")
    if value.strip()
}
SERVER_HOST = os.getenv("SERVER_HOST", "127.0.0.1").strip()
SERVER_PORT = int(os.getenv("SERVER_PORT", "9004"))
CLI_USER_ID = os.getenv("CLI_USER_ID", "user_admin").strip()

# Per-user Google OAuth. Authentication and Drive consent are intentionally
# separate so users can use the application without granting Drive access.
GOOGLE_OAUTH_CLIENT_ID = os.getenv("GOOGLE_OAUTH_CLIENT_ID", "").strip()
GOOGLE_OAUTH_CLIENT_SECRET = os.getenv("GOOGLE_OAUTH_CLIENT_SECRET", "").strip()
GOOGLE_OAUTH_REDIRECT_URI = os.getenv(
    "GOOGLE_OAUTH_REDIRECT_URI",
    "http://localhost:9004/api/auth/google/callback",
).strip()
GOOGLE_OAUTH_DRIVE_SCOPES = tuple(
    value.strip()
    for value in os.getenv(
        "GOOGLE_OAUTH_DRIVE_SCOPES",
        "https://www.googleapis.com/auth/drive.readonly",
    ).split(",")
    if value.strip()
)
GOOGLE_OAUTH_ALLOWED_DOMAIN = os.getenv(
    "GOOGLE_OAUTH_ALLOWED_DOMAIN", ""
).strip().casefold()
GOOGLE_TOKEN_ENCRYPTION_KEY = os.getenv(
    "GOOGLE_TOKEN_ENCRYPTION_KEY", ""
).strip()
GOOGLE_OAUTH_STATE_MINUTES = int(os.getenv("GOOGLE_OAUTH_STATE_MINUTES", "10"))
