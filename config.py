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
MEMORY_EXTRACTION_MODEL = os.getenv("MEMORY_EXTRACTION_MODEL", FCI_MODEL).strip()
TURN_PLANNER_MODEL = os.getenv(
    "TURN_PLANNER_MODEL",
    MEMORY_EXTRACTION_MODEL,
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
QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", "6333"))
QDRANT_URL = os.getenv("QDRANT_URL", "").strip()
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY", "").strip()
MEMORY_COLLECTION = os.getenv("MEMORY_COLLECTION", "agent_memory")

# Durable conversation history
DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql+psycopg://agent:agent@localhost:5432/agent_db",
).strip()
CHAT_CONTEXT_MAX_TOKENS = int(os.getenv("CHAT_CONTEXT_MAX_TOKENS", "12000"))
SERVICE_API_KEY = os.getenv("SERVICE_API_KEY", "sk-admin-001").strip()

# Google Drive configuration
GOOGLE_SERVICE_ACCOUNT_FILE = os.getenv("GOOGLE_SERVICE_ACCOUNT_FILE", "credentials.json")
GOOGLE_DRIVE_FOLDER_ID = os.getenv("GOOGLE_DRIVE_FOLDER_ID", "")
