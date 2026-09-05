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
FCI_BASE_URL = os.getenv(
    "FCI_BASE_URL",
    "https://mkp-api.fptcloud.com/v1",
).rstrip("/")

# Embedding configuration
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
EMBEDDING_DIM = 1536

# Vector store configuration
QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", "6333"))
MEMORY_COLLECTION = "agent_memory"

# Google Drive configuration
GOOGLE_SERVICE_ACCOUNT_FILE = os.getenv("GOOGLE_SERVICE_ACCOUNT_FILE", "credentials.json")
GOOGLE_DRIVE_FOLDER_ID = os.getenv("GOOGLE_DRIVE_FOLDER_ID", "")
