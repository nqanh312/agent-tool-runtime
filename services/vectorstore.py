"""Persist and retrieve the agent's long-term memory in Qdrant."""

import uuid
from config import QDRANT_HOST, QDRANT_PORT, MEMORY_COLLECTION, EMBEDDING_DIM


def ensure_collection():
    """Ensure the Qdrant collection exists."""
    # TODO: Create a cosine-distance collection when it does not exist.
    pass


def save_memory(text: str, embedding: list[float], metadata: dict = None):
    """Store one text, its vector, and metadata as a Qdrant point."""
    # TODO: Ensure the collection exists, then upsert a UUID-backed point.
    pass


def search_memory(query_vector: list[float], top_k: int = 5) -> list[dict]:
    """Return the memories most similar to a query vector."""
    # TODO: Query the collection and normalize text, score, and metadata.
    pass


def list_all_memories(limit: int = 100) -> list[dict]:
    """Return stored memories without semantic ranking."""
    # TODO: Scroll the collection and normalize each point for callers.
    pass
