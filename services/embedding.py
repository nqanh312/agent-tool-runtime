"""Create vector embeddings for memory storage and retrieval."""

from config import OPENAI_API_KEY, EMBEDDING_MODEL


def embed_texts(texts: list[str]) -> list[list[float]]:
    """Convert a batch of texts into embedding vectors."""
    # TODO: Request embeddings with OPENAI_API_KEY and EMBEDDING_MODEL.
    pass


def embed_query(query: str) -> list[float]:
    """Convert one search query into an embedding vector."""
    # TODO: Reuse embed_texts and return the first vector.
    pass
