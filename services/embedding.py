"""Create vector embeddings for memory storage and retrieval."""

from functools import lru_cache

from openai import OpenAI

from config import (
    EMBEDDING_DIM,
    EMBEDDING_MODEL,
    EMBEDDING_PROVIDER,
    FCI_API_KEY,
    FCI_BASE_URL,
    OPENAI_API_KEY,
)


@lru_cache(maxsize=1)
def _get_client() -> OpenAI:
    if EMBEDDING_PROVIDER == "openai":
        if not OPENAI_API_KEY or OPENAI_API_KEY.startswith("your_"):
            raise ValueError(
                "A valid OPENAI_API_KEY is required when "
                "EMBEDDING_PROVIDER=openai"
            )
        return OpenAI(api_key=OPENAI_API_KEY)

    if EMBEDDING_PROVIDER in {"fci", "fpt"}:
        if not FCI_API_KEY:
            raise ValueError(
                "FCI_API_KEY is required when EMBEDDING_PROVIDER=fci"
            )
        return OpenAI(api_key=FCI_API_KEY, base_url=FCI_BASE_URL)

    raise ValueError(
        f"Unsupported EMBEDDING_PROVIDER '{EMBEDDING_PROVIDER}'. "
        "Choose openai or fci."
    )


def _prepare_inputs(texts: list[str], *, query: bool = False) -> list[str]:
    """Apply retrieval prefixes expected by E5-family embedding models."""
    if "e5" not in EMBEDDING_MODEL.casefold():
        return texts
    prefix = "query: " if query else "passage: "
    return [text if text.casefold().startswith(prefix) else prefix + text for text in texts]


def _request_embeddings(texts: list[str], *, query: bool = False) -> list[list[float]]:
    response = _get_client().embeddings.create(
        model=EMBEDDING_MODEL,
        input=_prepare_inputs(texts, query=query),
    )
    ordered = sorted(response.data, key=lambda item: item.index)
    vectors = [list(item.embedding) for item in ordered]
    if len(vectors) != len(texts):
        raise RuntimeError("Embedding provider returned an unexpected result count")
    if any(len(vector) != EMBEDDING_DIM for vector in vectors):
        actual = len(vectors[0]) if vectors else 0
        raise ValueError(
            f"Embedding dimension mismatch: configured {EMBEDDING_DIM}, got {actual}. "
            "Set EMBEDDING_DIM correctly and use a new MEMORY_COLLECTION."
        )
    return vectors


def embed_texts(texts: list[str]) -> list[list[float]]:
    """Convert a batch of texts into embedding vectors."""
    if not texts:
        return []
    if any(not isinstance(text, str) or not text.strip() for text in texts):
        raise ValueError("Every embedding input must be a non-empty string")

    return _request_embeddings(texts, query=False)


def embed_query(query: str) -> list[float]:
    """Convert one search query into an embedding vector."""
    if not isinstance(query, str) or not query.strip():
        raise ValueError("Search query must be a non-empty string")
    return _request_embeddings([query], query=True)[0]
