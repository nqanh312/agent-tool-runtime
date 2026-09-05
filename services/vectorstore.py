"""Persist and retrieve the agent's long-term memory in Qdrant."""

from collections import Counter
from functools import lru_cache
import math
import re
import threading
import uuid

from qdrant_client import QdrantClient
from qdrant_client.http import models

from config import (
    EMBEDDING_DIM,
    MEMORY_COLLECTION,
    QDRANT_API_KEY,
    QDRANT_HOST,
    QDRANT_PORT,
    QDRANT_URL,
)


_collection_lock = threading.Lock()
_TOKEN_RE = re.compile(r"\w+", re.UNICODE)


@lru_cache(maxsize=1)
def _get_client() -> QdrantClient:
    options = {"api_key": QDRANT_API_KEY or None, "timeout": 15}
    if QDRANT_URL:
        return QdrantClient(url=QDRANT_URL, **options)
    return QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT, **options)


def _payload_filter(
    user_id: str | None,
    categories: set[str] | None = None,
):
    conditions = []
    if user_id:
        conditions.append(
            models.FieldCondition(
                key="metadata.user_id",
                match=models.MatchValue(value=user_id),
            )
        )
    if categories:
        conditions.append(
            models.FieldCondition(
                key="metadata.category",
                match=models.MatchAny(any=sorted(categories)),
            )
        )
    if not conditions:
        return None
    return models.Filter(must=conditions)


def _normalize_point(point, score: float | None = None) -> dict:
    payload = point.payload or {}
    result = {
        "id": str(point.id),
        "text": payload.get("text", ""),
        "metadata": payload.get("metadata", {}),
    }
    if score is not None:
        result["score"] = float(score)
    return result


def ensure_collection():
    """Ensure the Qdrant collection exists."""
    client = _get_client()
    with _collection_lock:
        if not client.collection_exists(MEMORY_COLLECTION):
            client.create_collection(
                collection_name=MEMORY_COLLECTION,
                vectors_config=models.VectorParams(
                    size=EMBEDDING_DIM,
                    distance=models.Distance.COSINE,
                ),
            )
    return client


def save_memory(text: str, embedding: list[float], metadata: dict = None):
    """Store one text, its vector, and metadata as a Qdrant point."""
    return save_memories([(text, embedding, metadata or {})])[0]


def save_memories(records: list[tuple[str, list[float], dict]]) -> list[dict]:
    """Store a batch of memory chunks in one Qdrant upsert."""
    if not records:
        return []
    client = ensure_collection()
    points = []
    saved = []
    for text, vector, metadata in records:
        if not text.strip():
            raise ValueError("Memory text must not be empty")
        if len(vector) != EMBEDDING_DIM:
            raise ValueError(
                f"Expected a {EMBEDDING_DIM}-dimension vector, got {len(vector)}"
            )
        memory_key = (metadata or {}).get("memory_key")
        if memory_key:
            user_id = (metadata or {}).get("user_id", "anonymous")
            chunk_index = (metadata or {}).get("chunk_index", 0)
            identity = f"agent-memory:{user_id}:{memory_key}"
            if (metadata or {}).get("chunk_count") != 1:
                identity = f"{identity}:{chunk_index}"
            point_id = str(
                uuid.uuid5(uuid.NAMESPACE_URL, identity)
            )
        else:
            point_id = str(uuid.uuid4())
        payload = {"text": text, "metadata": metadata or {}}
        points.append(models.PointStruct(id=point_id, vector=vector, payload=payload))
        saved.append({"id": point_id, **payload})

    client.upsert(
        collection_name=MEMORY_COLLECTION,
        points=points,
        wait=True,
    )
    return saved


def _scroll_all(
    user_id: str | None,
    limit: int = 10_000,
    categories: set[str] | None = None,
) -> list:
    client = ensure_collection()
    records = []
    offset = None
    while len(records) < limit:
        page, offset = client.scroll(
            collection_name=MEMORY_COLLECTION,
            scroll_filter=_payload_filter(user_id, categories),
            limit=min(256, limit - len(records)),
            offset=offset,
            with_payload=True,
            with_vectors=False,
        )
        records.extend(page)
        if offset is None or not page:
            break
    return records


def _tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.casefold())


def _bm25_scores(query: str, points: list) -> dict[str, float]:
    """Calculate BM25 over persisted raw text without a second database."""
    query_terms = _tokenize(query)
    documents = [_tokenize((point.payload or {}).get("text", "")) for point in points]
    if not query_terms or not documents:
        return {}

    average_length = sum(map(len, documents)) / len(documents) or 1.0
    document_frequency = Counter(
        term for document in documents for term in set(document)
    )
    scores = {}
    k1, b = 1.5, 0.75
    for point, document in zip(points, documents):
        frequencies = Counter(document)
        score = 0.0
        for term in query_terms:
            frequency = frequencies.get(term, 0)
            if not frequency:
                continue
            containing = document_frequency[term]
            idf = math.log(1 + (len(documents) - containing + 0.5) / (containing + 0.5))
            denominator = frequency + k1 * (
                1 - b + b * len(document) / average_length
            )
            score += idf * frequency * (k1 + 1) / denominator
        if score:
            scores[str(point.id)] = score
    return scores


def _current_memory_points(
    points: list,
    prefer_structured: bool | None = None,
) -> list:
    """Prefer structured current-state preferences over legacy append-only ones."""
    active = [
        point
        for point in points
        if (point.payload or {}).get("metadata", {}).get("active", True)
    ]
    has_structured_preferences = (
        prefer_structured
        if prefer_structured is not None
        else any(
            (point.payload or {}).get("metadata", {}).get("memory_type")
            == "preference"
            for point in active
        )
    )
    if not has_structured_preferences:
        return active
    return [
        point
        for point in active
        if (
            (point.payload or {}).get("metadata", {}).get("category")
            != "user_preference"
            or (point.payload or {}).get("metadata", {}).get("memory_type")
            == "preference"
        )
    ]


def search_memory(
    query_vector: list[float],
    top_k: int = 5,
    *,
    query_text: str = "",
    user_id: str | None = None,
) -> list[dict]:
    """Return hybrid semantic/BM25 memory results using reciprocal-rank fusion."""
    if top_k < 1 or top_k > 50:
        raise ValueError("top_k must be between 1 and 50")
    if len(query_vector) != EMBEDDING_DIM:
        raise ValueError(
            f"Expected a {EMBEDDING_DIM}-dimension query vector, got {len(query_vector)}"
        )

    client = ensure_collection()
    semantic = client.query_points(
        collection_name=MEMORY_COLLECTION,
        query=query_vector,
        query_filter=_payload_filter(user_id),
        limit=max(20, top_k * 4),
        with_payload=True,
    ).points
    corpus = _scroll_all(user_id) if query_text else []
    corpus = _current_memory_points(corpus)
    if any(
        (point.payload or {}).get("metadata", {}).get("memory_type") == "preference"
        for point in corpus
    ):
        semantic = _current_memory_points(semantic, prefer_structured=True)
    lexical_scores = _bm25_scores(query_text, corpus)
    lexical_ids = sorted(lexical_scores, key=lexical_scores.get, reverse=True)

    by_id = {str(point.id): point for point in corpus}
    by_id.update({str(point.id): point for point in semantic})
    fused = Counter()
    for rank, point in enumerate(semantic, start=1):
        fused[str(point.id)] += 1 / (60 + rank)
    for rank, point_id in enumerate(lexical_ids, start=1):
        fused[point_id] += 1 / (60 + rank)

    results = []
    for point_id, score in fused.most_common(top_k):
        item = _normalize_point(by_id[point_id], score)
        item["semantic_score"] = next(
            (float(point.score) for point in semantic if str(point.id) == point_id),
            None,
        )
        item["lexical_score"] = lexical_scores.get(point_id)
        results.append(item)
    return results


def list_all_memories(
    limit: int = 100,
    user_id: str | None = None,
    categories: set[str] | None = None,
) -> list[dict]:
    """Return stored memories without semantic ranking."""
    if limit < 1:
        return []
    points = _current_memory_points(_scroll_all(user_id, limit, categories))
    return [
        _normalize_point(point)
        for point in points
    ]
