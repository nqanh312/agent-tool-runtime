"""Persist and retrieve the agent's long-term memory in Qdrant."""

from collections import Counter
from functools import lru_cache
import math
import re
import threading
import uuid
import weakref
import zlib

from qdrant_client import QdrantClient
from qdrant_client.http import models

from config import (
    EMBEDDING_DIM,
    MEMORY_BM25_AVG_LEN,
    MEMORY_COLLECTION,
    QDRANT_API_KEY,
    QDRANT_HOST,
    QDRANT_PORT,
    QDRANT_URL,
)


BM25_VECTOR_NAME = "bm25"
_collection_lock = threading.Lock()
_ready_clients: weakref.WeakKeyDictionary = weakref.WeakKeyDictionary()
_TOKEN_RE = re.compile(r"\w+", re.UNICODE)
_BM25_K = 1.5
_BM25_B = 0.75
if MEMORY_BM25_AVG_LEN <= 0:
    raise ValueError("MEMORY_BM25_AVG_LEN must be positive")
_PAYLOAD_INDEXES = {
    "metadata.user_id": models.KeywordIndexParams(
        type=models.KeywordIndexType.KEYWORD,
        is_tenant=True,
    ),
    "metadata.category": models.PayloadSchemaType.KEYWORD,
    "metadata.active": models.PayloadSchemaType.BOOL,
    "metadata.memory_type": models.PayloadSchemaType.KEYWORD,
    "metadata.source_id": models.PayloadSchemaType.KEYWORD,
}


@lru_cache(maxsize=1)
def _get_client() -> QdrantClient:
    options = {"api_key": QDRANT_API_KEY or None, "timeout": 15}
    if QDRANT_URL:
        return QdrantClient(url=QDRANT_URL, **options)
    return QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT, **options)


def _payload_filter(
    user_id: str | None,
    categories: set[str] | None = None,
    *,
    current_only: bool = True,
) -> models.Filter | None:
    must = []
    must_not = []
    if user_id:
        must.append(
            models.FieldCondition(
                key="metadata.user_id",
                match=models.MatchValue(value=user_id),
            )
        )
    if categories:
        must.append(
            models.FieldCondition(
                key="metadata.category",
                match=models.MatchAny(any=sorted(categories)),
            )
        )
    if current_only:
        # Missing `active` is treated as true during a rolling migration.
        must_not.append(
            models.FieldCondition(
                key="metadata.active",
                match=models.MatchValue(value=False),
            )
        )
    if not must and not must_not:
        return None
    return models.Filter(must=must or None, must_not=must_not or None)


def _tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.casefold())


def _token_id(token: str) -> int:
    """Map a Unicode token to a stable unsigned sparse-vector dimension."""
    return zlib.crc32(token.encode("utf-8"))


def _bm25_document(text: str) -> models.SparseVector:
    """Encode the document-side term-frequency component of BM25."""
    tokens = _tokenize(text)
    frequencies = Counter(tokens)
    document_length = len(tokens)
    denominator_length = 1 - _BM25_B + (
        _BM25_B * document_length / MEMORY_BM25_AVG_LEN
    )
    encoded = {}
    for token, frequency in frequencies.items():
        token_id = _token_id(token)
        value = frequency * (_BM25_K + 1)
        value /= frequency + _BM25_K * denominator_length
        # Hash collisions are rare; summing is deterministic and prevents one
        # colliding term from silently replacing another.
        encoded[token_id] = encoded.get(token_id, 0.0) + value
    indices = sorted(encoded)
    return models.SparseVector(
        indices=indices,
        values=[encoded[index] for index in indices],
    )


def _bm25_query(text: str) -> models.SparseVector:
    """Encode a BM25 query; Qdrant supplies tenant-scoped IDF weights."""
    indices = sorted({_token_id(token) for token in _tokenize(text)})
    return models.SparseVector(
        indices=indices,
        values=[1.0] * len(indices),
    )


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


def _validate_collection(info) -> None:
    dense = info.config.params.vectors
    if not isinstance(dense, models.VectorParams) or dense.size != EMBEDDING_DIM:
        raise RuntimeError(
            f"Collection {MEMORY_COLLECTION!r} has an incompatible dense-vector "
            f"schema; expected one unnamed vector with size {EMBEDDING_DIM}."
        )
    sparse = info.config.params.sparse_vectors or {}
    bm25 = sparse.get(BM25_VECTOR_NAME)
    if bm25 is not None and bm25.modifier != models.Modifier.IDF:
        raise RuntimeError(
            f"Sparse vector {BM25_VECTOR_NAME!r} must use the IDF modifier."
        )


def ensure_collection() -> QdrantClient:
    """Ensure the hybrid collection schema and payload indexes exist once."""
    client = _get_client()
    with _collection_lock:
        if _ready_clients.get(client):
            return client

        if not client.collection_exists(MEMORY_COLLECTION):
            client.create_collection(
                collection_name=MEMORY_COLLECTION,
                vectors_config=models.VectorParams(
                    size=EMBEDDING_DIM,
                    distance=models.Distance.COSINE,
                ),
                sparse_vectors_config={
                    BM25_VECTOR_NAME: models.SparseVectorParams(
                        modifier=models.Modifier.IDF,
                    )
                },
            )

        info = client.get_collection(MEMORY_COLLECTION)
        _validate_collection(info)
        sparse = info.config.params.sparse_vectors or {}
        if BM25_VECTOR_NAME not in sparse:
            client.create_vector_name(
                collection_name=MEMORY_COLLECTION,
                vector_name=BM25_VECTOR_NAME,
                vector_name_config=models.SparseVectorNameConfig(
                    sparse=models.SparseVectorConfig(
                        modifier=models.Modifier.IDF,
                    )
                ),
                wait=True,
            )

        existing_indexes = info.payload_schema or {}
        for field_name, field_schema in _PAYLOAD_INDEXES.items():
            if field_name not in existing_indexes:
                client.create_payload_index(
                    collection_name=MEMORY_COLLECTION,
                    field_name=field_name,
                    field_schema=field_schema,
                    wait=True,
                )

        _ready_clients[client] = True
    return client


def save_memory(text: str, embedding: list[float], metadata: dict = None):
    """Store one text, its dense/sparse vectors, and metadata."""
    return save_memories([(text, embedding, metadata or {})])[0]


def save_memories(records: list[tuple[str, list[float], dict]]) -> list[dict]:
    """Store a batch of hybrid memory chunks in one Qdrant upsert."""
    if not records:
        return []
    client = ensure_collection()
    points = []
    saved = []
    for text, vector, supplied_metadata in records:
        if not text.strip():
            raise ValueError("Memory text must not be empty")
        if len(vector) != EMBEDDING_DIM:
            raise ValueError(
                f"Expected a {EMBEDDING_DIM}-dimension vector, got {len(vector)}"
            )
        metadata = dict(supplied_metadata or {})
        metadata.setdefault("active", True)
        memory_key = metadata.get("memory_key")
        if memory_key:
            user_id = metadata.get("user_id", "anonymous")
            chunk_index = metadata.get("chunk_index", 0)
            identity = f"agent-memory:{user_id}:{memory_key}"
            if metadata.get("chunk_count") != 1:
                identity = f"{identity}:{chunk_index}"
            point_id = str(uuid.uuid5(uuid.NAMESPACE_URL, identity))
        else:
            point_id = str(uuid.uuid4())
        payload = {"text": text, "metadata": metadata}
        points.append(
            models.PointStruct(
                id=point_id,
                vector={
                    "": vector,
                    BM25_VECTOR_NAME: _bm25_document(text),
                },
                payload=payload,
            )
        )
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
    """Scroll memories for bounded list/admin operations, never for recall."""
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


def _semantic_score(query_vector: list[float], stored_vector) -> float | None:
    if isinstance(stored_vector, dict):
        stored_vector = stored_vector.get("")
    if not stored_vector or len(stored_vector) != len(query_vector):
        return None
    dot_product = sum(a * b for a, b in zip(query_vector, stored_vector))
    query_norm = math.sqrt(sum(value * value for value in query_vector))
    stored_norm = math.sqrt(sum(value * value for value in stored_vector))
    if not query_norm or not stored_norm:
        return 0.0
    return dot_product / (query_norm * stored_norm)


def _has_lexical_match(query: str, text: str) -> bool:
    return bool(set(_tokenize(query)).intersection(_tokenize(text)))


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
    """Return indexed dense/BM25 results using server-side RRF."""
    if top_k < 1 or top_k > 50:
        raise ValueError("top_k must be between 1 and 50")
    if len(query_vector) != EMBEDDING_DIM:
        raise ValueError(
            f"Expected a {EMBEDDING_DIM}-dimension query vector, got {len(query_vector)}"
        )

    client = ensure_collection()
    query_filter = _payload_filter(user_id)
    candidate_k = max(20, top_k * 4)

    if not query_text.strip():
        points = client.query_points(
            collection_name=MEMORY_COLLECTION,
            query=query_vector,
            query_filter=query_filter,
            limit=top_k,
            with_payload=True,
        ).points
        results = []
        for point in _current_memory_points(points)[:top_k]:
            item = _normalize_point(point, point.score)
            item["semantic_score"] = float(point.score)
            item["lexical_score"] = None
            item["lexical_match"] = False
            results.append(item)
        return results

    sparse_query = _bm25_query(query_text)
    if not sparse_query.indices:
        return search_memory(
            query_vector,
            top_k,
            query_text="",
            user_id=user_id,
        )

    sparse_params = None
    if user_id:
        sparse_params = models.SearchParams(
            idf=models.IdfCorpusParams(
                corpus=_payload_filter(user_id),
            )
        )
    points = client.query_points(
        collection_name=MEMORY_COLLECTION,
        prefetch=[
            models.Prefetch(
                query=query_vector,
                filter=query_filter,
                limit=candidate_k,
            ),
            models.Prefetch(
                query=sparse_query,
                using=BM25_VECTOR_NAME,
                filter=query_filter,
                params=sparse_params,
                limit=candidate_k,
            ),
        ],
        query=models.RrfQuery(rrf=models.Rrf(k=60)),
        limit=min(candidate_k, max(top_k * 2, top_k + 10)),
        with_payload=True,
        with_vectors=[""],
    ).points

    results = []
    for point in _current_memory_points(points)[:top_k]:
        item = _normalize_point(point, point.score)
        item["hybrid_score"] = float(point.score)
        item["semantic_score"] = _semantic_score(query_vector, point.vector)
        item["lexical_score"] = None
        item["lexical_match"] = _has_lexical_match(query_text, item["text"])
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
    return [_normalize_point(point) for point in points]
