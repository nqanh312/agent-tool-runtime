"""Backfill BM25 vectors and normalize legacy memory payloads.

Run this once after deploying the hybrid vectorstore code. The migration is
idempotent: reruns only process points that still lack the BM25 vector.
"""

import argparse

from qdrant_client.http import models

from services import vectorstore


def _structured_preference_users(client, batch_size: int) -> set[str]:
    users: set[str] = set()
    offset = None
    preference_filter = models.Filter(
        must=[
            models.FieldCondition(
                key="metadata.memory_type",
                match=models.MatchValue(value="preference"),
            )
        ],
        must_not=[
            models.FieldCondition(
                key="metadata.active",
                match=models.MatchValue(value=False),
            )
        ],
    )
    while True:
        points, offset = client.scroll(
            collection_name=vectorstore.MEMORY_COLLECTION,
            scroll_filter=preference_filter,
            limit=batch_size,
            offset=offset,
            with_payload=True,
            with_vectors=False,
        )
        for point in points:
            user_id = (point.payload or {}).get("metadata", {}).get("user_id")
            if user_id:
                users.add(str(user_id))
        if offset is None or not points:
            return users


def _backfill_active(client) -> None:
    client.set_payload(
        collection_name=vectorstore.MEMORY_COLLECTION,
        payload={"active": True},
        key="metadata",
        points=models.Filter(
            must=[
                models.IsEmptyCondition(
                    is_empty=models.PayloadField(key="metadata.active")
                )
            ]
        ),
        wait=True,
    )


def _hide_legacy_preferences(client, user_ids: set[str], batch_size: int) -> None:
    ordered = sorted(user_ids)
    for start in range(0, len(ordered), batch_size):
        user_batch = ordered[start:start + batch_size]
        client.set_payload(
            collection_name=vectorstore.MEMORY_COLLECTION,
            payload={"active": False, "legacy_hidden": True},
            key="metadata",
            points=models.Filter(
                must=[
                    models.FieldCondition(
                        key="metadata.user_id",
                        match=models.MatchAny(any=user_batch),
                    ),
                    models.FieldCondition(
                        key="metadata.category",
                        match=models.MatchValue(value="user_preference"),
                    ),
                ],
                must_not=[
                    models.FieldCondition(
                        key="metadata.memory_type",
                        match=models.MatchValue(value="preference"),
                    )
                ],
            ),
            wait=True,
        )


def _missing_sparse_filter() -> models.Filter:
    return models.Filter(
        must_not=[
            models.FieldCondition(
                key="metadata.active",
                match=models.MatchValue(value=False),
            ),
            models.HasVectorCondition(
                has_vector=vectorstore.BM25_VECTOR_NAME,
            ),
        ]
    )


def _backfill_sparse_vectors(client, batch_size: int) -> int:
    updated = 0
    offset = None
    while True:
        points, offset = client.scroll(
            collection_name=vectorstore.MEMORY_COLLECTION,
            scroll_filter=_missing_sparse_filter(),
            limit=batch_size,
            offset=offset,
            with_payload=True,
            with_vectors=False,
        )
        if not points:
            break
        client.update_vectors(
            collection_name=vectorstore.MEMORY_COLLECTION,
            points=[
                models.PointVectors(
                    id=point.id,
                    vector={
                        vectorstore.BM25_VECTOR_NAME: vectorstore._bm25_document(
                            (point.payload or {}).get("text", "")
                        )
                    },
                )
                for point in points
            ],
            wait=True,
        )
        updated += len(points)
        print(f"Backfilled {updated} BM25 vectors...")
        if offset is None:
            break
    return updated


def migrate(batch_size: int = 256, *, dry_run: bool = False) -> dict:
    if batch_size < 1 or batch_size > 1_000:
        raise ValueError("batch_size must be between 1 and 1000")

    client = vectorstore._get_client()
    if not client.collection_exists(vectorstore.MEMORY_COLLECTION):
        if dry_run:
            return {"points": 0, "structured_users": 0, "missing_sparse": 0}
        vectorstore.ensure_collection()
        return {"points": 0, "structured_users": 0, "missing_sparse": 0}

    if not dry_run:
        client = vectorstore.ensure_collection()
        _backfill_active(client)

    users = _structured_preference_users(client, batch_size)
    total = client.count(
        collection_name=vectorstore.MEMORY_COLLECTION,
        exact=True,
    ).count
    collection = client.get_collection(vectorstore.MEMORY_COLLECTION)
    has_sparse_schema = vectorstore.BM25_VECTOR_NAME in (
        collection.config.params.sparse_vectors or {}
    )
    missing_sparse = (
        client.count(
            collection_name=vectorstore.MEMORY_COLLECTION,
            count_filter=_missing_sparse_filter(),
            exact=True,
        ).count
        if has_sparse_schema
        else total
    )
    summary = {
        "points": total,
        "structured_users": len(users),
        "missing_sparse": missing_sparse,
    }
    if dry_run:
        return summary

    _hide_legacy_preferences(client, users, batch_size)
    summary["backfilled_sparse"] = _backfill_sparse_vectors(client, batch_size)
    summary["remaining_sparse"] = client.count(
        collection_name=vectorstore.MEMORY_COLLECTION,
        count_filter=_missing_sparse_filter(),
        exact=True,
    ).count
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Migrate Qdrant memory to indexed dense/BM25 hybrid search."
    )
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    summary = migrate(args.batch_size, dry_run=args.dry_run)
    print(summary)


if __name__ == "__main__":
    main()
