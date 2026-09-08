"""Benchmark indexed hybrid recall at increasing per-user corpus sizes."""

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import math
import time
import uuid

from qdrant_client.http import models

from services import vectorstore


def _percentile(values: list[float], percentile: int) -> float:
    ordered = sorted(values)
    index = max(0, math.ceil(len(ordered) * percentile / 100) - 1)
    return ordered[index]


def _dense_vector(index: int) -> list[float]:
    vector = [0.0] * vectorstore.EMBEDDING_DIM
    vector[index % min(vectorstore.EMBEDDING_DIM, 32)] = 1.0
    return vector


def _create_collection(client, collection_name: str) -> None:
    client.create_collection(
        collection_name=collection_name,
        vectors_config=models.VectorParams(
            size=vectorstore.EMBEDDING_DIM,
            distance=models.Distance.COSINE,
        ),
        sparse_vectors_config={
            vectorstore.BM25_VECTOR_NAME: models.SparseVectorParams(
                modifier=models.Modifier.IDF,
            )
        },
    )
    for field_name, field_schema in vectorstore._PAYLOAD_INDEXES.items():
        client.create_payload_index(
            collection_name=collection_name,
            field_name=field_name,
            field_schema=field_schema,
            wait=True,
        )


def _insert_until(client, collection_name: str, start: int, stop: int) -> None:
    batch_size = 128
    for batch_start in range(start, stop, batch_size):
        batch_stop = min(stop, batch_start + batch_size)
        points = []
        for index in range(batch_start, batch_stop):
            text = (
                f"Synthetic memory chunk {index} with exact token needle_{index} "
                "and representative retrieval benchmark content."
            )
            points.append(
                models.PointStruct(
                    id=index,
                    vector={
                        "": _dense_vector(index),
                        vectorstore.BM25_VECTOR_NAME: vectorstore._bm25_document(text),
                    },
                    payload={
                        "text": text,
                        "metadata": {
                            "user_id": "benchmark-user",
                            "category": "document",
                            "active": True,
                            "source_id": f"benchmark-{index // 10}",
                        },
                    },
                )
            )
        client.upsert(collection_name, points=points, wait=True)


def _query(client, collection_name: str, target: int, top_k: int) -> None:
    query_filter = vectorstore._payload_filter("benchmark-user")
    candidate_k = max(20, top_k * 4)
    result = client.query_points(
        collection_name=collection_name,
        prefetch=[
            models.Prefetch(
                query=_dense_vector(target),
                filter=query_filter,
                limit=candidate_k,
            ),
            models.Prefetch(
                query=vectorstore._bm25_query(f"needle_{target}"),
                using=vectorstore.BM25_VECTOR_NAME,
                filter=query_filter,
                params=models.SearchParams(
                    idf=models.IdfCorpusParams(corpus=query_filter)
                ),
                limit=candidate_k,
            ),
        ],
        query=models.RrfQuery(rrf=models.Rrf(k=60)),
        limit=top_k,
        with_payload=True,
        with_vectors=False,
    )
    if target not in {point.id for point in result.points}:
        raise RuntimeError(
            f"Hybrid recall missed exact lexical target {target} at top_k={top_k}"
        )


def _measure(
    client,
    collection_name: str,
    size: int,
    samples: int,
    concurrency: int,
    top_k: int,
) -> dict:
    targets = [((index * 7_919) % size) for index in range(samples)]
    for target in targets[:min(20, samples)]:
        _query(client, collection_name, target, top_k)

    def timed_query(target: int) -> float:
        started = time.perf_counter()
        _query(client, collection_name, target, top_k)
        return (time.perf_counter() - started) * 1_000

    started = time.perf_counter()
    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        latencies = list(executor.map(timed_query, targets))
    elapsed = time.perf_counter() - started
    return {
        "chunks_per_user": size,
        "samples": samples,
        "concurrency": concurrency,
        "p50_ms": round(_percentile(latencies, 50), 3),
        "p95_ms": round(_percentile(latencies, 95), 3),
        "p99_ms": round(_percentile(latencies, 99), 3),
        "qps": round(samples / elapsed, 2),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sizes", default="1000,10000,100000")
    parser.add_argument("--samples", type=int, default=200)
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--keep", action="store_true")
    args = parser.parse_args()
    sizes = sorted({int(value) for value in args.sizes.split(",")})
    if not sizes or sizes[0] < 1:
        raise SystemExit("All benchmark sizes must be positive")
    if args.samples < 1 or args.concurrency < 1:
        raise SystemExit("samples and concurrency must be positive")

    client = vectorstore._get_client()
    prefix = f"{vectorstore.MEMORY_COLLECTION}_benchmark_"
    collection_name = f"{prefix}{uuid.uuid4().hex[:10]}"
    current_size = 0
    print(json.dumps({"collection": collection_name}))
    try:
        _create_collection(client, collection_name)
        for size in sizes:
            _insert_until(client, collection_name, current_size, size)
            current_size = size
            result = _measure(
                client,
                collection_name,
                size,
                args.samples,
                args.concurrency,
                args.top_k,
            )
            print(json.dumps(result, sort_keys=True))
    finally:
        if not args.keep:
            if not collection_name.startswith(prefix):
                raise RuntimeError("Refusing to delete an unexpected collection")
            if client.collection_exists(collection_name):
                client.delete_collection(collection_name)


if __name__ == "__main__":
    main()
