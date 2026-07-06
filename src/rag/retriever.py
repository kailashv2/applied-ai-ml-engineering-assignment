from __future__ import annotations

import time
from .config import settings
from .embeddings import embed_texts
from .store import connect_table, search


def retrieve(question: str, k: int | None = None, metadata_filter: dict | None = None) -> dict:
    started = time.perf_counter()
    k = k or settings.default_k
    table = connect_table(settings.vector_db_uri, settings.vector_table)
    query_vector = embed_texts([question], settings.embedding_model)[0]
    df = search(table, query_vector, k=k, metadata_filter=metadata_filter)
    elapsed_ms = (time.perf_counter() - started) * 1000

    chunks = []
    for row in df.to_dict(orient='records'):
        distance = float(row.get('_distance', 999.0))
        relevance = 1.0 / (1.0 + max(distance, 0.0))
        row['relevance_score'] = relevance
        chunks.append(row)

    return {
        'chunks': chunks,
        'latency_ms': elapsed_ms,
        'k': k,
        'metadata_filter': metadata_filter or {},
    }
