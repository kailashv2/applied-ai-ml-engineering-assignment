from __future__ import annotations

from typing import Iterable
import lancedb
import pandas as pd
from .chunking import Chunk


def connect_table(db_uri: str, table_name: str):
    db = lancedb.connect(db_uri)
    if table_name in db.table_names():
        return db.open_table(table_name)
    seed = [{
        'id': '__seed__',
        'vector': [0.0] * 384,
        'text': '',
        'source_path': '__seed__',
        'source_name': '__seed__',
        'source_hash': '__seed__',
        'chunk_hash': '__seed__',
        'chunk_index': -1,
        'doc_type': 'seed',
        'chunk_size': 0,
        'chunk_overlap': 0,
    }]
    table = db.create_table(table_name, data=seed)
    table.delete("id = '__seed__'")
    return table


def records_from_chunks(chunks: list[Chunk], vectors) -> list[dict]:
    records = []
    for chunk, vector in zip(chunks, vectors):
        record = {
            'id': chunk.chunk_id,
            'vector': vector.tolist(),
            'text': chunk.text,
        }
        record.update(chunk.metadata)
        records.append(record)
    return records


def upsert_source_chunks(table, chunks: list[Chunk], vectors) -> int:
    if not chunks:
        return 0
    # Idempotent by source path: replace every chunk for this source, then add current chunks.
    source_path = chunks[0].metadata['source_path'].replace("'", "\\'")
    table.delete(f"source_path = '{source_path}'")
    table.add(records_from_chunks(chunks, vectors))
    return len(chunks)


def vector_count(table) -> int:
    return int(table.count_rows())


def search(table, query_vector, k: int = 5, metadata_filter: dict | None = None) -> pd.DataFrame:
    q = table.search(query_vector).limit(k)
    if metadata_filter:
        clauses = []
        for key, value in metadata_filter.items():
            if isinstance(value, str):
                clauses.append(f"{key} = '{value}'")
            else:
                clauses.append(f"{key} = {value}")
        q = q.where(' AND '.join(clauses))
    return q.to_pandas()
