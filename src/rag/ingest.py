from __future__ import annotations

import argparse
from rich.console import Console
from .chunking import make_chunks
from .config import settings
from .embeddings import embed_texts
from .loaders import iter_files, load_text
from .store import connect_table, upsert_source_chunks, vector_count

console = Console()


def ingest(input_path: str, db_uri: str, table_name: str, chunk_size: int, chunk_overlap: int) -> dict:
    table = connect_table(db_uri, table_name)
    before = vector_count(table)
    files_seen = 0
    chunks_written = 0

    for path in iter_files(input_path):
        text, metadata = load_text(path)
        if not text.strip():
            console.print(f'[yellow]Skipping empty file:[/yellow] {path}')
            continue
        chunks = make_chunks(text, metadata, chunk_size, chunk_overlap)
        vectors = embed_texts([c.text for c in chunks], settings.embedding_model)
        written = upsert_source_chunks(table, chunks, vectors)
        files_seen += 1
        chunks_written += written
        console.print(f'[green]Ingested[/green] {path} -> {written} chunks')

    after = vector_count(table)
    result = {
        'files_seen': files_seen,
        'chunks_written_current_run': chunks_written,
        'vector_count_before': before,
        'vector_count_after': after,
    }
    console.print(result)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True)
    parser.add_argument('--db-uri', default=settings.vector_db_uri)
    parser.add_argument('--table', default=settings.vector_table)
    parser.add_argument('--chunk-size', type=int, default=900)
    parser.add_argument('--chunk-overlap', type=int, default=120)
    args = parser.parse_args()
    ingest(args.input, args.db_uri, args.table, args.chunk_size, args.chunk_overlap)


if __name__ == '__main__':
    main()
