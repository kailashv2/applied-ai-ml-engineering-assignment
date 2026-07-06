from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass

TOKEN_RE = re.compile(r"\S+")


@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    text: str
    metadata: dict


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode('utf-8')).hexdigest()


def split_tokens(text: str, chunk_size: int = 900, overlap: int = 120) -> list[str]:
    if chunk_size <= 0:
        raise ValueError('chunk_size must be positive')
    if overlap >= chunk_size:
        raise ValueError('overlap must be smaller than chunk_size')

    tokens = TOKEN_RE.findall(text)
    chunks = []
    step = chunk_size - overlap
    for start in range(0, len(tokens), step):
        window = tokens[start:start + chunk_size]
        if not window:
            continue
        chunks.append(' '.join(window))
        if start + chunk_size >= len(tokens):
            break
    return chunks


def make_chunks(text: str, base_metadata: dict, chunk_size: int = 900, overlap: int = 120) -> list[Chunk]:
    source_hash = sha256_text(text)
    source_path = base_metadata.get('source_path', 'unknown')
    chunks = []
    for idx, chunk_text in enumerate(split_tokens(text, chunk_size, overlap)):
        chunk_hash = sha256_text(chunk_text)
        chunk_id = sha256_text(f'{source_path}:{source_hash}:{idx}:{chunk_hash}')[:32]
        metadata = dict(base_metadata)
        metadata.update({
            'source_hash': source_hash,
            'chunk_index': idx,
            'chunk_hash': chunk_hash,
            'chunk_size': chunk_size,
            'chunk_overlap': overlap,
        })
        chunks.append(Chunk(chunk_id=chunk_id, text=chunk_text, metadata=metadata))
    return chunks
