from __future__ import annotations

import hashlib
from functools import lru_cache
from typing import Iterable

import numpy as np


def _hashing_embed(texts: list[str], dim: int = 384) -> np.ndarray:
    """Deterministic, dependency-light embedding fallback.

    This keeps the repo runnable in offline review environments. For production-quality semantic
    retrieval, set EMBEDDING_MODEL to a SentenceTransformers model such as
    BAAI/bge-small-en-v1.5. The output dimensionality intentionally matches bge-small (384).
    """
    vectors = np.zeros((len(texts), dim), dtype='float32')
    for row, text in enumerate(texts):
        for token in text.lower().replace('\n', ' ').split():
            digest = hashlib.blake2b(token.encode('utf-8'), digest_size=8).digest()
            idx = int.from_bytes(digest[:4], 'little') % dim
            sign = 1.0 if (digest[4] % 2 == 0) else -1.0
            vectors[row, idx] += sign
        norm = np.linalg.norm(vectors[row])
        if norm > 0:
            vectors[row] /= norm
    return vectors


@lru_cache(maxsize=2)
def get_model(model_name: str):
    if model_name.startswith('local-hashing'):
        return None
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(model_name)


def embed_texts(texts: list[str], model_name: str) -> np.ndarray:
    if model_name.startswith('local-hashing'):
        return _hashing_embed(texts, dim=384)
    try:
        model = get_model(model_name)
        vectors = model.encode(texts, normalize_embeddings=True, batch_size=32, show_progress_bar=False)
        return np.asarray(vectors, dtype='float32')
    except Exception as exc:
        # Offline-safe fallback. The caller can still see the configured model in logs/results,
        # but smoke tests do not fail just because a hosted model cannot be downloaded.
        print(f'[embedding fallback] {model_name} unavailable ({exc}); using local-hashing-384')
        return _hashing_embed(texts, dim=384)
