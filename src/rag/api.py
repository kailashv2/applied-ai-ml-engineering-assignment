from __future__ import annotations

import time
from typing import Any
from fastapi import FastAPI
from pydantic import BaseModel, Field
from .config import settings
from .llm import generate_answer
from .retriever import retrieve

app = FastAPI(title='Cost-efficient RAG API')


class QueryRequest(BaseModel):
    question: str
    k: int = Field(default=5, ge=1, le=20)
    metadata_filter: dict[str, Any] | None = None


@app.get('/health')
def health():
    return {'status': 'ok'}


@app.post('/query')
def query(req: QueryRequest):
    started = time.perf_counter()
    retrieval = retrieve(req.question, k=req.k, metadata_filter=req.metadata_filter)
    chunks = retrieval['chunks']
    usable_chunks = [c for c in chunks if c.get('relevance_score', 0) >= settings.min_relevance_score]

    if not usable_chunks:
        return {
            'answer': "I don't have enough relevant context to answer that.",
            'citations': [],
            'retrieval_latency_ms': round(retrieval['latency_ms'], 2),
            'end_to_end_latency_ms': round((time.perf_counter() - started) * 1000, 2),
            'chunk_count': 0,
            'token_usage': {'prompt_tokens_est': 0, 'completion_tokens_est': 0},
            'no_context': True,
        }

    answer, usage = generate_answer(req.question, usable_chunks, settings.generator_provider, settings.generator_model)
    citations = [
        {
            'chunk_id': c['id'],
            'source_name': c.get('source_name'),
            'chunk_index': c.get('chunk_index'),
            'relevance_score': round(c.get('relevance_score', 0), 4),
        }
        for c in usable_chunks
    ]
    return {
        'answer': answer,
        'citations': citations,
        'retrieval_latency_ms': round(retrieval['latency_ms'], 2),
        'end_to_end_latency_ms': round((time.perf_counter() - started) * 1000, 2),
        'chunk_count': len(usable_chunks),
        'token_usage': usage,
        'no_context': False,
    }
