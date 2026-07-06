# Applied AI / ML Engineering Take-Home

Candidate: Kailash  
Focus: low-cost RAG with honest evaluation + auditable LLM-as-judge pipeline.

This repo is designed so a reviewer can clone it, ingest a small corpus, run a query, and reproduce the evaluation reports in under 10 minutes on a CPU machine.

## Architecture choices

### Problem 1: Cost-efficient RAG
- **Vector store:** LanceDB, embedded/local mode.
- **Embedding model:** default `local-hashing-384` for zero-key/offline reproducibility; production setting `BAAI/bge-small-en-v1.5`, 384 dimensions.
- **Chunking default:** 900 tokens with 120 token overlap.
- **No-context behavior:** retrieval confidence threshold; the API refuses to answer when retrieved chunks are below threshold.
- **Idempotent ingest:** each chunk ID is derived from `source_path + source_hash + chunk_index`; re-ingest deletes existing chunks for the same source before adding fresh chunks.

### Problem 2: LLM-as-judge
- **Modes implemented:** pointwise and pairwise A-vs-B.
- **Bias checks:** position swap, verbosity probe, self-enhancement separation, sycophancy/style probe, score-spread tracking.
- **Parsing:** strict JSON first, then JSON-repair fallback, then schema validation.
- **Auditability:** every prompt and raw response is written to `logs/judge_audit.jsonl`.

## Quickstart

```bash
git clone https://github.com/kailashv2/applied-ai-ml-engineering-assignment.git
cd applied-ai-ml-engineering-assignment
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\\Scripts\\activate
pip install -r requirements.txt
cp .env.example .env
```

Set keys in `.env` only if you want hosted generation/judging. The code has a deterministic local embedding fallback and a `mock` provider for smoke tests, so reviewers can run the full pipeline without secrets.

## Ingest a corpus

```bash
python -m src.rag.ingest \
  --input data/corpus \
  --db-uri ./.lancedb \
  --table chunks \
  --chunk-size 900 \
  --chunk-overlap 120
```

## Run the RAG API

```bash
uvicorn src.rag.api:app --host 0.0.0.0 --port 8000
```

Example query:

```bash
curl -s http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question":"When should the service refuse to answer?","k":5,"metadata_filter":{"doc_type":"md"}}' | jq
```

## Run RAG evaluation

```bash
python -m src.rag.eval_rag \
  --questions eval/rag_questions.jsonl \
  --db-uri ./.lancedb \
  --table chunks \
  --out results/rag_eval_report.json
```

## Run judge evaluation

```bash
python -m src.judge.pipeline \
  --suite eval/judge_suite.yaml \
  --out results/judge_report.json \
  --audit-log logs/judge_audit.jsonl
```

## Run A/B comparison

```bash
python -m src.judge.pipeline \
  --suite eval/judge_ab_suite.yaml \
  --mode pairwise \
  --out results/judge_ab_report.json \
  --audit-log logs/judge_audit.jsonl
```

## Discussion

I would keep LanceDB for large, lightly queried corpora where storage cost and operational simplicity matter more than globally distributed low-latency SLA. I would switch back to a managed vector DB when traffic becomes high and bursty, when the product needs multi-region availability, or when non-ML engineers need fully managed backups, monitoring, and scaling.

The weak link in the RAG stack is usually retrieval, not generation. The generator can only stay grounded if the retrieved context contains the answer. That is why this repo tracks retrieval metrics separately from answer metrics instead of only reporting subjective answer quality.
