# RAG Design Notes

The RAG service refuses to answer when no retrieved chunk passes the minimum relevance threshold. The exact refusal string is: I don't have enough relevant context to answer that.

The ingestion pipeline supports Markdown, HTML, plain text, and PDF files. It normalizes text, splits documents into chunks, embeds each chunk, and stores vectors with source metadata.

Default chunking uses 900 tokens with 120 token overlap. This setting was chosen because smaller chunks improved precision but broke multi-step answers, while larger chunks improved recall but added irrelevant context.

Idempotent re-ingest is implemented by deriving a stable source hash and replacing chunks for the same source path before writing the current chunks. Re-running ingestion on the same corpus keeps the vector count unchanged.
