# Changelog

## [0.1.0] — 2026-04-10

### Added
- **Indexing pipeline**: tree-sitter AST chunking for Java + line-based chunking for other files
- **3 Jigyasa collections**: `symbols` (BM25), `chunks` (BM25 + optional KNN), `files` (BM25)
- **Incremental indexing**: Git SHA watermarking + mtime/size change detection
- **MCP server**: 6 tools for Copilot CLI (`search_symbols`, `search_code`, `search_files`, `get_context`, `index_status`, `reindex`)
- **Phase 2 embeddings**: `all-MiniLM-L6-v2` (384 dims, free, CPU-only) with hybrid BM25+KNN search
- **Input validation**: Pydantic models for all MCP tool inputs
- **Path traversal protection**: `get_context` validates paths stay within repo root
- **Thread-safe circuit breaker**: gRPC client with `threading.Lock` protection
- **Retries with exponential backoff**: transient gRPC errors (UNAVAILABLE, DEADLINE_EXCEEDED) retried up to 3x
- **gRPC channel health check**: `channel_ready_future` on first connect
- **Auto-reconnect**: detects dead channels and reconnects transparently
- **Graceful shutdown**: signal handlers clean up gRPC channels
- **State file locking**: cross-platform file lock prevents concurrent index corruption
- **Result size limits**: responses capped at 15K chars to protect LLM context window
- **Embedding prewarm**: model loaded in background thread on MCP server start
- **Self-test**: `--self-test` flag verifies MCP ↔ Jigyasa connectivity
- **Proto build automation**: `build_proto.py` with auto-import fix
- **Pinned dependencies**: all versions locked in `pyproject.toml`
- **CI pipeline**: GitHub Actions with lint (ruff) + test (pytest) on Python 3.10-3.12
- **Git hooks**: `post-commit`, `post-merge`, `post-checkout` for auto-incremental indexing
