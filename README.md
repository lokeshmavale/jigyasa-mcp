# Jigyasa MCP — Code Search for AI Agents

Local codebase indexing + MCP server that gives AI coding agents (GitHub Copilot, Cursor, etc.) deep code understanding — symbol search, code search, git history, and more.

## Features

- **9 MCP tools** for AI agents — symbol search, code search, file search, context retrieval, git history, indexing
- **Multi-language AST parsing** — Java, Python, TypeScript, JavaScript, Go, C#, Rust, C/C++, Ruby, Kotlin, Scala
- **Self-adaptive** — auto-detects languages in your repo and installs grammars on demand
- **Git history** — search commits, view diffs, trace file evolution
- **Hybrid search** — BM25 text + optional vector KNN (all-MiniLM-L6-v2)
- **Incremental indexing** — only re-indexes changed files via git diff + mtime tracking
- **Resilient** — circuit breaker, retries, atomic state files, cross-platform lock files

## Quick Start

```bash
# 1. Install jigyasa-mcp
pip install -e .

# 2. Start Jigyasa server (requires Java 21+)
java --add-modules jdk.incubator.vector -jar path/to/Jigyasa-all.jar

# 3. Index your repo (auto-installs language grammars)
jigyasa-index --repo /path/to/your/repo --auto-install-grammars

# 4. Start MCP server
jigyasa-mcp --endpoint localhost:50051 --repo /path/to/your/repo
```

### Install Options

```bash
pip install -e .                    # Core only (Java AST always included)
pip install -e ".[languages]"       # All language grammars (Python, TS, Go, etc.)
pip install -e ".[all]"             # Languages + embeddings
pip install -e ".[dev]"             # Dev tools (pytest, ruff, mypy)
```

## MCP Tools

| Tool | Description |
|------|-------------|
| `jigyasa_search_symbols` | Find classes, methods, fields by name, kind, visibility, package, annotations, inheritance |
| `jigyasa_search_code` | Full-text + optional vector search across code chunks with module/class filters |
| `jigyasa_search_files` | Find files by path, name, extension, module |
| `jigyasa_get_context` | Retrieve file content with line range and highlighted focus |
| `jigyasa_index_status` | Check index health — stale/current, commit SHA, collection stats, language support |
| `jigyasa_reindex` | Trigger incremental or full reindex |
| `jigyasa_search_commits` | Search git commits by keyword, author, date range, or file path |
| `jigyasa_commit_diff` | Full diff for a specific commit — metadata + per-file unified diffs |
| `jigyasa_file_history` | Trace how a file evolved over time with diffs, follows renames |

## Supported Languages

| Language | AST Parsing | Pip Package |
|----------|------------|-------------|
| Java | ✅ Always available | `tree-sitter-java` (required dependency) |
| Python | ✅ Auto-install | `tree-sitter-python` |
| TypeScript | ✅ Auto-install | `tree-sitter-typescript` |
| JavaScript | ✅ Auto-install | `tree-sitter-javascript` |
| Go | ✅ Auto-install | `tree-sitter-go` |
| C# | ✅ Auto-install | `tree-sitter-c-sharp` |
| Rust | ✅ Auto-install | `tree-sitter-rust` |
| C / C++ | ✅ Auto-install | `tree-sitter-c` / `tree-sitter-cpp` |
| Ruby | ✅ Auto-install | `tree-sitter-ruby` |
| Kotlin | ✅ Auto-install | `tree-sitter-kotlin` |
| Other (.xml, .yml, .md, etc.) | Line-based chunking | — |

## Architecture

```
┌─────────────────────┐     ┌──────────────────┐
│  AI Agent (Copilot)  │────▶│  MCP Server (9)  │
└─────────────────────┘     └────────┬─────────┘
                                     │
                    ┌────────────────┼────────────────┐
                    ▼                ▼                ▼
            ┌──────────┐    ┌──────────────┐   ┌──────────┐
            │  Jigyasa  │    │ Git History  │   │  Local   │
            │  (gRPC)   │    │ (subprocess) │   │  Files   │
            └──────────┘    └──────────────┘   └──────────┘
                 │
    ┌────────────┼────────────┐
    ▼            ▼            ▼
symbols       chunks        files
(BM25)     (BM25+KNN)     (BM25)
```

- **Indexer**: Language Registry → tree-sitter AST chunking → 3 Jigyasa collections
- **MCP Server**: 9 tools exposed via Model Context Protocol
- **Git History**: Local git operations for commit search, diffs, file evolution
- **Embeddings** (optional): all-MiniLM-L6-v2 for hybrid BM25+KNN search

## CLI Reference

```bash
# Indexing
jigyasa-index --repo .                          # Full index
jigyasa-index --repo . --incremental            # Changed files only
jigyasa-index --repo . --auto-install-grammars  # Auto-install language support
jigyasa-index --repo . --status                 # Show index status
jigyasa-index --repo . --embeddings             # Enable vector search
jigyasa-index --repo . --log-level DEBUG        # Verbose logging

# MCP Server
jigyasa-mcp --repo . --endpoint localhost:50051
jigyasa-mcp --repo . --auto-start               # Auto-start Jigyasa if not running
jigyasa-mcp --repo . --self-test                 # Verify connectivity
jigyasa-mcp --repo . --log-level WARNING         # Quiet mode
```

## Development

```bash
pip install -e ".[dev]"
pytest tests/                    # 164 tests
ruff check src/ tests/           # Lint
mypy src/                        # Type check
```

## License

Apache 2.0
