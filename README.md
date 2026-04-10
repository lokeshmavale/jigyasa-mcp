# Jigyasa MCP — Code Search for AI Agents

Local codebase indexing + MCP server for GitHub Copilot CLI integration.

## Quick Start

```bash
# Install
pip install -e .

# Start Jigyasa (must be running)
java -jar path/to/Jigyasa-all.jar

# Full index
jigyasa-index --repo C:\azs\repos\OpenSearch

# Incremental index
jigyasa-index --repo C:\azs\repos\OpenSearch --incremental

# Start MCP server
jigyasa-mcp --endpoint localhost:50051
```

## Architecture

- **Indexer**: tree-sitter AST chunking → 3 Jigyasa collections (symbols, chunks, files)
- **MCP Server**: 6 tools exposed to Copilot CLI (search_symbols, search_code, search_files, get_context, index_status, reindex)
- **Embeddings** (Phase 2): all-MiniLM-L6-v2 local embeddings for hybrid BM25+KNN search
