"""Jigyasa MCP Server — exposes code search tools to GitHub Copilot CLI.

Tools:
  - jigyasa_search_symbols: Search classes, methods, fields by name/signature
  - jigyasa_search_code: Search code content (function/class bodies)
  - jigyasa_search_files: Search files by path, name, description
  - jigyasa_get_context: Get surrounding code for a file location
  - jigyasa_index_status: Check index freshness and doc counts
  - jigyasa_reindex: Trigger incremental or full reindex
"""

import json
import logging
import os
import signal
import threading

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import TextContent, Tool
from pydantic import ValidationError

from jigyasa_mcp.git_history import (
    format_commit_diff,
    format_commits,
    format_file_history,
    get_commit_diff,
    get_file_history,
    search_commits,
)
from jigyasa_mcp.grpc_client import JigyasaClient
from jigyasa_mcp.indexer.embeddings import embed_single
from jigyasa_mcp.indexer.embeddings import is_available as embeddings_available
from jigyasa_mcp.indexer.pipeline import Indexer
from jigyasa_mcp.server.highlighter import highlight_search_result
from jigyasa_mcp.server.reranker import rerank
from jigyasa_mcp.server.validation import (
    GetCommitDiffInput,
    GetContextInput,
    GetFileHistoryInput,
    ReindexInput,
    SearchCodeInput,
    SearchCommitsInput,
    SearchFilesInput,
    SearchSymbolsInput,
    truncate_response,
    validate_path_within_root,
)

logger = logging.getLogger(__name__)


def _format_hits(result, max_results: int = 20, query: str = "") -> str:
    """Format search results for LLM consumption."""
    lines = [f"Found {result.total_hits} results ({result.latency_ms:.1f}ms):"]
    for i, hit in enumerate(result.hits[:max_results]):
        src = hit.source
        if "file_path" in src:
            loc = f"{src['file_path']}"
            if "line_start" in src:
                loc += f":{src['line_start']}-{src.get('line_end', '?')}"
        else:
            loc = hit.doc_id

        # Build a useful one-liner depending on collection
        if "signature" in src:
            # Symbol hit
            kind = src.get("kind", "")
            name = src.get("qualified_name", src.get("name", ""))
            sig = src.get("signature", "")
            lines.append(f"  [{i+1}] {kind} {name} — {sig}")
            lines.append(f"      {loc}")
            if src.get("extends_class"):
                lines.append(f"      extends {src['extends_class']}")
            if src.get("implements"):
                lines.append(f"      implements {src['implements']}")
            if src.get("type_references"):
                lines.append(f"      references: {src['type_references'][:100]}")
        elif "content" in src:
            # Chunk hit — use highlighter if query is available
            symbol = src.get("symbol_name", "")
            lines.append(f"  [{i+1}] {loc} ({src.get('kind', 'chunk')}: {symbol})")
            if query:
                highlighted = highlight_search_result(src, query)
                if highlighted:
                    lines.append(f"      {highlighted}")
                else:
                    preview = src['content'][:300]
                    lines.append(f"      {preview.replace(chr(10), chr(10) + '      ')}")
            else:
                preview = src['content'][:300]
                lines.append(f"      {preview.replace(chr(10), chr(10) + '      ')}")
        elif "path" in src:
            # File hit
            lines.append(
                f"  [{i+1}] {src['path']} "
                f"({src.get('loc', '?')} lines, module: {src.get('module', '?')})"
            )
            if src.get("class_names"):
                lines.append(f"      classes: {src['class_names']}")
        else:
            lines.append(f"  [{i+1}] {loc} (score: {hit.score:.3f})")

        lines.append(f"      score: {hit.score:.3f}")
    return "\n".join(lines)


def _resolve_repo(repo_root: str, cwd: str = "") -> tuple[str, str, dict[str, str]]:
    """Resolve which repo to target.

    Returns: (repo_root, prefix, collection_names_dict)
    Uses CWD auto-detection via the repo registry.
    """
    from jigyasa_mcp.indexer.pipeline import _collection_names, _derive_repo_prefix
    from jigyasa_mcp.registry import RepoRegistry

    registry = RepoRegistry.load()

    # 1. Try CWD detection
    if cwd:
        entry = registry.find_by_cwd(cwd)
        if entry:
            return entry.root, entry.prefix, _collection_names(entry.prefix)

    # 2. Fall back to explicit repo_root
    if repo_root:
        prefix = _derive_repo_prefix(repo_root)
        entry = registry.find_by_prefix(prefix)
        if entry:
            return entry.root, entry.prefix, _collection_names(entry.prefix)
        return repo_root, prefix, _collection_names(prefix)

    # 3. No repo found — use un-namespaced collections (backward compat)
    return "", "", {"symbols": "symbols", "chunks": "chunks", "files": "files"}


def create_mcp_server(
    endpoint: str = "localhost:50051",
    repo_root: str = "",
    use_embeddings: bool = False,
) -> Server:
    """Create and configure the MCP server with all tools."""
    server = Server("jigyasa-mcp")
    client = JigyasaClient(endpoint=endpoint)

    @server.list_tools()
    async def list_tools() -> list[Tool]:
        return [
            Tool(
                name="jigyasa_search_symbols",
                description=(
                    "Search code symbols (classes, methods, fields, enums, interfaces) "
                    "by name, signature, or metadata. Use for precise lookups like "
                    "'find class X', 'methods in Y', 'implementations of Z'."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": (
                                "Search query — matches against name, "
                                "qualified_name, signature, package"
                            ),
                        },
                        "kind": {
                            "type": "array",
                            "items": {
                                "type": "string",
                                "enum": [
                                    "class", "interface", "enum",
                                    "method", "constructor", "field",
                                ],
                            },
                            "description": "Filter by symbol kind",
                        },
                        "visibility": {
                            "type": "array",
                            "items": {
                                "type": "string",
                                "enum": [
                                    "public", "protected",
                                    "private", "package-private",
                                ],
                            },
                            "description": "Filter by visibility",
                        },
                        "package_prefix": {
                            "type": "string",
                            "description": (
                                "Filter by package prefix "
                                "(e.g., 'org.opensearch.cluster')"
                            ),
                        },
                        "file_pattern": {
                            "type": "string",
                            "description": "Filter by file path pattern (e.g., 'server/src')",
                        },
                        "extends_or_implements": {
                            "type": "string",
                            "description": (
                                "Filter symbols that extend or "
                                "implement this class/interface"
                            ),
                        },
                        "has_annotation": {
                            "type": "string",
                            "description": "Filter by annotation (e.g., 'Override', 'Inject')",
                        },
                        "limit": {
                            "type": "integer",
                            "description": "Max results (default: 30)",
                            "default": 30,
                        },
                    },
                    "required": ["query"],
                },
            ),
            Tool(
                name="jigyasa_search_code",
                description=(
                    "Search code content — function bodies, class implementations, "
                    "configuration blocks. Use for concept search like 'retry logic', "
                    "'shard allocation', 'connection pooling'. Returns matching code "
                    "snippets with file paths and line numbers."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "Search query — matches against code content",
                        },
                        "file_types": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Filter by file extensions (e.g., ['java', 'gradle'])",
                        },
                        "module_path": {
                            "type": "string",
                            "description": (
                                "Filter by module "
                                "(e.g., 'server', 'plugins/transport-nio')"
                            ),
                        },
                        "enclosing_class": {
                            "type": "string",
                            "description": "Filter chunks inside this class",
                        },
                        "exclude_tests": {
                            "type": "boolean",
                            "description": "Exclude test files (default: true)",
                            "default": True,
                        },
                        "limit": {
                            "type": "integer",
                            "description": "Max results (default: 15)",
                            "default": 15,
                        },
                    },
                    "required": ["query"],
                },
            ),
            Tool(
                name="jigyasa_search_files",
                description=(
                    "Search files by path, name, or content summary. Use for file "
                    "discovery like 'find test files for TransportAction', "
                    "'gradle configs in plugins/', 'all files in cluster module'."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": (
                                "Search query — matches path, "
                                "filename, class names, imports"
                            ),
                        },
                        "extension": {
                            "type": "string",
                            "description": "Filter by extension (e.g., 'java', 'gradle')",
                        },
                        "module": {
                            "type": "string",
                            "description": "Filter by module name",
                        },
                        "limit": {
                            "type": "integer",
                            "description": "Max results (default: 20)",
                            "default": 20,
                        },
                    },
                    "required": ["query"],
                },
            ),
            Tool(
                name="jigyasa_get_context",
                description=(
                    "Get surrounding source code for a specific file location. "
                    "Use after a search hit to see more context around a result."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {
                        "file_path": {
                            "type": "string",
                            "description": "Relative file path from repo root",
                        },
                        "line_start": {
                            "type": "integer",
                            "description": "Start line number",
                        },
                        "line_end": {
                            "type": "integer",
                            "description": "End line number",
                        },
                        "radius": {
                            "type": "integer",
                            "description": "Extra lines before and after (default: 10)",
                            "default": 10,
                        },
                    },
                    "required": ["file_path", "line_start", "line_end"],
                },
            ),
            Tool(
                name="jigyasa_index_status",
                description=(
                    "Check the current index status — last indexed commit, "
                    "freshness, document counts, and collection health."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {},
                },
            ),
            Tool(
                name="jigyasa_reindex",
                description=(
                    "Trigger a reindex of the codebase. Use 'incremental' for "
                    "fast updates (only changed files) or 'full' for a complete rebuild."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {
                        "mode": {
                            "type": "string",
                            "enum": ["incremental", "full"],
                            "description": "Reindex mode (default: incremental)",
                            "default": "incremental",
                        },
                    },
                },
            ),
            Tool(
                name="jigyasa_search_commits",
                description=(
                    "Search git commit history by keyword, author, date range, "
                    "or file path. Use to find when/why code changed, who made "
                    "changes, or what happened in a time period."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "Search text in commit messages",
                        },
                        "author": {
                            "type": "string",
                            "description": "Filter by author name or email",
                        },
                        "since": {
                            "type": "string",
                            "description": (
                                "Start date (ISO 8601 or relative, "
                                "e.g. '7 days ago', '2024-01-01')"
                            ),
                        },
                        "until": {
                            "type": "string",
                            "description": "End date",
                        },
                        "file_path": {
                            "type": "string",
                            "description": (
                                "Only commits that modified this file"
                            ),
                        },
                        "limit": {
                            "type": "integer",
                            "description": "Max results (default: 20)",
                            "default": 20,
                        },
                    },
                },
            ),
            Tool(
                name="jigyasa_commit_diff",
                description=(
                    "Get the full diff for a specific commit — metadata, "
                    "changed files, and line-level code changes. Use when "
                    "you need to see exactly what changed in a commit."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {
                        "sha": {
                            "type": "string",
                            "description": (
                                "Commit SHA (full or abbreviated, "
                                "min 4 chars)"
                            ),
                        },
                        "context_lines": {
                            "type": "integer",
                            "description": (
                                "Lines of context around changes "
                                "(default: 3)"
                            ),
                            "default": 3,
                        },
                    },
                    "required": ["sha"],
                },
            ),
            Tool(
                name="jigyasa_file_history",
                description=(
                    "Trace how a file evolved over time — all commits "
                    "that modified it with optional diffs. Follows "
                    "renames. Use to understand why code looks the way "
                    "it does."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {
                        "file_path": {
                            "type": "string",
                            "description": (
                                "Relative file path to trace"
                            ),
                        },
                        "limit": {
                            "type": "integer",
                            "description": (
                                "Max commits to return (default: 15)"
                            ),
                            "default": 15,
                        },
                        "include_diff": {
                            "type": "boolean",
                            "description": (
                                "Include diffs per commit (default: true)"
                            ),
                            "default": True,
                        },
                    },
                    "required": ["file_path"],
                },
            ),
        ]

    @server.call_tool()
    async def call_tool(name: str, arguments: dict) -> list[TextContent]:
        try:
            # Resolve repo + namespaced collections once per call
            resolved_root, _, cols = _resolve_repo(repo_root)

            if name == "jigyasa_search_symbols":
                validated = SearchSymbolsInput(**arguments)
                text = _handle_search_symbols(client, validated, cols)
            elif name == "jigyasa_search_code":
                validated = SearchCodeInput(**arguments)
                text = _handle_search_code(client, validated, use_embeddings, cols)
            elif name == "jigyasa_search_files":
                validated = SearchFilesInput(**arguments)
                text = _handle_search_files(client, validated, cols)
            elif name == "jigyasa_get_context":
                validated = GetContextInput(**arguments)
                text = _handle_get_context(validated, resolved_root or repo_root)
            elif name == "jigyasa_index_status":
                text = _handle_index_status(resolved_root or repo_root, endpoint)
            elif name == "jigyasa_reindex":
                validated = ReindexInput(**arguments)
                text = _handle_reindex(
                    validated, resolved_root or repo_root,
                    endpoint, use_embeddings,
                )
            elif name == "jigyasa_search_commits":
                validated = SearchCommitsInput(**arguments)
                root = resolved_root or repo_root
                if not root:
                    text = "ERROR: No repository configured"
                else:
                    commits = search_commits(
                        root,
                        query=validated.query,
                        author=validated.author,
                        since=validated.since,
                        until=validated.until,
                        file_path=validated.file_path,
                        max_results=validated.limit,
                    )
                    text = format_commits(commits)
            elif name == "jigyasa_commit_diff":
                validated = GetCommitDiffInput(**arguments)
                root = resolved_root or repo_root
                if not root:
                    text = "ERROR: No repository configured"
                else:
                    result = get_commit_diff(
                        root, validated.sha,
                        context_lines=validated.context_lines,
                    )
                    if result:
                        text = format_commit_diff(result)
                    else:
                        text = f"ERROR: Commit {validated.sha} not found"
            elif name == "jigyasa_file_history":
                validated = GetFileHistoryInput(**arguments)
                root = resolved_root or repo_root
                if not root:
                    text = "ERROR: No repository configured"
                else:
                    entries = get_file_history(
                        root, validated.file_path,
                        max_results=validated.limit,
                        include_diff=validated.include_diff,
                    )
                    text = format_file_history(
                        entries, validated.file_path,
                    )
            else:
                text = f"Unknown tool: {name}"
            return [TextContent(type="text", text=truncate_response(text))]
        except ValidationError as e:
            return [TextContent(type="text", text=f"VALIDATION ERROR: {e}")]
        except ConnectionError as e:
            return [TextContent(type="text", text=f"ERROR: Jigyasa connection failed — {e}")]
        except Exception as e:
            logger.exception(f"Tool error: {name}")
            return [TextContent(type="text", text=f"ERROR: {type(e).__name__}: {e}")]

    return server


def _handle_search_symbols(client: JigyasaClient, args: SearchSymbolsInput, cols: dict) -> str:
    query = args.query
    filters = []

    if args.kind:
        for kind_val in args.kind:
            filters.append({"field": "kind", "value": kind_val})
    if args.visibility:
        for vis_val in args.visibility:
            filters.append({"field": "visibility", "value": vis_val})
    if args.file_pattern:
        filters.append({"field": "file_path", "value": args.file_pattern})
    if args.extends_or_implements:
        query += f" {args.extends_or_implements}"
    if args.has_annotation:
        query += f" {args.has_annotation}"
    if args.package_prefix:
        query += f" {args.package_prefix}"

    result = client.query(cols["symbols"], text_query=query, filters=filters, top_k=args.limit)
    result = rerank(result, args.query)
    return _format_hits(result, max_results=args.limit, query=args.query)


def _handle_search_code(
    client: JigyasaClient, args: SearchCodeInput,
    use_embeddings: bool, cols: dict,
) -> str:
    query = args.query
    filters = []

    if args.file_types:
        for ft in args.file_types:
            filters.append({"field": "language", "value": ft})
    if args.module_path:
        filters.append({"field": "module", "value": args.module_path})
    if args.enclosing_class:
        filters.append({"field": "enclosing_class", "value": args.enclosing_class})

    vector = None
    if use_embeddings and embeddings_available():
        try:
            vector = embed_single(query)
        except Exception as e:
            logger.warning(f"Embedding failed, falling back to text-only search: {e}")
            vector = None

    result = client.query(
        cols["chunks"],
        text_query=query,
        filters=filters,
        top_k=args.limit,
        vector=vector,
        text_weight=0.4 if vector else 1.0,
    )
    result = rerank(result, args.query, exclude_tests=args.exclude_tests)
    return _format_hits(result, max_results=args.limit, query=args.query)


def _handle_search_files(client: JigyasaClient, args: SearchFilesInput, cols: dict) -> str:
    filters = []
    if args.extension:
        filters.append({"field": "extension", "value": args.extension})
    if args.module:
        filters.append({"field": "module", "value": args.module})

    result = client.query(cols["files"], text_query=args.query, filters=filters, top_k=args.limit)
    return _format_hits(result, max_results=args.limit, query=args.query)


def _handle_get_context(args: GetContextInput, repo_root: str) -> str:
    if not repo_root:
        return "ERROR: No repository configured"

    # Validate path stays within repo root (defense in depth beyond Pydantic)
    try:
        abs_path = validate_path_within_root(args.file_path, repo_root)
    except ValueError as e:
        return f"ERROR: {e}"

    if not os.path.exists(abs_path):
        return f"ERROR: File not found: {args.file_path}"

    try:
        with open(abs_path, encoding="utf-8", errors="ignore") as f:
            lines = f.readlines()
    except Exception as e:
        return f"ERROR: Cannot read {args.file_path}: {e}"

    start = max(0, args.line_start - 1 - args.radius)
    end = min(len(lines), args.line_end + args.radius)

    context_lines = []
    for i in range(start, end):
        marker = " " if (i + 1 < args.line_start or i + 1 > args.line_end) else ">"
        context_lines.append(f"{marker} {i+1:5d} | {lines[i].rstrip()}")

    header = (
        f"--- {args.file_path} "
        f"(lines {start+1}-{end}, highlighted {args.line_start}-{args.line_end}) ---"
    )
    return header + "\n" + "\n".join(context_lines)


def _handle_index_status(repo_root: str, endpoint: str) -> str:
    if not repo_root:
        return "ERROR: No repository configured"
    indexer = Indexer(repo_root, endpoint=endpoint)
    status = indexer.get_status()
    return json.dumps(status, indent=2)


def _handle_reindex(args: ReindexInput, repo_root: str, endpoint: str, use_embeddings: bool) -> str:
    if not repo_root:
        return "ERROR: No repository configured"
    indexer = Indexer(
        repo_root, endpoint=endpoint, use_embeddings=use_embeddings,
        auto_install_grammars=True,
    )

    if args.mode == "full":
        stats = indexer.full_index()
    else:
        stats = indexer.incremental_index()

    return (
        f"Reindex ({args.mode}) complete:\n"
        f"  Files indexed: {stats.files_indexed}\n"
        f"  Files deleted: {stats.files_deleted}\n"
        f"  Symbols: {stats.symbols_indexed}\n"
        f"  Chunks: {stats.chunks_indexed}\n"
        f"  Embeddings: {stats.embeddings_generated}\n"
        f"  Time: {stats.elapsed_seconds:.1f}s\n"
        f"  Errors: {len(stats.errors)}"
    )



def self_test(endpoint: str) -> bool:
    """Verify MCP ↔ Jigyasa connectivity: health check + create + index + query."""
    import uuid
    test_collection = f"_selftest_{uuid.uuid4().hex[:8]}"
    client = JigyasaClient(endpoint=endpoint, timeout=5.0)
    try:
        # Step 1: Health check
        health = client.health()
        if health["status"] != "SERVING":
            logger.error("Self-test: Jigyasa not serving")
            return False

        # Step 2: Create, index, query
        schema = json.dumps({"fields": [
            {"name": "id", "type": "STRING", "key": True},
            {"name": "content", "type": "STRING", "searchable": True},
        ]})
        client.create_collection(test_collection, schema)
        client.index_batch(test_collection, [
            {"id": "test1", "content": "self test document for jigyasa mcp"}
        ], refresh="WAIT_FOR")
        import time
        time.sleep(0.5)
        result = client.query(test_collection, text_query="self test", top_k=1)
        return result.total_hits >= 1
    except Exception as e:
        logger.error(f"Self-test failed: {e}")
        return False
    finally:
        client.close()


async def run_server(endpoint: str, repo_root: str, use_embeddings: bool):
    """Run the MCP server over stdio with graceful shutdown."""
    server = create_mcp_server(
        endpoint=endpoint,
        repo_root=repo_root,
        use_embeddings=use_embeddings,
    )

    # Prewarm embedding model in background to avoid 20s delay on first query
    if use_embeddings:
        def _prewarm():
            try:
                from jigyasa_mcp.indexer.embeddings import embed_single
                embed_single("warmup")
                logger.info("Embedding model prewarmed")
            except Exception as e:
                logger.warning(f"Embedding prewarm failed: {e}")
        threading.Thread(target=_prewarm, daemon=True).start()

    # Graceful shutdown on SIGINT/SIGTERM
    client = JigyasaClient(endpoint=endpoint)

    def _shutdown(signum, frame):
        logger.info("Shutting down MCP server...")
        client.close()
        raise SystemExit(0)

    signal.signal(signal.SIGINT, _shutdown)
    signal.signal(signal.SIGTERM, _shutdown)

    try:
        async with stdio_server() as (read_stream, write_stream):
            await server.run(
                read_stream, write_stream, server.create_initialization_options()
            )
    finally:
        client.close()
