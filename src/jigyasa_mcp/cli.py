"""CLI entry points for jigyasa-index and jigyasa-mcp commands."""

import asyncio
import logging
import sys

import click

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)


@click.command("jigyasa-index")
@click.option("--repo", required=True, help="Path to the Git repository to index")
@click.option("--endpoint", default="localhost:50051", help="Jigyasa gRPC endpoint")
@click.option("--incremental", is_flag=True, help="Only index changed files")
@click.option("--embeddings", is_flag=True, help="Generate local embeddings (Phase 2)")
@click.option("--status", is_flag=True, help="Show index status and exit")
def index_cli(repo: str, endpoint: str, incremental: bool, embeddings: bool, status: bool):
    """Index a Git repository into Jigyasa for AI-powered code search."""
    from jigyasa_mcp.indexer.pipeline import Indexer

    indexer = Indexer(repo, endpoint=endpoint, use_embeddings=embeddings)

    if status:
        import json
        print(json.dumps(indexer.get_status(), indent=2))
        return

    if incremental:
        stats = indexer.incremental_index()
    else:
        stats = indexer.full_index()

    print(f"\n{'Incremental' if incremental else 'Full'} index complete:")
    print(f"  Files: {stats.files_indexed} indexed, {stats.files_skipped} skipped, {stats.files_deleted} deleted")
    print(f"  Symbols: {stats.symbols_indexed}")
    print(f"  Chunks: {stats.chunks_indexed}")
    if stats.embeddings_generated:
        print(f"  Embeddings: {stats.embeddings_generated}")
    print(f"  Time: {stats.elapsed_seconds:.1f}s")
    if stats.errors:
        print(f"  Errors ({len(stats.errors)}):")
        for err in stats.errors[:10]:
            print(f"    - {err}")


@click.command("jigyasa-mcp")
@click.option("--endpoint", default="localhost:50051", help="Jigyasa gRPC endpoint")
@click.option("--repo", default="", help="Repository root for context lookups and reindexing")
@click.option("--embeddings", is_flag=True, help="Enable hybrid search with local embeddings")
@click.option("--self-test", "run_self_test", is_flag=True, help="Run connectivity self-test and exit")
def mcp_cli(endpoint: str, repo: str, embeddings: bool, run_self_test: bool):
    """Start the Jigyasa MCP server for GitHub Copilot CLI integration."""
    if run_self_test:
        from jigyasa_mcp.server.mcp_server import self_test
        click.echo(f"Running self-test against {endpoint}...")
        if self_test(endpoint):
            click.echo("SELF-TEST PASSED: MCP ↔ Jigyasa connectivity OK")
            sys.exit(0)
        else:
            click.echo("SELF-TEST FAILED: Cannot reach Jigyasa or query failed", err=True)
            sys.exit(1)

    from jigyasa_mcp.server.mcp_server import run_server
    click.echo(f"Starting Jigyasa MCP server (endpoint={endpoint}, repo={repo})", err=True)
    asyncio.run(run_server(endpoint, repo, embeddings))


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "mcp":
        mcp_cli()
    else:
        index_cli()
