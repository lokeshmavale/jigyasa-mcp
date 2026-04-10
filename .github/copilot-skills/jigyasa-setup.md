# Skill: jigyasa-setup

## Description
Sets up the full Jigyasa MCP code search pipeline for any Git repository. Handles Jigyasa server, indexing, MCP server configuration, and git hook installation — fully automated, repo-agnostic.

## When to invoke
- User asks to "set up code search", "index my repo", "set up Jigyasa", or "add MCP search for this repo"
- User wants to make a new codebase searchable via AI agents
- User wants to add another repo to an existing Jigyasa setup

## Step 0: Preflight check (MUST DO FIRST)

Run all checks in a single pass. Auto-fix everything possible. Only ask the user for things you genuinely can't determine.

**Jigyasa source repo:** `https://github.com/lokeshmavale/jigyasa`

```powershell
# Run all at once
python --version && java -version && git --version && pip show jigyasa-mcp 2>$null
```

**Decision table — check each row, auto-fix before asking:**

| Check | Command | If missing → Auto-fix |
|-------|---------|----------------------|
| Python 3.10+ | `python --version` | Stop. Tell user to install Python 3.10+. |
| Java 21+ | `java -version` | Stop. Tell user to install Java 21+. |
| Git | `git --version` | Stop. Tell user to install Git. |
| Jigyasa repo cloned | Look for `jigyasa` dir in `~/.jigyasa-mcp/` or common paths | Auto-fix: `git clone https://github.com/lokeshmavale/jigyasa ~/.jigyasa-mcp/jigyasa` |
| Jigyasa JAR built | Find `*Jigyasa*all*.jar` in cloned repo's `build/libs/` | Auto-fix: `cd <jigyasa-repo> && gradlew shadowJar --quiet` |
| jigyasa-mcp installed | `pip show jigyasa-mcp` | Auto-fix: `pip install -e C:\engram\jigyasa-mcp\jigyasa-mcp` (or wherever the package lives) |
| gRPC stubs generated | Check `dpSearch_pb2.py` exists | Auto-fix: `python build_proto.py --proto-dir <jigyasa-repo>/src/main/proto` |

**Then ask exactly 2 user questions** (skip any already known from CWD):

1. **Which repo to index?** → ask_user with choices: `["This repo ({CWD})", "Let me provide a path"]`
2. **Embeddings?** → ask_user with choices: `["No — BM25 only (Recommended, faster)", "Yes — add semantic search (90MB model)"]`

MCP config location is auto-detected: if `.vscode/` exists in the target repo, offer VS Code config; otherwise default to Copilot CLI config.

**Persist** collected paths so next run skips everything:
```powershell
[Environment]::SetEnvironmentVariable("JIGYASA_JAR", "<path>", "User")
```

## Setup procedure (run AFTER Step 0 passes all checks)

### Step 1: Install jigyasa-mcp

```powershell
pip install -e C:\engram\jigyasa-mcp\jigyasa-mcp
# Or if published: pip install jigyasa-mcp
```

If gRPC stubs aren't generated yet:
```powershell
cd C:\engram\jigyasa-mcp\jigyasa-mcp
python build_proto.py --proto-dir C:\azs\repos\jigyasa\src\main\proto
```

### Step 2: Build and start Jigyasa server

Check if Jigyasa is already running:
```powershell
python -c "from jigyasa_mcp.grpc_client import JigyasaClient; c = JigyasaClient(); print(c.health())"
```

If not running, build and start using the launcher (stores data in `~/.jigyasa-mcp/data/`):
```powershell
# Build JAR (one-time)
cd C:\azs\repos\jigyasa
.\gradlew.bat shadowJar --quiet

# Start via launcher — handles data dirs, env vars, PID tracking
jigyasa-server
# Or with custom settings:
jigyasa-server --port 50051 --heap-max 2g --jar C:\path\to\Jigyasa-all.jar
```

The launcher automatically:
- Creates `~/.jigyasa-mcp/data/IndexData/` for Lucene indexes
- Creates `~/.jigyasa-mcp/data/TransLog/` for write-ahead logs
- Sets `INDEX_CACHE_DIR`, `TRANSLOG_DIRECTORY`, `GRPC_SERVER_PORT` env vars
- Saves PID to `~/.jigyasa-mcp/jigyasa.pid`
- Polls until healthy (max 20s)

Check status or stop:
```powershell
jigyasa-server --status    # shows running, port, PID, index size
jigyasa-server --stop      # stops the server
```

### Step 3: Run self-test

```powershell
jigyasa-mcp --self-test
```

Expected output: `SELF-TEST PASSED: MCP ↔ Jigyasa connectivity OK`

### Step 4: Index the repository

For a **first-time full index**:
```powershell
jigyasa-index --repo <REPO_PATH> --endpoint localhost:50051
```

For **Phase 2 with embeddings** (optional, adds ~2min for large repos):
```powershell
jigyasa-index --repo <REPO_PATH> --endpoint localhost:50051 --embeddings
```

This creates 3 namespaced collections:
- `{prefix}_symbols` — classes, methods, fields, enums
- `{prefix}_chunks` — code bodies (BM25, optionally + KNN vectors)
- `{prefix}_files` — file-level metadata

Expected output example:
```
Full index complete:
  Files: 10914 indexed, 3453 skipped, 0 deleted
  Symbols: 155919
  Chunks: 182237
  Time: 92.3s
```

### Step 5: Register the repo

```python
from jigyasa_mcp.registry import RepoRegistry
registry = RepoRegistry.load()
registry.register(
    repo_root=r"<REPO_PATH>",
    prefix="<PREFIX>",
    use_embeddings=False  # or True if embeddings were used
)
print("Registered repos:", [(e.root, e.prefix) for e in registry.list_repos()])
```

### Step 6: Install git hooks for auto-indexing

**Windows:**
```powershell
$repoPath = "<REPO_PATH>"
$hookSource = "C:\engram\jigyasa-mcp\jigyasa-mcp\hooks\jigyasa-hook.bat"
foreach ($hook in @("post-commit", "post-merge", "post-checkout")) {
    Copy-Item $hookSource "$repoPath\.git\hooks\$hook"
}
Write-Host "Git hooks installed"
```

**Linux/macOS:**
```bash
REPO_PATH="<REPO_PATH>"
for hook in post-commit post-merge post-checkout; do
    cp hooks/jigyasa-hook.sh "$REPO_PATH/.git/hooks/$hook"
    chmod +x "$REPO_PATH/.git/hooks/$hook"
done
```

### Step 7: Configure MCP server for Copilot CLI

Add to the user's MCP configuration (location depends on the IDE/CLI):

For **GitHub Copilot CLI** (`~/.config/github-copilot/mcp.json` or equivalent):
```json
{
  "servers": {
    "jigyasa": {
      "command": "jigyasa-mcp",
      "args": [
        "--endpoint", "localhost:50051",
        "--repo", "<REPO_PATH>",
        "--auto-start"
      ]
    }
  }
}
```

The `--auto-start` flag makes the MCP server automatically start Jigyasa if it's not running — zero manual steps after initial setup.

For **VS Code Copilot** (`.vscode/mcp.json` in the workspace):
```json
{
  "servers": {
    "jigyasa": {
      "command": "jigyasa-mcp",
      "args": [
        "--endpoint", "localhost:50051",
        "--repo", "${workspaceFolder}",
        "--auto-start"
      ]
    }
  }
}
```

### Step 8: Verify end-to-end

After MCP is configured, test by asking Copilot:
- "Search for classes that implement ActionFilter" → should use `jigyasa_search_symbols`
- "How does shard allocation work?" → should use `jigyasa_search_code`
- "Find all gradle files in plugins/" → should use `jigyasa_search_files`

### Adding another repo to the same Jigyasa instance

Repeat Steps 1, 5, 6, 7 for the new repo. No need to restart Jigyasa or the MCP server.
Collections are namespaced automatically (e.g., `opensearch_symbols`, `jigyasa_symbols`).
The MCP server auto-detects which repo to query based on CWD.

## Troubleshooting

| Problem | Diagnosis | Fix |
|---------|-----------|-----|
| `ConnectionError: circuit breaker open` | Jigyasa not running | Start Jigyasa JAR |
| `RuntimeError: gRPC stubs not generated` | Proto not compiled | Run `python build_proto.py` |
| `VersionError: Protobuf gencode/runtime mismatch` | Proto stubs from different protobuf version | Re-run `python build_proto.py` to regenerate |
| Index is stale after branch switch | Git hook not installed or not firing | Check `.git/hooks/post-checkout` exists and is executable |
| `RuntimeError: Another indexing process is already running` | Concurrent index | Wait for the other process or delete `.jigyasa/index.lock` |
| Queries return 0 results | Collections empty or wrong prefix | Run `jigyasa-index --repo <path> --status` to check |
| Embedding model takes 20s on first query | Model cold start | Add `--embeddings` flag to MCP server for prewarm |

## Architecture reference

```
Single Jigyasa (localhost:50051)
├── opensearch_symbols     ← Repo 1: C:\azs\repos\OpenSearch
├── opensearch_chunks
├── opensearch_files
├── jigyasa_symbols        ← Repo 2: C:\azs\repos\jigyasa
├── jigyasa_chunks
├── jigyasa_files
└── myapp_symbols          ← Repo N: any git repo
    ...

Single MCP Server (stdio, launched by Copilot CLI)
├── Auto-detects repo from CWD via ~/.jigyasa-mcp/repos.json
├── Routes to correct namespaced collections
└── Falls back to un-namespaced if no registry match

Git Hooks (per repo)
├── post-commit    → jigyasa-index --incremental
├── post-merge     → jigyasa-index --incremental
└── post-checkout  → jigyasa-index --incremental
    └── Detects branch switch, diffs across branches
    └── Falls back to full reindex if old commit is gone
```
