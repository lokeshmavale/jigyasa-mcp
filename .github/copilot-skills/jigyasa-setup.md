# Skill: jigyasa-setup

## Description
Sets up the full Jigyasa MCP code search pipeline for any Git repository. Handles Jigyasa server, indexing, MCP server configuration, and git hook installation — fully automated, repo-agnostic.

## When to invoke
- User asks to "set up code search", "index my repo", "set up Jigyasa", or "add MCP search for this repo"
- User wants to make a new codebase searchable via AI agents
- User wants to add another repo to an existing Jigyasa setup

## Prerequisites check
Before starting, verify these are available. If any are missing, guide the user to install them.

1. **Python 3.10+**: `python --version`
2. **Java 21+**: `java -version` (needed for Jigyasa server)
3. **Git**: `git --version`
4. **jigyasa-mcp package**: `pip show jigyasa-mcp` (may need install)
5. **Jigyasa JAR**: Check if built or available

## Setup procedure

### Step 1: Identify the target repo

Ask the user which repository to index. Get the absolute path. If they say "this repo", use the current working directory and find the git root:
```powershell
git rev-parse --show-toplevel
```

Derive the repo name for collection namespacing:
```python
import os, re
repo_root = r"<path>"
name = os.path.basename(os.path.abspath(repo_root))
prefix = re.sub(r"[^a-z0-9]", "_", name.lower()).strip("_")
# e.g., "OpenSearch" → "opensearch", "my-cool-project" → "my_cool_project"
```

### Step 2: Install jigyasa-mcp

```powershell
pip install -e C:\engram\jigyasa-mcp\jigyasa-mcp
# Or if published: pip install jigyasa-mcp
```

If gRPC stubs aren't generated yet:
```powershell
cd C:\engram\jigyasa-mcp\jigyasa-mcp
python build_proto.py --proto-dir C:\azs\repos\jigyasa\src\main\proto
```

### Step 3: Build and start Jigyasa server

Check if Jigyasa is already running:
```powershell
python -c "from jigyasa_mcp.grpc_client import JigyasaClient; c = JigyasaClient(); print(c.health())"
```

If not running, build and start:
```powershell
cd C:\azs\repos\jigyasa
.\gradlew.bat shadowJar --quiet
# Start in background (detached)
Start-Process java -ArgumentList '--add-modules','jdk.incubator.vector','-Xms512m','-Xmx1g','-jar','build\libs\Jigyasa-1.0-SNAPSHOT-all.jar' -WindowStyle Hidden
```

Wait for it to be ready:
```powershell
# Poll until healthy (max 15 seconds)
python -c "
import time
from jigyasa_mcp.grpc_client import JigyasaClient
client = JigyasaClient()
for i in range(15):
    try:
        h = client.health()
        if h['status'] == 'SERVING': print('Jigyasa ready'); break
    except: pass
    time.sleep(1)
else:
    print('ERROR: Jigyasa did not start')
"
```

### Step 4: Run self-test

```powershell
jigyasa-mcp --self-test
```

Expected output: `SELF-TEST PASSED: MCP ↔ Jigyasa connectivity OK`

### Step 5: Index the repository

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

### Step 6: Register the repo

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

### Step 7: Install git hooks for auto-indexing

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

### Step 8: Configure MCP server for Copilot CLI

Add to the user's MCP configuration (location depends on the IDE/CLI):

For **GitHub Copilot CLI** (`~/.config/github-copilot/mcp.json` or equivalent):
```json
{
  "servers": {
    "jigyasa": {
      "command": "jigyasa-mcp",
      "args": [
        "--endpoint", "localhost:50051",
        "--repo", "<REPO_PATH>"
      ]
    }
  }
}
```

For **VS Code Copilot** (`.vscode/mcp.json` in the workspace):
```json
{
  "servers": {
    "jigyasa": {
      "command": "jigyasa-mcp",
      "args": [
        "--endpoint", "localhost:50051",
        "--repo", "${workspaceFolder}"
      ]
    }
  }
}
```

### Step 9: Verify end-to-end

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
