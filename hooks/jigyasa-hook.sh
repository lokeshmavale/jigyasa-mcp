#!/bin/bash
# Git hook: post-commit / post-merge / post-checkout
# Triggers incremental reindexing after git operations.
#
# Install:
#   cp hooks/jigyasa-hook.sh .git/hooks/post-commit
#   cp hooks/jigyasa-hook.sh .git/hooks/post-merge
#   cp hooks/jigyasa-hook.sh .git/hooks/post-checkout
#   chmod +x .git/hooks/post-commit .git/hooks/post-merge .git/hooks/post-checkout

# Check if jigyasa-index is available
command -v jigyasa-index >/dev/null 2>&1 || exit 0

REPO_ROOT="$(git rev-parse --show-toplevel)"

# Skip if another indexer is already running
[ -f "${REPO_ROOT}/.jigyasa/index.lock" ] && exit 0

# Ensure log directory exists
LOG_DIR="${HOME}/.jigyasa-mcp"
mkdir -p "${LOG_DIR}"

# Run indexer in background, log errors
(jigyasa-index --repo "${REPO_ROOT}" --incremental >> "${LOG_DIR}/hook.log" 2>&1 &)
exit 0
