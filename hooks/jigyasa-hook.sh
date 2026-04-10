#!/bin/bash
# Git hook: post-commit / post-merge / post-checkout
# Triggers incremental reindexing after git operations.
#
# Install:
#   cp hooks/jigyasa-hook.sh .git/hooks/post-commit
#   cp hooks/jigyasa-hook.sh .git/hooks/post-merge
#   cp hooks/jigyasa-hook.sh .git/hooks/post-checkout
#   chmod +x .git/hooks/post-commit .git/hooks/post-merge .git/hooks/post-checkout

REPO_ROOT="$(git rev-parse --show-toplevel)"

# Run indexer in background so it doesn't slow down git operations
(jigyasa-index --repo "$REPO_ROOT" --incremental 2>/dev/null &)
