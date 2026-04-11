"""Multi-repo registry — maps repo roots to namespaced Jigyasa collections.

The registry is a JSON config file that tracks all indexed repos:
  ~/.jigyasa-mcp/repos.json

Each entry maps a repo root path to its collection prefix.
The MCP server uses CWD to auto-detect which repo is active.
"""

import json
import logging
import os
from dataclasses import asdict, dataclass, field

logger = logging.getLogger(__name__)

CONFIG_DIR = os.path.join(os.path.expanduser("~"), ".jigyasa-mcp")
REPOS_FILE = "repos.json"


@dataclass
class RepoEntry:
    root: str  # absolute path to repo root
    prefix: str  # collection namespace (e.g., "opensearch")
    indexed_at: str = ""  # last indexed timestamp
    use_embeddings: bool = False


@dataclass
class RepoRegistry:
    repos: dict[str, RepoEntry] = field(default_factory=dict)
    # key = normalized absolute path

    def register(self, repo_root: str, prefix: str, use_embeddings: bool = False):
        """Register a repo for indexing."""
        normalized = os.path.realpath(repo_root).replace("\\", "/").rstrip("/")
        self.repos[normalized] = RepoEntry(
            root=normalized,
            prefix=prefix,
            use_embeddings=use_embeddings,
        )
        self.save()
        logger.info(f"Registered repo: {normalized} → prefix '{prefix}'")

    def unregister(self, repo_root: str):
        normalized = os.path.realpath(repo_root).replace("\\", "/").rstrip("/")
        if normalized in self.repos:
            del self.repos[normalized]
            self.save()

    def find_by_cwd(self, cwd: str) -> RepoEntry | None:
        """Find which registered repo contains the given CWD.

        Walks up from CWD to find the best (deepest) match.
        """
        cwd_normalized = os.path.realpath(cwd).replace("\\", "/").rstrip("/")

        best_match: RepoEntry | None = None
        best_depth = -1

        for path, entry in self.repos.items():
            if cwd_normalized == path or cwd_normalized.startswith(path + "/"):
                depth = path.count("/")
                if depth > best_depth:
                    best_match = entry
                    best_depth = depth

        return best_match

    def find_by_prefix(self, prefix: str) -> RepoEntry | None:
        for entry in self.repos.values():
            if entry.prefix == prefix:
                return entry
        return None

    def list_repos(self) -> list[RepoEntry]:
        return list(self.repos.values())

    def save(self):
        os.makedirs(CONFIG_DIR, exist_ok=True)
        path = os.path.join(CONFIG_DIR, REPOS_FILE)
        data = {k: asdict(v) for k, v in self.repos.items()}
        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    @classmethod
    def load(cls) -> "RepoRegistry":
        path = os.path.join(CONFIG_DIR, REPOS_FILE)
        registry = cls()
        if not os.path.exists(path):
            return registry
        try:
            with open(path) as f:
                data = json.load(f)
            for key, entry_data in data.items():
                registry.repos[key] = RepoEntry(**entry_data)
        except Exception as e:
            logger.warning(f"Failed to load repo registry: {e}")
        return registry
