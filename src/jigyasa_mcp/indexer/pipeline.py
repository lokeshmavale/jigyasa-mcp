"""Indexing pipeline: full and incremental indexing of a Git repository."""

import json
import logging
import os
import subprocess
import tempfile
import time
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from pathlib import Path

from jigyasa_mcp.grpc_client import JigyasaClient
from jigyasa_mcp.indexer.chunker import (
    Chunk,
    FileDoc,
    JavaChunker,
    Symbol,
    TextChunker,
    should_skip_file,
)
from jigyasa_mcp.indexer.embeddings import embed_texts
from jigyasa_mcp.indexer.embeddings import is_available as embeddings_available
from jigyasa_mcp.indexer.generic_ast_chunker import GenericASTChunker
from jigyasa_mcp.indexer.lang_registry import get_registry
from jigyasa_mcp.schemas.collections import get_schema_json

logger = logging.getLogger(__name__)

STATE_DIR = ".jigyasa"
STATE_FILE = "index_state.json"

INDEXABLE_EXTENSIONS = {
    ".java", ".gradle", ".xml", ".yml", ".yaml", ".json", ".md",
    ".properties", ".cfg", ".txt", ".proto", ".py", ".sh", ".bat",
}

# Batch sizes for Jigyasa upserts
SYMBOL_BATCH_SIZE = 500
CHUNK_BATCH_SIZE = 200
FILE_BATCH_SIZE = 200
EMBEDDING_BATCH_SIZE = 64


LOCK_FILE = "index.lock"


def _is_pid_alive(pid: int) -> bool:
    """Check if a process with the given PID is still running (cross-platform)."""
    try:
        os.kill(pid, 0)
        return True
    except (OSError, PermissionError):
        return False


@contextmanager
def _file_lock(repo_root: str):
    """PID-based lock to prevent concurrent index operations.

    Uses atomic O_CREAT|O_EXCL to avoid TOCTOU race conditions.
    """
    lock_dir = os.path.join(repo_root, STATE_DIR)
    os.makedirs(lock_dir, exist_ok=True)
    lock_path = os.path.join(lock_dir, LOCK_FILE)

    # Attempt atomic lock acquisition
    try:
        fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        os.write(fd, str(os.getpid()).encode())
        os.close(fd)
    except FileExistsError:
        # Lock file exists — check if holder is still alive
        try:
            with open(lock_path) as f:
                old_pid = int(f.read().strip())
            if _is_pid_alive(old_pid):
                raise RuntimeError(
                    f"Another indexing process is already running (PID {old_pid})"
                )
            logger.info(f"Removing stale lock from dead PID {old_pid}")
        except (ValueError, OSError):
            logger.info("Removing stale lock file (invalid contents)")
        # Stale lock — remove and retry atomically
        try:
            os.remove(lock_path)
        except OSError:
            pass
        fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        os.write(fd, str(os.getpid()).encode())
        os.close(fd)

    try:
        yield
    finally:
        try:
            os.remove(lock_path)
        except OSError:
            pass


@dataclass
class IndexState:
    last_indexed_commit: str = ""
    last_indexed_at: str = ""
    last_indexed_branch: str = ""  # track branch for switch detection
    total_symbols: int = 0
    total_chunks: int = 0
    total_files: int = 0
    use_embeddings: bool = False
    file_checksums: dict[str, tuple[float, int]] = field(default_factory=dict)
    # mtime, size pairs

    def save(self, repo_root: str):
        """Save state atomically — write to temp file then rename to prevent corruption."""
        state_dir = os.path.join(repo_root, STATE_DIR)
        os.makedirs(state_dir, exist_ok=True)
        path = os.path.join(state_dir, STATE_FILE)
        fd, tmp_path = tempfile.mkstemp(dir=state_dir, suffix=".tmp")
        try:
            with os.fdopen(fd, "w") as f:
                json.dump(asdict(self), f, indent=2)
            os.replace(tmp_path, path)  # atomic on same filesystem
        except BaseException:
            try:
                os.remove(tmp_path)
            except OSError:
                pass
            raise

    @classmethod
    def load(cls, repo_root: str) -> "IndexState":
        path = os.path.join(repo_root, STATE_DIR, STATE_FILE)
        if not os.path.exists(path):
            return cls()
        with open(path) as f:
            data = json.load(f)
        state = cls()
        for k, v in data.items():
            if hasattr(state, k):
                setattr(state, k, v)
        return state


@dataclass
class IndexStats:
    files_scanned: int = 0
    files_indexed: int = 0
    files_skipped: int = 0
    files_deleted: int = 0
    symbols_indexed: int = 0
    chunks_indexed: int = 0
    embeddings_generated: int = 0
    elapsed_seconds: float = 0.0
    errors: list[str] = field(default_factory=list)


def _git_head_sha(repo_root: str) -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo_root, capture_output=True, text=True,
    )
    return result.stdout.strip() if result.returncode == 0 else ""


def _git_current_branch(repo_root: str) -> str:
    result = subprocess.run(
        ["git", "rev-parse", "--abbrev-ref", "HEAD"],
        cwd=repo_root, capture_output=True, text=True,
    )
    return result.stdout.strip() if result.returncode == 0 else ""


def _git_commit_exists(repo_root: str, sha: str) -> bool:
    """Check if a commit SHA still exists in the repo (survives gc, force-push)."""
    result = subprocess.run(
        ["git", "cat-file", "-t", sha],
        cwd=repo_root, capture_output=True, text=True,
    )
    return result.returncode == 0 and result.stdout.strip() == "commit"


def _git_diff_files(
    repo_root: str, from_sha: str, to_sha: str = "HEAD",
) -> list[tuple[str, str, str]]:
    """Get changed files between two commits. Returns (status, old_path, new_path)."""
    result = subprocess.run(
        ["git", "diff", "--name-status", "--find-renames=50%", from_sha, to_sha],
        cwd=repo_root, capture_output=True, text=True,
    )
    if result.returncode != 0:
        logger.warning(f"git diff failed: {result.stderr}")
        return []

    changes = []
    for line in result.stdout.strip().split("\n"):
        if not line.strip():
            continue
        parts = line.split("\t")
        status = parts[0][0]  # A, M, D, R (rename has Rxxx)
        if status == "R" and len(parts) >= 3:
            changes.append(("R", parts[1], parts[2]))
        elif len(parts) >= 2:
            changes.append((status, parts[1], parts[1]))
    return changes


def _git_ls_files(repo_root: str) -> list[str]:
    """List all tracked files."""
    result = subprocess.run(
        ["git", "ls-files"],
        cwd=repo_root, capture_output=True, text=True,
    )
    return result.stdout.strip().split("\n") if result.returncode == 0 else []


def _git_modified_unstaged(repo_root: str) -> list[str]:
    """Get modified but uncommitted files."""
    result = subprocess.run(
        ["git", "ls-files", "-m"],
        cwd=repo_root, capture_output=True, text=True,
    )
    return [f for f in result.stdout.strip().split("\n") if f.strip()]


def _get_mtime_size(file_path: str) -> tuple[float, int] | None:
    try:
        stat = os.stat(file_path)
        return (stat.st_mtime, stat.st_size)
    except OSError:
        return None


def _is_binary_file(file_path: str, check_bytes: int = 8192) -> bool:
    """Detect binary files by checking for null bytes in the first N bytes."""
    try:
        with open(file_path, "rb") as f:
            chunk = f.read(check_bytes)
        return b"\x00" in chunk
    except OSError:
        return True


def _is_indexable(file_path: str) -> bool:
    return Path(file_path).suffix.lower() in INDEXABLE_EXTENSIONS


def _symbol_to_doc(sym: Symbol) -> dict:
    return {
        "id": sym.id,
        "name": sym.name,
        "qualified_name": sym.qualified_name,
        "signature": sym.signature,
        "kind": sym.kind,
        "visibility": sym.visibility,
        "file_path": sym.file_path,
        "package": sym.package,
        "module": sym.module,
        "parent_class": sym.parent_class,
        "implements": sym.implements,
        "extends_class": sym.extends_class,
        "annotations": sym.annotations,
        "line_start": sym.line_start,
        "line_end": sym.line_end,
        "body_preview": sym.body_preview,
        "imports": sym.imports,
        "type_references": sym.type_references,
    }


def _chunk_to_doc(chunk: Chunk) -> dict:
    doc = {
        "id": chunk.id,
        "content": chunk.content,
        "file_path": chunk.file_path,
        "symbol_name": chunk.symbol_name,
        "kind": chunk.kind,
        "module": chunk.module,
        "language": chunk.language,
        "enclosing_class": chunk.enclosing_class,
        "enclosing_method": chunk.enclosing_method,
        "line_start": chunk.line_start,
        "line_end": chunk.line_end,
        "token_count": chunk.token_count,
    }
    if chunk.embedding:
        doc["embedding"] = chunk.embedding
    return doc


def _file_to_doc(fdoc: FileDoc) -> dict:
    return {
        "id": fdoc.id,
        "path": fdoc.path,
        "filename": fdoc.filename,
        "extension": fdoc.extension,
        "module": fdoc.module,
        "package": fdoc.package,
        "class_names": fdoc.class_names,
        "imports_summary": fdoc.imports_summary,
        "loc": fdoc.loc,
        "last_commit_sha": fdoc.last_commit_sha,
    }


def _derive_repo_prefix(repo_root: str) -> str:
    """Derive a safe collection name prefix from the repo directory name.

    e.g. C:\\azs\\repos\\OpenSearch → opensearch
         /home/user/my-cool-project → my_cool_project
    """
    name = os.path.basename(os.path.abspath(repo_root))
    # Sanitize: lowercase, replace non-alphanumeric with underscore
    import re
    prefix = re.sub(r"[^a-z0-9]", "_", name.lower()).strip("_")
    return prefix or "default"


def _collection_names(prefix: str) -> dict[str, str]:
    """Return namespaced collection names for a repo prefix."""
    return {
        "symbols": f"{prefix}_symbols",
        "chunks": f"{prefix}_chunks",
        "files": f"{prefix}_files",
    }


def _ensure_collections(client: JigyasaClient, use_embeddings: bool, prefix: str = ""):
    """Ensure collections exist — try reopen first, then create if needed."""
    try:
        health = client.health()
        existing = {c["name"] for c in health["collections"]}
    except Exception:
        existing = set()

    names = (
        _collection_names(prefix) if prefix
        else {"symbols": "symbols", "chunks": "chunks", "files": "files"}
    )
    for logical, actual in names.items():
        if actual not in existing:
            # Try to reopen persisted collection first (survives Jigyasa restart)
            try:
                client.open_collection(actual)
                logger.info(f"Reopened persisted collection: {actual}")
                continue
            except Exception:
                pass
            # Doesn't exist at all — create new
            schema = get_schema_json(logical, use_embeddings=use_embeddings)
            client.create_collection(actual, schema)
            logger.info(f"Created collection: {actual}")


class Indexer:
    """Main indexing pipeline for a Git repository.

    Supports multi-repo via namespaced collections. Each repo gets its own
    set of collections: {prefix}_symbols, {prefix}_chunks, {prefix}_files.
    """

    def __init__(
        self,
        repo_root: str,
        endpoint: str = "localhost:50051",
        use_embeddings: bool = False,
        repo_prefix: str = "",
        auto_install_grammars: bool = False,
    ):
        self.repo_root = os.path.abspath(repo_root)
        self.client = JigyasaClient(endpoint=endpoint, timeout=30.0)
        self.use_embeddings = use_embeddings and embeddings_available()
        self.java_chunker = JavaChunker()
        self.text_chunker = TextChunker()
        self.lang_registry = get_registry(auto_install=auto_install_grammars)
        self.prefix = repo_prefix or _derive_repo_prefix(repo_root)
        self.collections = _collection_names(self.prefix)

    def full_index(self) -> IndexStats:
        """Full reindex of the entire repository."""
        with _file_lock(self.repo_root):
            return self._full_index_impl()

    def _full_index_impl(self) -> IndexStats:
        start = time.time()
        stats = IndexStats()
        state = IndexState(use_embeddings=self.use_embeddings)

        _ensure_collections(self.client, self.use_embeddings, self.prefix)

        all_files = _git_ls_files(self.repo_root)
        head_sha = _git_head_sha(self.repo_root)

        # Auto-detect languages and install missing grammars
        self._auto_detect_and_install_grammars(all_files)

        symbol_batch: list[dict] = []
        chunk_batch: list[Chunk] = []
        file_batch: list[dict] = []

        for rel_path in all_files:
            abs_path = os.path.join(self.repo_root, rel_path)
            stats.files_scanned += 1

            if not _is_indexable(rel_path) or should_skip_file(rel_path):
                stats.files_skipped += 1
                continue

            if _is_binary_file(abs_path):
                stats.files_skipped += 1
                continue

            try:
                with open(abs_path, encoding="utf-8", errors="replace") as f:
                    source = f.read()
            except Exception as e:
                stats.errors.append(f"Read error {rel_path}: {e}")
                continue

            if rel_path.endswith(".java"):
                symbols, chunks, file_doc = self.java_chunker.parse_file(
                    abs_path, source, self.repo_root, head_sha
                )
                symbol_batch.extend([_symbol_to_doc(s) for s in symbols])
                chunk_batch.extend(chunks)
            else:
                result = self._try_ast_parse(
                    abs_path, source, head_sha,
                )
                if result is not None:
                    symbols, chunks, file_doc = result
                    symbol_batch.extend(
                        [_symbol_to_doc(s) for s in symbols]
                    )
                    chunk_batch.extend(chunks)
                else:
                    chunks, file_doc = self.text_chunker.chunk_file(
                        abs_path, source, self.repo_root, head_sha
                    )
                    chunk_batch.extend(chunks)

            file_batch.append(_file_to_doc(file_doc))

            # Track file checksums
            mts = _get_mtime_size(abs_path)
            if mts:
                state.file_checksums[rel_path] = mts

            stats.files_indexed += 1

            # Flush batches
            if len(symbol_batch) >= SYMBOL_BATCH_SIZE:
                self.client.index_batch(self.collections["symbols"], symbol_batch)
                stats.symbols_indexed += len(symbol_batch)
                symbol_batch = []

            if len(chunk_batch) >= CHUNK_BATCH_SIZE:
                self._flush_chunks(chunk_batch, stats)
                chunk_batch = []

            if len(file_batch) >= FILE_BATCH_SIZE:
                self.client.index_batch(self.collections["files"], file_batch)
                file_batch = []

        # Flush remaining
        if symbol_batch:
            self.client.index_batch(self.collections["symbols"], symbol_batch)
            stats.symbols_indexed += len(symbol_batch)
        if chunk_batch:
            self._flush_chunks(chunk_batch, stats)
        if file_batch:
            self.client.index_batch(self.collections["files"], file_batch)

        # Save state
        state.last_indexed_commit = head_sha
        state.last_indexed_branch = _git_current_branch(self.repo_root)
        state.last_indexed_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        state.total_symbols = stats.symbols_indexed
        state.total_chunks = stats.chunks_indexed
        state.total_files = stats.files_indexed
        state.save(self.repo_root)

        stats.elapsed_seconds = time.time() - start
        logger.info(
            f"Full index complete: {stats.files_indexed} files, "
            f"{stats.symbols_indexed} symbols, {stats.chunks_indexed} chunks "
            f"in {stats.elapsed_seconds:.1f}s"
        )
        return stats

    def incremental_index(self) -> IndexStats:
        """Incremental index: only re-index changed files."""
        with _file_lock(self.repo_root):
            return self._incremental_index_impl()

    def _incremental_index_impl(self) -> IndexStats:
        start = time.time()
        stats = IndexStats()
        state = IndexState.load(self.repo_root)
        head_sha = _git_head_sha(self.repo_root)
        current_branch = _git_current_branch(self.repo_root)

        # Safety: if last indexed commit no longer exists (gc, force-push,
        # shallow clone), fall back to full reindex.
        if state.last_indexed_commit and not _git_commit_exists(
            self.repo_root, state.last_indexed_commit
        ):
            logger.warning(
                f"Last indexed commit {state.last_indexed_commit[:12]} no longer "
                f"exists. Falling back to full reindex."
            )
            return self._full_index_impl()

        if current_branch != state.last_indexed_branch and state.last_indexed_branch:
            logger.info(
                f"Branch switch detected: {state.last_indexed_branch} → {current_branch}"
            )

        _ensure_collections(self.client, self.use_embeddings, self.prefix)

        # Collect all changed file paths
        changed_files: dict[str, str] = {}  # rel_path → status

        # 1. Git diff for committed changes
        if state.last_indexed_commit:
            if state.last_indexed_commit != head_sha:
                for status, old_path, new_path in _git_diff_files(
                    self.repo_root, state.last_indexed_commit, head_sha
                ):
                    if status == "D":
                        changed_files[old_path] = "D"
                    elif status == "R":
                        changed_files[old_path] = "D"
                        changed_files[new_path] = "M"
                    else:
                        changed_files[new_path] = status

        # 2. Uncommitted changes
        for rel_path in _git_modified_unstaged(self.repo_root):
            if rel_path not in changed_files:
                changed_files[rel_path] = "M"

        # 3. Filesystem mtime/size check for anything git missed
        for rel_path, cached_cs in state.file_checksums.items():
            if rel_path in changed_files:
                continue
            abs_path = os.path.join(self.repo_root, rel_path)
            current = _get_mtime_size(abs_path)
            if current is None:
                changed_files[rel_path] = "D"
            elif current != tuple(cached_cs):
                changed_files[rel_path] = "M"

        if not changed_files:
            logger.info("No changes detected — index is up to date")
            stats.elapsed_seconds = time.time() - start
            return stats

        logger.info(f"Incremental index: {len(changed_files)} files changed")

        symbol_batch: list[dict] = []
        chunk_batch: list[Chunk] = []
        file_batch: list[dict] = []

        for rel_path, status in changed_files.items():
            stats.files_scanned += 1

            if status == "D":
                # Delete all docs for this file from all collections
                for collection in self.collections.values():
                    try:
                        self.client.delete_by_query(
                            collection,
                            [{"field": "file_path", "value": rel_path}],
                        )
                    except Exception as e:
                        stats.errors.append(f"Delete error {rel_path}/{collection}: {e}")
                state.file_checksums.pop(rel_path, None)
                stats.files_deleted += 1
                continue

            abs_path = os.path.join(self.repo_root, rel_path)
            if not os.path.exists(abs_path):
                continue
            if not _is_indexable(rel_path) or should_skip_file(rel_path):
                stats.files_skipped += 1
                continue

            if _is_binary_file(abs_path):
                stats.files_skipped += 1
                continue

            # Delete old docs first (upsert = delete + insert)
            for collection in self.collections.values():
                try:
                    self.client.delete_by_query(
                        collection,
                        [{"field": "file_path", "value": rel_path}],
                    )
                except Exception:
                    pass

            try:
                with open(abs_path, encoding="utf-8", errors="replace") as f:
                    source = f.read()
            except Exception as e:
                stats.errors.append(f"Read error {rel_path}: {e}")
                continue

            if rel_path.endswith(".java"):
                symbols, chunks, file_doc = self.java_chunker.parse_file(
                    abs_path, source, self.repo_root, head_sha
                )
                symbol_batch.extend([_symbol_to_doc(s) for s in symbols])
                chunk_batch.extend(chunks)
            else:
                result = self._try_ast_parse(
                    abs_path, source, head_sha,
                )
                if result is not None:
                    symbols, chunks, file_doc = result
                    symbol_batch.extend(
                        [_symbol_to_doc(s) for s in symbols]
                    )
                    chunk_batch.extend(chunks)
                else:
                    chunks, file_doc = self.text_chunker.chunk_file(
                        abs_path, source, self.repo_root, head_sha
                    )
                    chunk_batch.extend(chunks)

            file_batch.append(_file_to_doc(file_doc))

            mts = _get_mtime_size(abs_path)
            if mts:
                state.file_checksums[rel_path] = mts

            stats.files_indexed += 1

        # Flush all batches
        if symbol_batch:
            self.client.index_batch(self.collections["symbols"], symbol_batch)
            stats.symbols_indexed += len(symbol_batch)
        if chunk_batch:
            self._flush_chunks(chunk_batch, stats)
        if file_batch:
            self.client.index_batch(self.collections["files"], file_batch)

        # Update state
        state.last_indexed_commit = head_sha
        state.last_indexed_branch = current_branch
        state.last_indexed_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        state.total_symbols += stats.symbols_indexed
        state.total_chunks += stats.chunks_indexed
        state.total_files += stats.files_indexed - stats.files_deleted
        state.save(self.repo_root)

        stats.elapsed_seconds = time.time() - start
        logger.info(
            f"Incremental index complete: {stats.files_indexed} files updated, "
            f"{stats.files_deleted} deleted in {stats.elapsed_seconds:.1f}s"
        )
        return stats

    def _flush_chunks(self, chunks: list[Chunk], stats: IndexStats):
        """Flush chunk batch, optionally generating embeddings."""
        if self.use_embeddings:
            texts = [c.content for c in chunks]
            try:
                vectors = embed_texts(texts, batch_size=EMBEDDING_BATCH_SIZE)
                for chunk, vec in zip(chunks, vectors, strict=True):
                    chunk.embedding = vec
                stats.embeddings_generated += len(vectors)
            except Exception as e:
                logger.warning(f"Embedding generation failed, indexing without: {e}")

    def _try_ast_parse(
        self,
        abs_path: str,
        source: str,
        commit_sha: str,
    ) -> tuple[list[Symbol], list[Chunk], FileDoc] | None:
        """Try to parse a file with GenericASTChunker via the language registry.

        Returns (symbols, chunks, file_doc) if a grammar is available,
        or None to fall back to TextChunker.
        """
        parser, profile = self.lang_registry.get_parser(abs_path)
        if parser is None or profile is None:
            return None
        try:
            chunker = GenericASTChunker(parser, profile)
            return chunker.parse_file(
                abs_path, source, self.repo_root, commit_sha,
            )
        except Exception as e:
            rel = os.path.relpath(abs_path, self.repo_root)
            logger.warning(
                f"AST parse failed for {rel} ({profile.name}), "
                f"falling back to text chunker: {e}"
            )
            return None

    def _auto_detect_and_install_grammars(
        self, file_paths: list[str],
    ) -> None:
        """Scan repo files to detect languages and auto-install missing grammars.

        This makes the MCP server self-adaptive — it inspects what languages
        exist in the repo and ensures grammars are available for AST parsing.
        """
        from collections import Counter
        from pathlib import Path as PurePath

        from jigyasa_mcp.indexer.lang_registry import ALL_PROFILES

        # Count files per extension
        ext_counts: Counter[str] = Counter()
        for fp in file_paths:
            ext = PurePath(fp).suffix.lower()
            if ext:
                ext_counts[ext] += 1

        # Find languages with files but no grammar installed
        missing = []
        for profile in ALL_PROFILES:
            file_count = sum(ext_counts.get(ext, 0) for ext in profile.extensions)
            if file_count == 0:
                continue
            # Check if grammar is available
            parser, _ = self.lang_registry.get_parser(
                f"test{profile.extensions[0]}",
            )
            if parser is None and profile.name not in self.lang_registry._load_failed:
                # Grammar not installed but not failed — means it wasn't tried
                pass
            if parser is None:
                missing.append((profile, file_count))

        if not missing:
            return

        # Log what we found
        for profile, count in missing:
            logger.info(
                f"Detected {count} {profile.name} files but grammar "
                f"not installed: {profile.pip_package}"
            )

        # Auto-install if enabled
        if self.lang_registry.auto_install:
            for profile, count in missing:
                logger.info(
                    f"Auto-installing {profile.pip_package} "
                    f"for {count} {profile.name} files..."
                )
                parser = self.lang_registry._install_and_load(profile)
                if parser:
                    self.lang_registry._parsers[profile.name] = parser
        else:
            names = [p.pip_package for p, _ in missing]
            logger.info(
                f"To enable AST parsing for these languages, run:\n"
                f"  pip install {' '.join(names)}\n"
                f"Or use: jigyasa-index --auto-install-grammars"
            )

    def get_status(self) -> dict:
        """Get current index status."""
        state = IndexState.load(self.repo_root)
        head_sha = _git_head_sha(self.repo_root)
        is_stale = state.last_indexed_commit != head_sha

        try:
            health = self.client.health()
            collections = health["collections"]
        except Exception:
            collections = []

        return {
            "last_indexed_commit": state.last_indexed_commit,
            "current_head": head_sha,
            "is_stale": is_stale,
            "last_indexed_at": state.last_indexed_at,
            "use_embeddings": state.use_embeddings,
            "totals": {
                "symbols": state.total_symbols,
                "chunks": state.total_chunks,
                "files": state.total_files,
            },
            "collections": collections,
            "languages": self.lang_registry.status(),
        }

