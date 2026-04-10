"""Indexing pipeline: full and incremental indexing of a Git repository."""

import json
import logging
import os
import subprocess
import sys
import time
from contextlib import contextmanager
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

from jigyasa_mcp.grpc_client import JigyasaClient
from jigyasa_mcp.indexer.chunker import (
    JavaChunker, TextChunker, Symbol, Chunk, FileDoc, should_skip_file,
)
from jigyasa_mcp.indexer.embeddings import is_available as embeddings_available, embed_texts
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


@contextmanager
def _file_lock(repo_root: str):
    """Cross-platform file lock to prevent concurrent index operations."""
    lock_path = os.path.join(repo_root, STATE_DIR, LOCK_FILE)
    os.makedirs(os.path.dirname(lock_path), exist_ok=True)

    if sys.platform == "win32":
        import msvcrt
        lock_fh = open(lock_path, "w")
        try:
            msvcrt.locking(lock_fh.fileno(), msvcrt.LK_NBLCK, 1)
        except OSError:
            lock_fh.close()
            raise RuntimeError("Another indexing process is already running")
        try:
            yield
        finally:
            try:
                msvcrt.locking(lock_fh.fileno(), msvcrt.LK_UNLCK, 1)
            except OSError:
                pass
            lock_fh.close()
            try:
                os.remove(lock_path)
            except OSError:
                pass
    else:
        import fcntl
        lock_fh = open(lock_path, "w")
        try:
            fcntl.flock(lock_fh, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            lock_fh.close()
            raise RuntimeError("Another indexing process is already running")
        try:
            yield
        finally:
            fcntl.flock(lock_fh, fcntl.LOCK_UN)
            lock_fh.close()
            try:
                os.remove(lock_path)
            except OSError:
                pass


@dataclass
class IndexState:
    last_indexed_commit: str = ""
    last_indexed_at: str = ""
    total_symbols: int = 0
    total_chunks: int = 0
    total_files: int = 0
    use_embeddings: bool = False
    file_checksums: dict[str, tuple[float, int]] = field(default_factory=dict)
    # mtime, size pairs

    def save(self, repo_root: str):
        state_dir = os.path.join(repo_root, STATE_DIR)
        os.makedirs(state_dir, exist_ok=True)
        path = os.path.join(state_dir, STATE_FILE)
        with open(path, "w") as f:
            json.dump(asdict(self), f, indent=2)

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


def _git_diff_files(repo_root: str, from_sha: str, to_sha: str = "HEAD") -> list[tuple[str, str, str]]:
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


def _get_mtime_size(file_path: str) -> Optional[tuple[float, int]]:
    try:
        stat = os.stat(file_path)
        return (stat.st_mtime, stat.st_size)
    except OSError:
        return None


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


def _ensure_collections(client: JigyasaClient, use_embeddings: bool):
    """Create collections if they don't exist."""
    try:
        health = client.health()
        existing = {c["name"] for c in health["collections"]}
    except Exception:
        existing = set()

    for name in ("symbols", "chunks", "files"):
        if name not in existing:
            schema = get_schema_json(name, use_embeddings=use_embeddings)
            client.create_collection(name, schema)
            logger.info(f"Created collection: {name}")


class Indexer:
    """Main indexing pipeline for a Git repository."""

    def __init__(
        self,
        repo_root: str,
        endpoint: str = "localhost:50051",
        use_embeddings: bool = False,
    ):
        self.repo_root = os.path.abspath(repo_root)
        self.client = JigyasaClient(endpoint=endpoint)
        self.use_embeddings = use_embeddings and embeddings_available()
        self.java_chunker = JavaChunker()
        self.text_chunker = TextChunker()

    def full_index(self) -> IndexStats:
        """Full reindex of the entire repository."""
        with _file_lock(self.repo_root):
            return self._full_index_impl()

    def _full_index_impl(self) -> IndexStats:
        start = time.time()
        stats = IndexStats()
        state = IndexState(use_embeddings=self.use_embeddings)

        _ensure_collections(self.client, self.use_embeddings)

        all_files = _git_ls_files(self.repo_root)
        head_sha = _git_head_sha(self.repo_root)

        symbol_batch: list[dict] = []
        chunk_batch: list[Chunk] = []
        file_batch: list[dict] = []

        for rel_path in all_files:
            abs_path = os.path.join(self.repo_root, rel_path)
            stats.files_scanned += 1

            if not _is_indexable(rel_path) or should_skip_file(rel_path):
                stats.files_skipped += 1
                continue

            try:
                with open(abs_path, "r", encoding="utf-8", errors="ignore") as f:
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
                self.client.index_batch("symbols", symbol_batch)
                stats.symbols_indexed += len(symbol_batch)
                symbol_batch = []

            if len(chunk_batch) >= CHUNK_BATCH_SIZE:
                self._flush_chunks(chunk_batch, stats)
                chunk_batch = []

            if len(file_batch) >= FILE_BATCH_SIZE:
                self.client.index_batch("files", file_batch)
                file_batch = []

        # Flush remaining
        if symbol_batch:
            self.client.index_batch("symbols", symbol_batch)
            stats.symbols_indexed += len(symbol_batch)
        if chunk_batch:
            self._flush_chunks(chunk_batch, stats)
        if file_batch:
            self.client.index_batch("files", file_batch)

        # Save state
        state.last_indexed_commit = head_sha
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

        _ensure_collections(self.client, self.use_embeddings)

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
                for collection in ("symbols", "chunks", "files"):
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

            # Delete old docs first (upsert = delete + insert)
            for collection in ("symbols", "chunks", "files"):
                try:
                    self.client.delete_by_query(
                        collection,
                        [{"field": "file_path", "value": rel_path}],
                    )
                except Exception:
                    pass

            try:
                with open(abs_path, "r", encoding="utf-8", errors="ignore") as f:
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
            self.client.index_batch("symbols", symbol_batch)
            stats.symbols_indexed += len(symbol_batch)
        if chunk_batch:
            self._flush_chunks(chunk_batch, stats)
        if file_batch:
            self.client.index_batch("files", file_batch)

        # Update state
        state.last_indexed_commit = head_sha
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
                for chunk, vec in zip(chunks, vectors):
                    chunk.embedding = vec
                stats.embeddings_generated += len(vectors)
            except Exception as e:
                logger.warning(f"Embedding generation failed, indexing without: {e}")

        docs = [_chunk_to_doc(c) for c in chunks]
        self.client.index_batch("chunks", docs)
        stats.chunks_indexed += len(chunks)

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
        }
