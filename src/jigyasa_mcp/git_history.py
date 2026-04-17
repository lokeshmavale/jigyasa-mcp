"""Git history tools: commit search, diff view, and file evolution tracking.

Provides the same core capabilities as Bluebird's code_history tool,
but runs locally against the git repo — no ADO/GitHub API needed.
"""

import logging
import re
import subprocess
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class CommitInfo:
    """A single git commit."""
    sha: str
    short_sha: str
    author_name: str
    author_email: str
    date: str  # ISO 8601
    subject: str
    body: str = ""
    files_changed: int = 0
    insertions: int = 0
    deletions: int = 0


@dataclass
class FileDiff:
    """A file-level diff within a commit."""
    path: str
    old_path: str  # different from path if renamed
    status: str  # A(dded), M(odified), D(eleted), R(enamed)
    diff: str = ""  # unified diff content


@dataclass
class CommitDiff:
    """Full diff for a single commit."""
    commit: CommitInfo
    files: list[FileDiff] = field(default_factory=list)


@dataclass
class FileHistoryEntry:
    """One change in a file's history."""
    commit: CommitInfo
    diff: str = ""  # unified diff for this file in this commit


def _run_git(
    repo_root: str, args: list[str], max_output: int = 500_000,
) -> str | None:
    """Run a git command and return stdout, or None on failure."""
    try:
        result = subprocess.run(
            ["git", "--no-pager", *args],
            cwd=repo_root,
            capture_output=True,
            text=True,
            timeout=30,
        )
        if result.returncode != 0:
            logger.warning(f"git {args[0]} failed: {result.stderr.strip()}")
            return None
        output = result.stdout
        if len(output) > max_output:
            output = output[:max_output] + "\n... (truncated)"
        return output
    except subprocess.TimeoutExpired:
        logger.warning(f"git {args[0]} timed out (30s)")
        return None
    except FileNotFoundError:
        logger.error("git not found on PATH")
        return None
    except (NotADirectoryError, OSError) as e:
        logger.warning(f"git failed — invalid repo path: {e}")
        return None


def _parse_log_entry(entry: str) -> CommitInfo | None:
    """Parse a single --format entry into CommitInfo."""
    # Format: SHA|SHORT|AUTHOR_NAME|AUTHOR_EMAIL|DATE|SUBJECT
    parts = entry.split("|", 5)
    if len(parts) < 6:
        return None
    return CommitInfo(
        sha=parts[0],
        short_sha=parts[1],
        author_name=parts[2],
        author_email=parts[3],
        date=parts[4],
        subject=parts[5].strip(),
    )


LOG_FORMAT = "%H|%h|%an|%ae|%aI|%s"


def search_commits(
    repo_root: str,
    query: str = "",
    author: str = "",
    since: str = "",
    until: str = "",
    file_path: str = "",
    max_results: int = 20,
) -> list[CommitInfo]:
    """Search git commits by keyword, author, date range, or file path.

    Args:
        query: Search text in commit messages (grep).
        author: Filter by author name or email.
        since: Start date (ISO 8601 or relative like '7 days ago').
        until: End date.
        file_path: Only commits that touched this file.
        max_results: Max commits to return.
    """
    args = ["log", f"--format={LOG_FORMAT}", f"-{max_results}"]

    if query:
        args.extend(["--grep", query, "-i"])
    if author:
        args.extend(["--author", author])
    if since:
        args.extend(["--since", since])
    if until:
        args.extend(["--until", until])
    if file_path:
        args.extend(["--", file_path])

    output = _run_git(repo_root, args)
    if not output:
        return []

    commits = []
    for line in output.strip().split("\n"):
        if not line.strip():
            continue
        commit = _parse_log_entry(line)
        if commit:
            commits.append(commit)
    return commits


def get_commit_diff(
    repo_root: str,
    sha: str,
    context_lines: int = 3,
) -> CommitDiff | None:
    """Get the full diff for a specific commit.

    Returns commit metadata + per-file diffs with unified diff content.
    """
    # Get commit metadata
    meta_output = _run_git(
        repo_root,
        ["show", "--format=" + LOG_FORMAT + "|%b", "--stat", "-q", sha],
    )
    if not meta_output:
        return None

    first_line = meta_output.strip().split("\n")[0]
    # Format: SHA|SHORT|AUTHOR|EMAIL|DATE|SUBJECT|BODY
    parts = first_line.split("|", 6)
    if len(parts) < 6:
        return None

    commit = CommitInfo(
        sha=parts[0],
        short_sha=parts[1],
        author_name=parts[2],
        author_email=parts[3],
        date=parts[4],
        subject=parts[5].strip(),
        body=parts[6].strip() if len(parts) > 6 else "",
    )

    # Get file-level changes
    name_status = _run_git(
        repo_root,
        ["diff-tree", "--no-commit-id", "-r",
         "--name-status", "--find-renames=50%", sha],
    )
    files: list[FileDiff] = []
    file_paths: list[str] = []
    if name_status:
        for line in name_status.strip().split("\n"):
            if not line.strip():
                continue
            fparts = line.split("\t")
            status = fparts[0][0]
            if status == "R" and len(fparts) >= 3:
                files.append(FileDiff(
                    path=fparts[2], old_path=fparts[1], status="R",
                ))
                file_paths.append(fparts[2])
            elif len(fparts) >= 2:
                files.append(FileDiff(
                    path=fparts[1], old_path=fparts[1], status=status,
                ))
                file_paths.append(fparts[1])

    # Get actual diffs
    diff_output = _run_git(
        repo_root,
        ["show", "--format=", f"-U{context_lines}", sha],
    )
    if diff_output:
        _attach_diffs_to_files(files, diff_output)

    commit.files_changed = len(files)
    return CommitDiff(commit=commit, files=files)


def _attach_diffs_to_files(files: list[FileDiff], diff_output: str):
    """Parse unified diff output and attach to corresponding FileDiff objects."""
    file_map = {}
    for f in files:
        file_map[f.path] = f
        if f.old_path != f.path:
            file_map[f.old_path] = f

    # Split by "diff --git" boundaries
    chunks = re.split(r"(?=^diff --git )", diff_output, flags=re.MULTILINE)
    for chunk in chunks:
        if not chunk.startswith("diff --git"):
            continue
        # Extract file path from "diff --git a/path b/path"
        match = re.match(r"diff --git a/(.*?) b/(.*?)$", chunk, re.MULTILINE)
        if match:
            b_path = match.group(2)
            a_path = match.group(1)
            target = file_map.get(b_path) or file_map.get(a_path)
            if target:
                target.diff = chunk.strip()


def get_file_history(
    repo_root: str,
    file_path: str,
    max_results: int = 15,
    include_diff: bool = True,
) -> list[FileHistoryEntry]:
    """Trace how a file evolved over time.

    Returns commits that modified the file, optionally with diffs.
    Follows renames automatically.
    """
    args = [
        "log", f"--format={LOG_FORMAT}",
        f"-{max_results}", "--follow",
    ]
    if include_diff:
        args.extend(["-p", "-U3"])
    args.extend(["--", file_path])

    output = _run_git(repo_root, args)
    if not output:
        return []

    entries: list[FileHistoryEntry] = []

    if include_diff:
        # Split by commit boundaries (SHA line)
        # Each section: metadata line, then optional diff
        sections = re.split(
            r"^([0-9a-f]{40}\|)",
            output,
            flags=re.MULTILINE,
        )
        # sections[0] is empty, then pairs of (sha_prefix, rest)
        i = 1
        while i < len(sections) - 1:
            sha_prefix = sections[i]
            rest = sections[i + 1]
            full_line = sha_prefix + rest.split("\n")[0]
            commit = _parse_log_entry(full_line)
            if commit:
                # Everything after the first line is the diff
                diff_lines = rest.split("\n")[1:]
                diff_text = "\n".join(diff_lines).strip()
                entries.append(FileHistoryEntry(
                    commit=commit, diff=diff_text,
                ))
            i += 2
    else:
        for line in output.strip().split("\n"):
            if not line.strip():
                continue
            commit = _parse_log_entry(line)
            if commit:
                entries.append(FileHistoryEntry(commit=commit))

    return entries


def format_commits(commits: list[CommitInfo]) -> str:
    """Format commit list for MCP tool response."""
    if not commits:
        return "No commits found."
    lines = []
    for c in commits:
        lines.append(
            f"  {c.short_sha}  {c.date[:10]}  "
            f"{c.author_name}  {c.subject}"
        )
    return f"Found {len(commits)} commits:\n" + "\n".join(lines)


def format_commit_diff(cd: CommitDiff) -> str:
    """Format a commit diff for MCP tool response."""
    c = cd.commit
    lines = [
        f"Commit: {c.sha}",
        f"Author: {c.author_name} <{c.author_email}>",
        f"Date:   {c.date}",
        "",
        f"    {c.subject}",
    ]
    if c.body:
        for body_line in c.body.split("\n"):
            lines.append(f"    {body_line}")
    lines.append("")
    lines.append(f"Changed {len(cd.files)} file(s):")
    for f in cd.files:
        status_label = {
            "A": "added", "M": "modified",
            "D": "deleted", "R": "renamed",
        }.get(f.status, f.status)
        if f.status == "R":
            lines.append(f"  [{status_label}] {f.old_path} → {f.path}")
        else:
            lines.append(f"  [{status_label}] {f.path}")
    lines.append("")

    # Include diffs (truncated per file)
    for f in cd.files:
        if f.diff:
            lines.append(f"--- {f.path} ---")
            # Limit diff to 200 lines per file
            diff_lines = f.diff.split("\n")
            if len(diff_lines) > 200:
                lines.extend(diff_lines[:200])
                lines.append(f"... ({len(diff_lines) - 200} more lines)")
            else:
                lines.extend(diff_lines)
            lines.append("")
    return "\n".join(lines)


def format_file_history(
    entries: list[FileHistoryEntry], file_path: str,
) -> str:
    """Format file history for MCP tool response."""
    if not entries:
        return f"No history found for {file_path}."
    lines = [f"History of {file_path} ({len(entries)} commits):"]
    for entry in entries:
        c = entry.commit
        lines.append(
            f"\n--- {c.short_sha}  {c.date[:10]}  "
            f"{c.author_name} ---"
        )
        lines.append(f"    {c.subject}")
        if entry.diff:
            diff_lines = entry.diff.split("\n")
            if len(diff_lines) > 100:
                lines.extend(diff_lines[:100])
                lines.append(
                    f"... ({len(diff_lines) - 100} more lines)"
                )
            else:
                lines.extend(diff_lines)
    return "\n".join(lines)
