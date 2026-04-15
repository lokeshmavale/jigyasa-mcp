"""Tests for git history tools."""

import subprocess
import textwrap

import pytest

from jigyasa_mcp.git_history import (
    format_commit_diff,
    format_commits,
    format_file_history,
    get_commit_diff,
    get_file_history,
    search_commits,
)


@pytest.fixture()
def git_repo(tmp_path):
    """Create a temporary git repo with a few commits."""
    repo = tmp_path / "test_repo"
    repo.mkdir()

    def run_git(*args):
        subprocess.run(
            ["git", *args], cwd=str(repo),
            capture_output=True, check=True,
        )

    # Init repo
    run_git("init")
    run_git("config", "user.email", "test@example.com")
    run_git("config", "user.name", "Test User")

    # Commit 1: initial file
    (repo / "hello.py").write_text(textwrap.dedent("""\
        def greet(name):
            return f"Hello, {name}!"
    """))
    run_git("add", "hello.py")
    run_git("commit", "-m", "feat: add greeting function")

    # Commit 2: add another file
    (repo / "math.py").write_text(textwrap.dedent("""\
        def add(a, b):
            return a + b
    """))
    run_git("add", "math.py")
    run_git("commit", "-m", "feat: add math utilities")

    # Commit 3: modify first file
    (repo / "hello.py").write_text(textwrap.dedent("""\
        def greet(name):
            return f"Hello, {name}!"

        def farewell(name):
            return f"Goodbye, {name}!"
    """))
    run_git("add", "hello.py")
    run_git("commit", "-m", "feat: add farewell function")

    return str(repo)


class TestSearchCommits:
    def test_finds_all_commits(self, git_repo):
        commits = search_commits(git_repo)
        assert len(commits) == 3

    def test_search_by_keyword(self, git_repo):
        commits = search_commits(git_repo, query="greeting")
        assert len(commits) == 1
        assert "greeting" in commits[0].subject

    def test_search_by_author(self, git_repo):
        commits = search_commits(git_repo, author="Test User")
        assert len(commits) == 3

    def test_search_by_file(self, git_repo):
        commits = search_commits(git_repo, file_path="math.py")
        assert len(commits) == 1
        assert "math" in commits[0].subject

    def test_search_no_results(self, git_repo):
        commits = search_commits(git_repo, query="nonexistent-xyz")
        assert len(commits) == 0

    def test_limit_results(self, git_repo):
        commits = search_commits(git_repo, max_results=2)
        assert len(commits) == 2

    def test_commit_fields(self, git_repo):
        commits = search_commits(git_repo, max_results=1)
        c = commits[0]
        assert len(c.sha) == 40
        assert len(c.short_sha) >= 7
        assert c.author_name == "Test User"
        assert c.author_email == "test@example.com"
        assert c.date  # ISO 8601 format
        assert c.subject


class TestGetCommitDiff:
    def test_returns_diff(self, git_repo):
        commits = search_commits(git_repo, query="farewell")
        assert len(commits) == 1
        cd = get_commit_diff(git_repo, commits[0].sha)
        assert cd is not None
        assert cd.commit.sha == commits[0].sha
        assert len(cd.files) > 0

    def test_diff_contains_changes(self, git_repo):
        commits = search_commits(git_repo, query="farewell")
        cd = get_commit_diff(git_repo, commits[0].sha)
        file_paths = [f.path for f in cd.files]
        assert "hello.py" in file_paths
        # The diff should contain the added function
        hello_diff = next(f for f in cd.files if f.path == "hello.py")
        assert "farewell" in hello_diff.diff

    def test_invalid_sha_returns_none(self, git_repo):
        result = get_commit_diff(git_repo, "0" * 40)
        assert result is None

    def test_short_sha(self, git_repo):
        commits = search_commits(git_repo, max_results=1)
        cd = get_commit_diff(git_repo, commits[0].short_sha)
        assert cd is not None


class TestGetFileHistory:
    def test_returns_history(self, git_repo):
        entries = get_file_history(git_repo, "hello.py")
        assert len(entries) == 2  # created + modified

    def test_history_order(self, git_repo):
        entries = get_file_history(git_repo, "hello.py")
        # Most recent first
        assert "farewell" in entries[0].commit.subject
        assert "greeting" in entries[1].commit.subject

    def test_includes_diff(self, git_repo):
        entries = get_file_history(
            git_repo, "hello.py", include_diff=True,
        )
        assert entries[0].diff  # should have diff content

    def test_without_diff(self, git_repo):
        entries = get_file_history(
            git_repo, "hello.py", include_diff=False,
        )
        for entry in entries:
            assert not entry.diff

    def test_nonexistent_file(self, git_repo):
        entries = get_file_history(git_repo, "nonexistent.py")
        assert len(entries) == 0


class TestFormatting:
    def test_format_commits(self, git_repo):
        commits = search_commits(git_repo)
        text = format_commits(commits)
        assert "Found 3 commits" in text
        assert "Test User" in text

    def test_format_empty(self):
        text = format_commits([])
        assert "No commits found" in text

    def test_format_commit_diff(self, git_repo):
        commits = search_commits(git_repo, query="farewell")
        cd = get_commit_diff(git_repo, commits[0].sha)
        text = format_commit_diff(cd)
        assert "Commit:" in text
        assert "Author:" in text
        assert "hello.py" in text

    def test_format_file_history(self, git_repo):
        entries = get_file_history(git_repo, "hello.py")
        text = format_file_history(entries, "hello.py")
        assert "History of hello.py" in text
        assert "2 commits" in text
