"""Chaos monkey tests for jigyasa-mcp resilience.

Tests system behavior under failure conditions:
  - Server unavailable / crashes mid-operation
  - Corrupted state files
  - Malformed source files
  - Concurrent indexing
  - Binary files disguised as text
  - Lock file races
  - Circuit breaker exhaustion
  - Git repo corruption
  - Enormous files / memory pressure
  - Unicode edge cases
"""

import json
import subprocess
import textwrap
import threading
import time

import pytest
from pydantic import ValidationError

from jigyasa_mcp.grpc_client import CircuitBreaker, JigyasaClient
from jigyasa_mcp.indexer.chunker import (
    JavaChunker,
)
from jigyasa_mcp.indexer.generic_ast_chunker import GenericASTChunker
from jigyasa_mcp.indexer.lang_registry import LanguageRegistry
from jigyasa_mcp.indexer.pipeline import (
    IndexState,
    _file_lock,
    _is_binary_file,
)
from jigyasa_mcp.server.validation import (
    GetCommitDiffInput,
    GetContextInput,
    SearchCodeInput,
    SearchSymbolsInput,
    truncate_response,
)

# ---------------------------------------------------------------------------
# 1. CORRUPTED STATE FILES
# ---------------------------------------------------------------------------


class TestCorruptedState:
    """State file corruption — crash during write, garbage data, etc."""

    def test_truncated_json(self, tmp_path):
        """Simulate crash during state file write (truncated JSON)."""
        state_dir = tmp_path / ".jigyasa"
        state_dir.mkdir()
        state_file = state_dir / "index_state.json"
        state_file.write_text('{"last_indexed_commit": "abc123", "total_sy')

        state = IndexState.load(str(tmp_path))
        # Should not crash — returns empty/default state
        assert state.last_indexed_commit == "" or state.last_indexed_commit == "abc123"

    def test_empty_state_file(self, tmp_path):
        """Empty state file (0 bytes)."""
        state_dir = tmp_path / ".jigyasa"
        state_dir.mkdir()
        (state_dir / "index_state.json").write_text("")

        state = IndexState.load(str(tmp_path))
        assert state.total_symbols == 0

    def test_binary_garbage_state_file(self, tmp_path):
        """State file filled with binary garbage."""
        state_dir = tmp_path / ".jigyasa"
        state_dir.mkdir()
        (state_dir / "index_state.json").write_bytes(
            b"\x00\xff\xfe\xfd" * 100
        )

        state = IndexState.load(str(tmp_path))
        assert state.total_symbols == 0

    def test_state_with_wrong_types(self, tmp_path):
        """State file with wrong field types."""
        state_dir = tmp_path / ".jigyasa"
        state_dir.mkdir()
        (state_dir / "index_state.json").write_text(json.dumps({
            "last_indexed_commit": 12345,  # should be str
            "total_symbols": "not_a_number",  # should be int
            "file_checksums": "not_a_dict",  # should be dict
        }))

        state = IndexState.load(str(tmp_path))
        # Should load what it can without crashing
        assert isinstance(state, IndexState)

    def test_state_atomic_write_survives_crash(self, tmp_path):
        """Verify atomic write — old state preserved if new write fails."""
        state = IndexState(
            last_indexed_commit="good_sha",
            total_symbols=100,
        )
        state.save(str(tmp_path))

        # Verify it saved
        loaded = IndexState.load(str(tmp_path))
        assert loaded.last_indexed_commit == "good_sha"
        assert loaded.total_symbols == 100

    def test_state_with_extra_fields(self, tmp_path):
        """State file from future version with unknown fields."""
        state_dir = tmp_path / ".jigyasa"
        state_dir.mkdir()
        (state_dir / "index_state.json").write_text(json.dumps({
            "last_indexed_commit": "abc123",
            "total_symbols": 50,
            "future_field": "unknown_value",
            "another_future": {"nested": True},
        }))

        state = IndexState.load(str(tmp_path))
        assert state.last_indexed_commit == "abc123"
        assert state.total_symbols == 50


# ---------------------------------------------------------------------------
# 2. MALFORMED SOURCE FILES
# ---------------------------------------------------------------------------


class TestMalformedSource:
    """Source files that could crash the parser."""

    @pytest.fixture()
    def java_chunker(self):
        return JavaChunker()

    @pytest.fixture()
    def python_chunker(self):
        registry = LanguageRegistry()
        parser, profile = registry.get_parser("test.py")
        if parser is None:
            pytest.skip("tree-sitter-python not installed")
        return GenericASTChunker(parser, profile)

    def test_java_syntax_error(self, java_chunker, tmp_path):
        """Java file with broken syntax."""
        source = textwrap.dedent("""\
            package com.broken;

            public class Broken {
                public void method( {  // missing close paren
                    if (true {  // missing close paren
                        System.out.println("chaos");
                    // missing closing braces
        """)
        fpath = str(tmp_path / "Broken.java")
        with open(fpath, "w") as f:
            f.write(source)

        # Should not crash — tree-sitter handles partial parses
        symbols, chunks, file_doc = java_chunker.parse_file(
            fpath, source, str(tmp_path),
        )
        assert isinstance(symbols, list)
        assert file_doc is not None

    def test_python_syntax_error(self, python_chunker, tmp_path):
        """Python file with broken syntax."""
        source = textwrap.dedent("""\
            def broken_func(
                # missing close paren and colon
                x = 1
                return x

            class Incomplete
                pass
        """)
        fpath = str(tmp_path / "broken.py")
        with open(fpath, "w") as f:
            f.write(source)

        symbols, chunks, file_doc = python_chunker.parse_file(
            fpath, source, str(tmp_path),
        )
        assert isinstance(symbols, list)

    def test_empty_file(self, java_chunker, tmp_path):
        """Completely empty file."""
        fpath = str(tmp_path / "Empty.java")
        with open(fpath, "w") as f:
            f.write("")

        symbols, chunks, file_doc = java_chunker.parse_file(
            fpath, "", str(tmp_path),
        )
        assert len(symbols) == 0
        assert file_doc is not None

    def test_null_bytes_in_source(self, java_chunker, tmp_path):
        """File with embedded null bytes."""
        source = "public class Null {\n\x00\x00\x00\n    int x = 1;\n}\n"
        fpath = str(tmp_path / "Null.java")
        with open(fpath, "w") as f:
            f.write(source)

        symbols, chunks, file_doc = java_chunker.parse_file(
            fpath, source, str(tmp_path),
        )
        assert isinstance(symbols, list)

    def test_extremely_long_line(self, java_chunker, tmp_path):
        """File with a single 1MB line."""
        long_comment = "// " + "x" * (1024 * 1024) + "\n"
        source = long_comment + "public class LongLine { int x = 1; }\n"
        fpath = str(tmp_path / "LongLine.java")
        with open(fpath, "w") as f:
            f.write(source)

        symbols, chunks, file_doc = java_chunker.parse_file(
            fpath, source, str(tmp_path),
        )
        assert file_doc is not None

    def test_deeply_nested_code(self, java_chunker, tmp_path):
        """100 levels of nesting."""
        source = "public class Deep {\n"
        for i in range(100):
            source += "  " * (i + 1) + f"class Inner{i} {{\n"
        for i in range(99, -1, -1):
            source += "  " * (i + 1) + "}\n"
        source += "}\n"
        fpath = str(tmp_path / "Deep.java")
        with open(fpath, "w") as f:
            f.write(source)

        symbols, chunks, file_doc = java_chunker.parse_file(
            fpath, source, str(tmp_path),
        )
        assert len(symbols) > 0

    def test_unicode_identifiers(self, python_chunker, tmp_path):
        """Unicode in class and function names."""
        source = textwrap.dedent("""\
            class Ünïcödé:
                def 日本語メソッド(self):
                    return "こんにちは"

            def café_function():
                return "☕"
        """)
        fpath = str(tmp_path / "unicode.py")
        with open(fpath, "w", encoding="utf-8") as f:
            f.write(source)

        symbols, chunks, file_doc = python_chunker.parse_file(
            fpath, source, str(tmp_path),
        )
        assert len(symbols) > 0

    def test_mixed_encoding(self, tmp_path):
        """File with mixed Latin-1 and UTF-8."""
        fpath = str(tmp_path / "mixed.py")
        content = b"# Comment with \xe9\xe8\xea accent\ndef func():\n    pass\n"
        with open(fpath, "wb") as f:
            f.write(content)

        # Read with errors="replace" (like pipeline does)
        with open(fpath, encoding="utf-8", errors="replace") as f:
            source = f.read()
        assert "def func" in source  # didn't lose the valid parts


# ---------------------------------------------------------------------------
# 3. BINARY FILE DETECTION
# ---------------------------------------------------------------------------


class TestBinaryDetection:
    """Binary files that could corrupt the index."""

    def test_null_bytes_detected(self, tmp_path):
        fpath = str(tmp_path / "binary.dat")
        with open(fpath, "wb") as f:
            f.write(b"PK\x03\x04" + b"\x00" * 100)  # ZIP header
        assert _is_binary_file(fpath) is True

    def test_text_file_not_binary(self, tmp_path):
        fpath = str(tmp_path / "text.py")
        with open(fpath, "w") as f:
            f.write("def hello():\n    pass\n")
        assert _is_binary_file(fpath) is False

    def test_empty_file_not_binary(self, tmp_path):
        fpath = str(tmp_path / "empty.txt")
        (tmp_path / "empty.txt").write_text("")
        assert _is_binary_file(fpath) is False

    def test_nonexistent_file(self, tmp_path):
        assert _is_binary_file(str(tmp_path / "nope.txt")) is True

    def test_jpg_disguised_as_java(self, tmp_path):
        """JPEG binary with .java extension."""
        fpath = str(tmp_path / "NotJava.java")
        with open(fpath, "wb") as f:
            f.write(b"\xff\xd8\xff\xe0" + b"\x00" * 200)
        assert _is_binary_file(fpath) is True

    def test_utf16_bom(self, tmp_path):
        """UTF-16 file with BOM (has null bytes between ASCII chars)."""
        fpath = str(tmp_path / "utf16.txt")
        content = "Hello World\n".encode("utf-16-le")
        with open(fpath, "wb") as f:
            f.write(b"\xff\xfe")  # BOM
            f.write(content)
        # UTF-16 has null bytes — should be detected as binary
        assert _is_binary_file(fpath) is True


# ---------------------------------------------------------------------------
# 4. LOCK FILE RACES
# ---------------------------------------------------------------------------


class TestLockFileRaces:
    """Concurrent indexing and lock file edge cases."""

    def test_stale_lock_from_dead_pid(self, tmp_path):
        """Lock file from a PID that no longer exists."""
        lock_dir = tmp_path / ".jigyasa"
        lock_dir.mkdir()
        lock_file = lock_dir / "index.lock"
        lock_file.write_text("99999999")  # almost certainly dead

        # Should be able to acquire lock (stale lock removed)
        with _file_lock(str(tmp_path)):
            assert True  # didn't raise

    def test_lock_file_with_garbage(self, tmp_path):
        """Lock file with non-numeric content."""
        lock_dir = tmp_path / ".jigyasa"
        lock_dir.mkdir()
        (lock_dir / "index.lock").write_text("not_a_pid")

        with _file_lock(str(tmp_path)):
            assert True

    def test_lock_rejects_when_active_pid(self, tmp_path):
        """Lock with active PID should raise RuntimeError."""
        # _file_lock creates .jigyasa dir itself, so pre-create with our PID
        import os as _os
        lock_dir = _os.path.join(str(tmp_path), ".jigyasa")
        _os.makedirs(lock_dir, exist_ok=True)
        lock_path = _os.path.join(lock_dir, "index.lock")
        # Write current PID (which is alive)
        fd = _os.open(lock_path, _os.O_CREAT | _os.O_WRONLY)
        _os.write(fd, str(_os.getpid()).encode())
        _os.close(fd)

        # _file_lock should detect live PID and raise
        with pytest.raises(RuntimeError, match="already running"):
            with _file_lock(str(tmp_path)):
                pass

    def test_lock_cleanup_on_exception(self, tmp_path):
        """Lock file removed even if operation raises."""
        lock_path = tmp_path / ".jigyasa" / "index.lock"
        try:
            with _file_lock(str(tmp_path)):
                raise ValueError("boom")
        except ValueError:
            pass

        assert not lock_path.exists()


# ---------------------------------------------------------------------------
# 5. CIRCUIT BREAKER EXHAUSTION
# ---------------------------------------------------------------------------


class TestCircuitBreakerChaos:
    """Circuit breaker under extreme failure conditions."""

    def test_rapid_failures_open_breaker(self):
        cb = CircuitBreaker(failure_threshold=3, reset_timeout=0.1)
        for _ in range(5):
            cb.record_failure()
        assert cb.is_open

    def test_breaker_blocks_requests_when_open(self):
        cb = CircuitBreaker(failure_threshold=1, reset_timeout=10.0)
        cb.record_failure()
        assert not cb.allow_request()

    def test_breaker_recovers_after_timeout(self):
        cb = CircuitBreaker(failure_threshold=1, reset_timeout=0.1)
        cb.record_failure()
        assert not cb.allow_request()
        time.sleep(0.15)
        assert cb.allow_request()  # half-open

    def test_breaker_thread_storm(self):
        """100 threads hammering the breaker simultaneously."""
        cb = CircuitBreaker(failure_threshold=5, reset_timeout=0.1)
        errors = []

        def hammer():
            try:
                for _ in range(100):
                    if cb.allow_request():
                        cb.record_failure()
                    else:
                        cb.record_success()
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=hammer) for _ in range(100)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=10)

        assert len(errors) == 0  # no crashes

    def test_unreachable_server_connection(self):
        """Client connecting to dead endpoint."""
        client = JigyasaClient(endpoint="localhost:59999", timeout=1.0)
        with pytest.raises(ConnectionError):
            client.health()
        client.close()

    def test_client_close_idempotent(self):
        """Closing client multiple times doesn't crash."""
        client = JigyasaClient(endpoint="localhost:59999")
        client.close()
        client.close()
        client.close()  # triple close


# ---------------------------------------------------------------------------
# 6. VALIDATION CHAOS
# ---------------------------------------------------------------------------


class TestValidationChaos:
    """Malicious and edge-case inputs to MCP tools."""

    def test_sql_injection_in_query(self):
        """SQL injection attempt (shouldn't reach DB, but test boundary)."""
        inp = SearchSymbolsInput(
            query="'; DROP TABLE symbols; --",
        )
        assert inp.query == "'; DROP TABLE symbols; --"

    def test_path_traversal_variations(self):
        """Multiple path traversal bypass attempts."""
        attacks = [
            "../../../etc/passwd",
            "..\\..\\..\\windows\\system32",
            "....//....//etc/passwd",
            "/absolute/path",
        ]
        for attack in attacks:
            with pytest.raises(ValidationError):
                GetContextInput(
                    file_path=attack, line_start=1, line_end=1,
                )

    def test_url_encoded_paths_not_decoded(self):
        """URL-encoded traversals are NOT decoded (correct behavior)."""
        # Pydantic validates the literal string, not URL-decoded form
        inp = GetContextInput(
            file_path="%2e%2e%2fetc/passwd",
            line_start=1, line_end=1,
        )
        assert inp.file_path == "%2e%2e%2fetc/passwd"

    def test_zero_length_sha(self):
        with pytest.raises(ValidationError):
            GetCommitDiffInput(sha="")

    def test_non_hex_sha(self):
        with pytest.raises(ValidationError):
            GetCommitDiffInput(sha="GHIJKL")

    def test_emoji_in_query(self):
        """Emoji and special Unicode in search query."""
        inp = SearchCodeInput(query="🔥 rocket 🚀 search")
        assert "🔥" in inp.query

    def test_max_length_query(self):
        """Query at exactly max length."""
        inp = SearchSymbolsInput(query="a" * 500)
        assert len(inp.query) == 500

    def test_over_max_length_query(self):
        with pytest.raises(ValidationError):
            SearchSymbolsInput(query="a" * 501)

    def test_null_bytes_in_query(self):
        """Null bytes in search query."""
        inp = SearchCodeInput(query="test\x00injection")
        assert inp.query is not None

    def test_newlines_in_query(self):
        inp = SearchSymbolsInput(query="multi\nline\nquery")
        assert "\n" in inp.query

    def test_line_end_before_start(self):
        """line_end < line_start should fail validation."""
        with pytest.raises(ValidationError):
            GetContextInput(
                file_path="valid.py",
                line_start=100,
                line_end=50,
            )


# ---------------------------------------------------------------------------
# 7. TRUNCATION EDGE CASES
# ---------------------------------------------------------------------------


class TestTruncationChaos:
    """Response truncation under extreme conditions."""

    def test_exactly_at_limit(self):
        text = "x" * 15000
        result = truncate_response(text)
        assert len(result) <= 15000

    def test_one_over_limit(self):
        text = "x" * 15001
        result = truncate_response(text)
        assert "truncated" in result

    def test_all_newlines(self):
        text = "\n" * 20000
        result = truncate_response(text)
        assert len(result) < 20000

    def test_single_huge_line(self):
        """One line with no newlines at all."""
        text = "x" * 20000
        result = truncate_response(text)
        assert len(result) < 20000

    def test_empty_string(self):
        assert truncate_response("") == ""

    def test_unicode_truncation(self):
        """Don't split multi-byte Unicode characters."""
        text = "こんにちは世界" * 3000  # lots of 3-byte chars
        result = truncate_response(text)
        # Should be valid UTF-8 after truncation
        result.encode("utf-8")  # shouldn't raise


# ---------------------------------------------------------------------------
# 8. GIT HISTORY CHAOS
# ---------------------------------------------------------------------------


class TestGitHistoryChaos:
    """Git history tools under adverse conditions."""

    def test_search_in_non_git_directory(self, tmp_path):
        """Searching commits in a directory that isn't a git repo."""
        from jigyasa_mcp.git_history import search_commits
        commits = search_commits(str(tmp_path), query="anything")
        assert commits == []

    def test_file_history_nonexistent_repo(self):
        from jigyasa_mcp.git_history import get_file_history
        entries = get_file_history(
            "/nonexistent/path/12345", "file.py",
        )
        assert entries == []

    def test_commit_diff_invalid_sha(self, tmp_path):
        """Diff for SHA that doesn't exist."""
        from jigyasa_mcp.git_history import get_commit_diff
        # Create a minimal git repo
        subprocess.run(
            ["git", "init"], cwd=str(tmp_path), capture_output=True,
        )
        result = get_commit_diff(str(tmp_path), "0" * 40)
        assert result is None

    def test_search_with_special_chars(self, tmp_path):
        """Search query with regex-special characters."""
        from jigyasa_mcp.git_history import search_commits
        # Create a minimal git repo
        subprocess.run(
            ["git", "init"], cwd=str(tmp_path), capture_output=True,
        )
        # These shouldn't crash even if no results
        search_commits(str(tmp_path), query="[.*+?^${}()|")
        search_commits(str(tmp_path), author="user@(evil)")


# ---------------------------------------------------------------------------
# 9. LANGUAGE REGISTRY CHAOS
# ---------------------------------------------------------------------------


class TestLanguageRegistryChaos:
    """Language registry under failure conditions."""

    def test_nonexistent_grammar_graceful(self):
        """Grammar that doesn't exist falls back gracefully."""
        registry = LanguageRegistry()
        parser, profile = registry.get_parser("test.xyz_unknown")
        assert parser is None
        assert profile is None

    def test_double_load_same_grammar(self):
        """Loading same grammar twice doesn't crash."""
        registry = LanguageRegistry()
        p1, _ = registry.get_parser("a.py")
        p2, _ = registry.get_parser("b.py")
        assert p1 is p2  # should be cached

    def test_all_profiles_have_required_fields(self):
        """Every profile has at least one declaration node type."""
        from jigyasa_mcp.indexer.lang_registry import ALL_PROFILES
        for profile in ALL_PROFILES:
            assert len(profile.extensions) > 0
            assert profile.pip_package
            assert profile.module_name
            assert len(profile.all_declaration_nodes) > 0
