"""Unit tests for the indexing pipeline — state management and change detection."""

import json
import os

import pytest

from jigyasa_mcp.indexer.pipeline import (
    IndexState,
    _is_indexable,
)


class TestIndexState:
    def test_save_and_load(self, tmp_path):
        state = IndexState(
            last_indexed_commit="abc123" * 7,  # 42 chars
            last_indexed_at="2026-04-10T17:00:00Z",
            total_symbols=100,
            total_chunks=200,
            total_files=50,
            file_checksums={"foo.java": (1234.0, 5678)},
        )
        state.save(str(tmp_path))
        loaded = IndexState.load(str(tmp_path))

        assert loaded.last_indexed_commit == state.last_indexed_commit
        assert loaded.total_symbols == 100
        assert loaded.total_chunks == 200
        assert loaded.file_checksums["foo.java"] == [1234.0, 5678]

    def test_load_missing_returns_empty(self, tmp_path):
        state = IndexState.load(str(tmp_path / "nonexistent"))
        assert state.last_indexed_commit == ""
        assert state.total_symbols == 0

    def test_save_creates_directory(self, tmp_path):
        state = IndexState(last_indexed_commit="test")
        state.save(str(tmp_path / "deep" / "nested"))
        assert os.path.exists(tmp_path / "deep" / "nested" / ".jigyasa" / "index_state.json")

    def test_ignores_unknown_fields(self, tmp_path):
        """Forward compatibility: unknown fields in state file shouldn't crash."""
        state_dir = tmp_path / ".jigyasa"
        state_dir.mkdir()
        state_file = state_dir / "index_state.json"
        state_file.write_text(json.dumps({
            "last_indexed_commit": "abc",
            "future_field": "should be ignored",
            "total_symbols": 42,
        }))
        loaded = IndexState.load(str(tmp_path))
        assert loaded.last_indexed_commit == "abc"
        assert loaded.total_symbols == 42


class TestIsIndexable:
    @pytest.mark.parametrize("path,expected", [
        ("Foo.java", True),
        ("build.gradle", True),
        ("config.yml", True),
        ("schema.json", True),
        ("README.md", True),
        ("image.png", False),
        ("archive.tar.gz", False),
        ("binary.exe", False),
        ("Foo.class", False),
    ])
    def test_extension_filter(self, path, expected):
        assert _is_indexable(path) == expected
