"""Unit tests for input validation."""

import pytest
from pydantic import ValidationError

from jigyasa_mcp.server.validation import (
    GetContextInput,
    ReindexInput,
    SearchCodeInput,
    SearchSymbolsInput,
    truncate_response,
    validate_path_within_root,
)


class TestSearchSymbolsInput:
    def test_valid_input(self):
        inp = SearchSymbolsInput(query="ActionType", kind=["class"], limit=10)
        assert inp.query == "ActionType"
        assert inp.kind == ["class"]

    def test_invalid_kind(self):
        with pytest.raises(ValidationError, match="Invalid kind"):
            SearchSymbolsInput(query="test", kind=["bogus"])

    def test_invalid_visibility(self):
        with pytest.raises(ValidationError, match="Invalid visibility"):
            SearchSymbolsInput(query="test", visibility=["secret"])

    def test_empty_query_rejected(self):
        with pytest.raises(ValidationError):
            SearchSymbolsInput(query="")

    def test_query_too_long(self):
        with pytest.raises(ValidationError):
            SearchSymbolsInput(query="x" * 501)

    def test_limit_bounds(self):
        with pytest.raises(ValidationError):
            SearchSymbolsInput(query="test", limit=0)
        with pytest.raises(ValidationError):
            SearchSymbolsInput(query="test", limit=101)

    def test_multiple_kinds(self):
        inp = SearchSymbolsInput(query="test", kind=["class", "interface"])
        assert inp.kind == ["class", "interface"]


class TestSearchCodeInput:
    def test_defaults(self):
        inp = SearchCodeInput(query="retry logic")
        assert inp.exclude_tests is True
        assert inp.limit == 15

    def test_override_defaults(self):
        inp = SearchCodeInput(query="test", exclude_tests=False, limit=50)
        assert inp.exclude_tests is False
        assert inp.limit == 50


class TestGetContextInput:
    def test_valid(self):
        inp = GetContextInput(file_path="src/Foo.java", line_start=10, line_end=20)
        assert inp.radius == 10

    def test_path_traversal_dotdot(self):
        with pytest.raises(ValidationError, match="relative|traversal"):
            GetContextInput(file_path="../../../etc/passwd", line_start=1, line_end=1)

    def test_path_traversal_absolute(self):
        with pytest.raises(ValidationError, match="relative|traversal"):
            GetContextInput(file_path="/etc/passwd", line_start=1, line_end=1)

    def test_path_traversal_windows(self):
        with pytest.raises(ValidationError, match="relative|traversal"):
            GetContextInput(file_path="..\\..\\windows\\system32", line_start=1, line_end=1)

    def test_negative_line(self):
        with pytest.raises(ValidationError):
            GetContextInput(file_path="foo.java", line_start=-1, line_end=5)

    def test_radius_bounds(self):
        with pytest.raises(ValidationError):
            GetContextInput(file_path="foo.java", line_start=1, line_end=5, radius=101)


class TestReindexInput:
    def test_valid_modes(self):
        assert ReindexInput(mode="incremental").mode == "incremental"
        assert ReindexInput(mode="full").mode == "full"

    def test_invalid_mode(self):
        with pytest.raises(ValidationError, match="Invalid mode"):
            ReindexInput(mode="delete_everything")


class TestValidatePathWithinRoot:
    def test_valid_path(self, tmp_path):
        (tmp_path / "src" / "Foo.java").parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / "src" / "Foo.java").touch()
        result = validate_path_within_root("src/Foo.java", str(tmp_path))
        assert "Foo.java" in result

    def test_traversal_blocked(self, tmp_path):
        with pytest.raises(ValueError, match="traversal"):
            validate_path_within_root("../../etc/passwd", str(tmp_path))


class TestTruncateResponse:
    def test_short_text_unchanged(self):
        text = "Hello world"
        assert truncate_response(text) == text

    def test_long_text_truncated(self):
        text = "x\n" * 20000
        result = truncate_response(text, max_chars=100)
        assert len(result) < 300
        assert "TRUNCATED" in result
        assert "chars omitted" in result

    def test_truncates_at_newline(self):
        text = "line1\nline2\nline3\nline4\n" * 1000
        result = truncate_response(text, max_chars=50)
        assert "[TRUNCATED" in result
