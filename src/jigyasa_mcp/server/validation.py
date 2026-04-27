"""Input validation models for MCP tool arguments.

Uses Pydantic for strict validation of all tool inputs.
"""

import os

from pydantic import BaseModel, Field, field_validator, model_validator

# Valid enum values
VALID_SYMBOL_KINDS = frozenset({"class", "interface", "enum", "method", "constructor", "field"})
VALID_VISIBILITIES = frozenset({"public", "protected", "private", "package-private"})
VALID_REINDEX_MODES = frozenset({"incremental", "full"})

MAX_QUERY_LENGTH = 500
MAX_LIMIT = 100
MAX_RESPONSE_CHARS = 15_000  # ~3750 tokens, safe for LLM context


class SearchSymbolsInput(BaseModel):
    query: str = Field(..., min_length=1, max_length=MAX_QUERY_LENGTH)
    kind: list[str] | None = None
    visibility: list[str] | None = None
    package_prefix: str | None = None
    file_pattern: str | None = None
    extends_or_implements: str | None = None
    has_annotation: str | None = None
    limit: int = Field(default=30, ge=1, le=MAX_LIMIT)

    @field_validator("kind")
    @classmethod
    def validate_kind(cls, v):
        if v is not None:
            invalid = [k for k in v if k not in VALID_SYMBOL_KINDS]
            if invalid:
                raise ValueError(
                    f"Invalid kind(s): {invalid}. "
                    f"Valid: {sorted(VALID_SYMBOL_KINDS)}"
                )
        return v

    @field_validator("visibility")
    @classmethod
    def validate_visibility(cls, v):
        if v is not None:
            invalid = [vis for vis in v if vis not in VALID_VISIBILITIES]
            if invalid:
                raise ValueError(
                    f"Invalid visibility: {invalid}. "
                    f"Valid: {sorted(VALID_VISIBILITIES)}"
                )
        return v


class SearchCodeInput(BaseModel):
    query: str = Field(..., min_length=1, max_length=MAX_QUERY_LENGTH)
    file_types: list[str] | None = None
    module_path: str | None = None
    enclosing_class: str | None = None
    exclude_tests: bool = True
    limit: int = Field(default=15, ge=1, le=MAX_LIMIT)


class SearchFilesInput(BaseModel):
    query: str = Field(..., min_length=1, max_length=MAX_QUERY_LENGTH)
    extension: str | None = None
    module: str | None = None
    limit: int = Field(default=20, ge=1, le=MAX_LIMIT)


class GetContextInput(BaseModel):
    file_path: str = Field(..., min_length=1)
    line_start: int = Field(..., ge=1)
    line_end: int = Field(..., ge=1)
    radius: int = Field(default=10, ge=0, le=100)

    @field_validator("file_path")
    @classmethod
    def validate_no_traversal(cls, v):
        """Prevent path traversal attacks."""
        normalized = os.path.normpath(v).replace("\\", "/")
        if normalized.startswith("..") or normalized.startswith("/"):
            raise ValueError(
                "Invalid file_path: must be relative and within repo root"
            )
        # Block obvious traversal patterns
        if ".." in v.split("/") or ".." in v.split("\\"):
            raise ValueError("Path traversal detected in file_path")
        return v

    @model_validator(mode="after")
    def validate_line_range(self):
        if self.line_end < self.line_start:
            raise ValueError(
                f"line_end ({self.line_end}) must be >= line_start ({self.line_start})"
            )
        return self


class ReindexInput(BaseModel):
    mode: str = Field(default="incremental")

    @field_validator("mode")
    @classmethod
    def validate_mode(cls, v):
        if v not in VALID_REINDEX_MODES:
            raise ValueError(
                f"Invalid mode: {v}. Valid: {sorted(VALID_REINDEX_MODES)}"
            )
        return v


class SearchCommitsInput(BaseModel):
    query: str = Field(default="", max_length=MAX_QUERY_LENGTH)
    author: str = Field(default="", max_length=200)


class FindImplementationsInput(BaseModel):
    symbol_name: str = Field(
        ..., min_length=1, max_length=MAX_QUERY_LENGTH,
        description="Class or interface name to find implementations of",
    )


class FindReferencesInput(BaseModel):
    symbol_name: str = Field(
        ..., min_length=1, max_length=MAX_QUERY_LENGTH,
        description="Type/class name to find references to",
    )


class DependencyGraphInput(BaseModel):
    file_path: str = Field(
        ..., min_length=1, max_length=500,
        description="File path to analyze dependencies for",
    )
    depth: int = Field(
        default=1, ge=1, le=3,
        description="How many hops to traverse (1=direct, 2=transitive)",
    )
    since: str = Field(default="", max_length=50)
    until: str = Field(default="", max_length=50)
    file_path: str = Field(default="", max_length=500)
    limit: int = Field(default=20, ge=1, le=MAX_LIMIT)


class GetCommitDiffInput(BaseModel):
    sha: str = Field(..., min_length=4, max_length=40)
    context_lines: int = Field(default=3, ge=0, le=20)

    @field_validator("sha")
    @classmethod
    def validate_sha(cls, v):
        if not all(c in "0123456789abcdef" for c in v.lower()):
            raise ValueError("sha must be a hex string")
        return v


class GetFileHistoryInput(BaseModel):
    file_path: str = Field(..., min_length=1, max_length=500)
    limit: int = Field(default=15, ge=1, le=50)
    include_diff: bool = Field(default=True)


def validate_path_within_root(file_path: str, repo_root: str) -> str:
    """Resolve a file path and ensure it stays within repo_root.

    Returns the absolute path if safe, raises ValueError otherwise.
    """
    abs_root = os.path.realpath(repo_root)
    abs_path = os.path.realpath(os.path.join(repo_root, file_path))
    if not abs_path.startswith(abs_root + os.sep) and abs_path != abs_root:
        raise ValueError(
            f"Path traversal blocked: {file_path} resolves outside repo root"
        )
    return abs_path


def truncate_response(text: str, max_chars: int = MAX_RESPONSE_CHARS) -> str:
    """Truncate response text to fit LLM context window.

    Appends a notice with the number of omitted characters so agents
    know results are incomplete.
    """
    if len(text) <= max_chars:
        return text
    truncated = text[:max_chars]
    # Find last newline to avoid cutting mid-line
    last_nl = truncated.rfind("\n")
    if last_nl > max_chars * 0.8:
        truncated = truncated[:last_nl]
    omitted = len(text) - len(truncated)
    return (
        truncated
        + f"\n\n[TRUNCATED — {omitted} chars omitted. "
        f"Use narrower filters or lower limit to get complete results.]"
    )
