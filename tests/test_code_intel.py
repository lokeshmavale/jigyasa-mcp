"""End-to-end tests for code intelligence functions with a mock gRPC client.

Tests dependency_graph, find_implementations, find_references with
controlled search results to verify:
  - Exact TermFilter usage (not BM25-only)
  - Edge type classification (EXTENDS/IMPLEMENTS/ANNOTATION/IMPORT)
  - Import-to-file resolution
  - Backward compatibility (old indexes without imports_full)
"""

from dataclasses import dataclass, field
from unittest.mock import MagicMock, patch

import pytest

from jigyasa_mcp.code_intel import (
    _classify_import,
    _resolve_import_to_file,
    dependency_graph,
    find_implementations,
    find_references,
)
from jigyasa_mcp.grpc_client import SearchHit, SearchResult


class MockJigyasaClient:
    """Mock client that returns controlled search results.

    Tracks query calls so tests can verify that TermFilters are used.
    """

    def __init__(self):
        self.query_log: list[dict] = []
        self._responses: dict[str, list[SearchResult]] = {}

    def add_response(self, collection: str, result: SearchResult):
        """Queue a response for a collection. Responses are returned FIFO."""
        self._responses.setdefault(collection, []).append(result)

    def query(
        self,
        collection: str,
        text_query: str = "",
        filters: list[dict] | None = None,
        top_k: int = 20,
        include_source: bool = True,
        vector: list[float] | None = None,
        vector_field: str = "embedding",
        text_weight: float = 0.5,
    ) -> SearchResult:
        self.query_log.append({
            "collection": collection,
            "text_query": text_query,
            "filters": filters or [],
            "top_k": top_k,
        })
        responses = self._responses.get(collection, [])
        if responses:
            return responses.pop(0)
        return SearchResult(total_hits=0, hits=[], latency_ms=0.0)


# ---------------------------------------------------------------------------
# Test _classify_import
# ---------------------------------------------------------------------------

class TestClassifyImport:
    def test_extends_detected(self):
        symbols = {
            "ClusterService": {
                "extends_class": "AbstractLifecycleComponent",
                "implements": "",
                "annotations": "PublicApi",
            }
        }
        assert _classify_import(
            "org.opensearch.common.lifecycle.AbstractLifecycleComponent",
            symbols,
        ) == "EXTENDS"

    def test_implements_detected(self):
        symbols = {
            "Dog": {
                "extends_class": "Animal",
                "implements": "Pet, Trainable",
                "annotations": "",
            }
        }
        assert _classify_import("com.example.Pet", symbols) == "IMPLEMENTS"
        assert _classify_import("com.example.Trainable", symbols) == "IMPLEMENTS"

    def test_annotation_detected(self):
        symbols = {
            "ClusterService": {
                "extends_class": "AbstractLifecycleComponent",
                "implements": "",
                "annotations": "PublicApi",
            }
        }
        assert _classify_import(
            "org.opensearch.common.annotation.PublicApi", symbols,
        ) == "ANNOTATION"

    def test_plain_import_fallback(self):
        symbols = {
            "ClusterService": {
                "extends_class": "AbstractLifecycleComponent",
                "implements": "",
                "annotations": "PublicApi",
            }
        }
        assert _classify_import(
            "java.util.Collections", symbols,
        ) == "IMPORT"

    def test_empty_symbols(self):
        assert _classify_import("java.util.Map", {}) == "IMPORT"


# ---------------------------------------------------------------------------
# Test _resolve_import_to_file
# ---------------------------------------------------------------------------

class TestResolveImportToFile:
    def test_resolves_by_package_filter(self):
        client = MockJigyasaClient()
        client.add_response("files", SearchResult(
            total_hits=1,
            hits=[SearchHit(
                score=1.0, doc_id="1",
                source={
                    "path": "server/src/main/java/org/opensearch/cluster/ClusterState.java",
                    "class_names": "ClusterState",
                    "package": "org.opensearch.cluster",
                },
            )],
        ))

        result = _resolve_import_to_file(
            client, "org.opensearch.cluster.ClusterState", "files",
        )
        assert result == "server/src/main/java/org/opensearch/cluster/ClusterState.java"

        # Verify it used a TermFilter on package
        assert len(client.query_log) == 1
        filters = client.query_log[0]["filters"]
        assert any(f["field"] == "package" for f in filters), (
            f"Expected TermFilter on 'package', got: {filters}"
        )

    def test_resolves_inner_class(self):
        """Setting.Property → resolves to Setting's file."""
        client = MockJigyasaClient()
        # First query (direct package match) returns nothing
        client.add_response("files", SearchResult(total_hits=0, hits=[]))
        # Second query (inner class fallback) finds it
        client.add_response("files", SearchResult(
            total_hits=1,
            hits=[SearchHit(
                score=1.0, doc_id="1",
                source={
                    "path": "server/src/.../Setting.java",
                    "class_names": "Setting",
                    "package": "org.opensearch.common.settings",
                },
            )],
        ))

        result = _resolve_import_to_file(
            client, "org.opensearch.common.settings.Setting.Property", "files",
        )
        assert result == "server/src/.../Setting.java"

    def test_returns_empty_for_wildcard(self):
        client = MockJigyasaClient()
        result = _resolve_import_to_file(client, "java.util.*", "files")
        assert result == ""
        assert len(client.query_log) == 0  # no queries made

    def test_returns_empty_for_empty_string(self):
        client = MockJigyasaClient()
        result = _resolve_import_to_file(client, "", "files")
        assert result == ""

    def test_returns_empty_when_not_found(self):
        client = MockJigyasaClient()
        client.add_response("files", SearchResult(total_hits=0, hits=[]))
        result = _resolve_import_to_file(
            client, "com.nonexistent.FakeClass", "files",
        )
        assert result == ""


# ---------------------------------------------------------------------------
# Test dependency_graph
# ---------------------------------------------------------------------------

class TestDependencyGraph:
    def _make_file_hit(self, path, classes, package, imports_full="", imports_summary=""):
        return SearchHit(
            score=1.0, doc_id=path,
            source={
                "path": path,
                "class_names": classes,
                "package": package,
                "imports_full": imports_full,
                "imports_summary": imports_summary,
            },
        )

    def _make_symbol_hit(self, name, kind, file_path, extends="", implements="", annotations=""):
        return SearchHit(
            score=1.0, doc_id=f"{file_path}::{kind}::{name}",
            source={
                "name": name,
                "qualified_name": f"org.example.{name}",
                "kind": kind,
                "file_path": file_path,
                "extends_class": extends,
                "implements": implements,
                "annotations": annotations,
                "type_references": "",
                "line_start": 1,
                "line_end": 50,
            },
        )

    def test_uses_term_filter_for_file_lookup(self):
        client = MockJigyasaClient()
        # Exact path filter response
        client.add_response("files", SearchResult(
            total_hits=1,
            hits=[self._make_file_hit(
                "src/Dog.java", "Dog", "com.example",
                imports_full="com.example.Animal",
            )],
        ))
        # Symbol query
        client.add_response("symbols", SearchResult(
            total_hits=1,
            hits=[self._make_symbol_hit(
                "Dog", "class", "src/Dog.java",
                extends="Animal",
            )],
        ))
        # Import resolution for com.example.Animal
        client.add_response("files", SearchResult(
            total_hits=1,
            hits=[self._make_file_hit("src/Animal.java", "Animal", "com.example")],
        ))
        # find_references for Dog class
        client.add_response("symbols", SearchResult(total_hits=0, hits=[]))

        result = dependency_graph(
            client, "src/Dog.java", "files", "symbols", depth=1,
        )

        assert result["target"] == "src/Dog.java"
        assert result["classes_defined"] == ["Dog"]
        assert len(result["depends_on"]) == 1
        assert result["depends_on"][0]["type"] == "EXTENDS"
        assert result["depends_on"][0]["import"] == "com.example.Animal"

        # Verify first query used TermFilter on path
        first_query = client.query_log[0]
        assert any(
            f["field"] == "path" for f in first_query["filters"]
        ), f"Expected TermFilter on 'path', got: {first_query['filters']}"

        # Verify symbol query used TermFilter on file_path
        symbol_query = client.query_log[1]
        assert any(
            f["field"] == "file_path" for f in symbol_query["filters"]
        ), f"Expected TermFilter on 'file_path', got: {symbol_query['filters']}"

    def test_backward_compat_imports_summary_fallback(self):
        """Old indexes without imports_full should fall back to imports_summary."""
        client = MockJigyasaClient()
        # File with only imports_summary (no imports_full)
        client.add_response("files", SearchResult(
            total_hits=1,
            hits=[SearchHit(
                score=1.0, doc_id="src/Old.java",
                source={
                    "path": "src/Old.java",
                    "class_names": "Old",
                    "package": "com.example",
                    "imports_summary": "com.example.Dep",
                    # No imports_full field at all
                },
            )],
        ))
        # Symbol query
        client.add_response("symbols", SearchResult(total_hits=0, hits=[]))
        # Import resolution
        client.add_response("files", SearchResult(total_hits=0, hits=[]))
        # find_references
        client.add_response("symbols", SearchResult(total_hits=0, hits=[]))

        result = dependency_graph(
            client, "src/Old.java", "files", "symbols",
        )

        assert "error" not in result
        assert len(result["depends_on"]) == 1
        assert result["depends_on"][0]["import"] == "com.example.Dep"

    def test_file_not_found(self):
        client = MockJigyasaClient()
        # Empty for exact filter
        client.add_response("files", SearchResult(total_hits=0, hits=[]))
        # Empty for BM25 fallback
        client.add_response("files", SearchResult(total_hits=0, hits=[]))

        result = dependency_graph(
            client, "nonexistent.java", "files", "symbols",
        )
        assert "error" in result

    def test_completeness_field_present(self):
        client = MockJigyasaClient()
        client.add_response("files", SearchResult(
            total_hits=1,
            hits=[self._make_file_hit("src/A.java", "A", "com.example")],
        ))
        client.add_response("symbols", SearchResult(total_hits=0, hits=[]))
        client.add_response("symbols", SearchResult(total_hits=0, hits=[]))

        result = dependency_graph(client, "src/A.java", "files", "symbols")
        assert result["completeness"] == "full"
        assert "depends_on_count" in result
        assert "depended_by_count" in result


# ---------------------------------------------------------------------------
# Test find_implementations
# ---------------------------------------------------------------------------

class TestFindImplementations:
    def test_finds_implementing_class(self):
        client = MockJigyasaClient()
        client.add_response("symbols", SearchResult(
            total_hits=2,
            hits=[
                SearchHit(score=1.0, doc_id="1", source={
                    "name": "AutoTaggingFilter",
                    "qualified_name": "com.example.AutoTaggingFilter",
                    "kind": "class",
                    "file_path": "src/AutoTaggingFilter.java",
                    "implements": "ActionFilter",
                    "extends_class": "",
                    "line_start": 5, "line_end": 100,
                }),
                SearchHit(score=0.8, doc_id="2", source={
                    "name": "ActionFilter",
                    "qualified_name": "com.example.ActionFilter",
                    "kind": "interface",
                    "file_path": "src/ActionFilter.java",
                    "implements": "",
                    "extends_class": "",
                    "line_start": 1, "line_end": 20,
                }),
            ],
        ))

        refs = find_implementations(client, "ActionFilter", "symbols")
        assert len(refs) == 1
        assert refs[0].name == "AutoTaggingFilter"
        assert refs[0].relationship == "implements"

    def test_finds_extending_class(self):
        client = MockJigyasaClient()
        client.add_response("symbols", SearchResult(
            total_hits=1,
            hits=[SearchHit(score=1.0, doc_id="1", source={
                "name": "Dog",
                "qualified_name": "com.example.Dog",
                "kind": "class",
                "file_path": "src/Dog.java",
                "implements": "",
                "extends_class": "Animal",
                "line_start": 1, "line_end": 30,
            })],
        ))

        refs = find_implementations(client, "Animal", "symbols")
        assert len(refs) == 1
        assert refs[0].relationship == "extends"

    def test_no_implementations(self):
        client = MockJigyasaClient()
        client.add_response("symbols", SearchResult(total_hits=0, hits=[]))
        refs = find_implementations(client, "NonExistent", "symbols")
        assert refs == []


# ---------------------------------------------------------------------------
# Test find_references
# ---------------------------------------------------------------------------

class TestFindReferences:
    def test_finds_type_reference(self):
        client = MockJigyasaClient()
        client.add_response("symbols", SearchResult(
            total_hits=1,
            hits=[SearchHit(score=1.0, doc_id="1", source={
                "name": "TransportAction",
                "qualified_name": "org.opensearch.TransportAction",
                "kind": "class",
                "file_path": "src/TransportAction.java",
                "type_references": "ClusterState, ActionRequest",
                "imports": "",
                "line_start": 1, "line_end": 200,
            })],
        ))

        refs = find_references(client, "ClusterState", "symbols")
        assert len(refs) == 1
        assert refs[0].name == "TransportAction"
        assert refs[0].relationship == "references"

    def test_skips_self_reference(self):
        client = MockJigyasaClient()
        client.add_response("symbols", SearchResult(
            total_hits=1,
            hits=[SearchHit(score=1.0, doc_id="1", source={
                "name": "ClusterState",
                "qualified_name": "org.opensearch.ClusterState",
                "kind": "class",
                "file_path": "src/ClusterState.java",
                "type_references": "ClusterState",
                "imports": "",
                "line_start": 1, "line_end": 50,
            })],
        ))

        refs = find_references(client, "ClusterState", "symbols")
        assert refs == []  # self-reference skipped

    def test_no_references(self):
        client = MockJigyasaClient()
        client.add_response("symbols", SearchResult(total_hits=0, hits=[]))
        refs = find_references(client, "Orphan", "symbols")
        assert refs == []
