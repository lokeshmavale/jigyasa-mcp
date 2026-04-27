"""Unit tests for the hybrid code chunker."""

import os
import textwrap

import pytest

from jigyasa_mcp.indexer.chunker import (
    JavaChunker,
    TextChunker,
    _estimate_tokens,
    _extract_imports,
    _extract_module,
    _extract_package,
    should_skip_file,
)

# --- should_skip_file ---

class TestShouldSkipFile:
    def test_skips_build_dir(self):
        assert should_skip_file("build/classes/Foo.java") is True

    def test_skips_generated_dir(self):
        assert should_skip_file("generated/proto/Bar.java") is True

    def test_skips_class_files(self):
        assert should_skip_file("Foo.class") is True

    def test_skips_jar_files(self):
        assert should_skip_file("lib/foo.jar") is True

    def test_allows_normal_java(self):
        assert should_skip_file("server/src/main/java/Foo.java") is False

    def test_allows_gradle(self):
        assert should_skip_file("build.gradle") is False

    def test_skips_git_dir(self):
        assert should_skip_file(".git/config") is True

    def test_skips_idea_dir(self):
        assert should_skip_file(".idea/workspace.xml") is True


# --- Helper functions ---

class TestHelpers:
    def test_extract_package(self):
        source = "package org.opensearch.cluster;\n\nimport foo;"
        assert _extract_package(source) == "org.opensearch.cluster"

    def test_extract_package_missing(self):
        assert _extract_package("// no package") == ""

    def test_extract_imports(self):
        source = textwrap.dedent("""\
            import org.opensearch.cluster.ClusterState;
            import org.opensearch.action.ActionType;
            import java.util.List;
            import static org.junit.Assert.assertEquals;
        """)
        full, prefixes = _extract_imports(source)
        # Full paths preserved
        assert "org.opensearch.cluster.ClusterState" in full
        assert "org.opensearch.action.ActionType" in full
        assert "java.util.List" in full
        assert "org.junit.Assert.assertEquals" in full
        # Backward-compat prefixes still work
        assert "org.opensearch.cluster" in prefixes
        assert "org.opensearch.action" in prefixes
        assert "java.util.List" in prefixes

    def test_extract_imports_inner_class(self):
        """Inner class imports (e.g., Setting.Property) must be preserved."""
        source = textwrap.dedent("""\
            import org.opensearch.common.settings.Setting;
            import org.opensearch.common.settings.Setting.Property;
        """)
        full, prefixes = _extract_imports(source)
        assert "org.opensearch.common.settings.Setting" in full
        assert "org.opensearch.common.settings.Setting.Property" in full

    def test_extract_imports_jdk_types(self):
        """Standard library imports must not be silently dropped."""
        source = textwrap.dedent("""\
            import java.util.Collections;
            import java.util.Map;
            import java.util.concurrent.ConcurrentHashMap;
        """)
        full, _ = _extract_imports(source)
        assert "java.util.Collections" in full
        assert "java.util.Map" in full
        assert "java.util.concurrent.ConcurrentHashMap" in full

    def test_extract_imports_annotations(self):
        """Annotation imports must be preserved."""
        source = textwrap.dedent("""\
            import org.opensearch.common.annotation.PublicApi;
        """)
        full, _ = _extract_imports(source)
        assert "org.opensearch.common.annotation.PublicApi" in full

    def test_extract_imports_wildcard(self):
        """Wildcard imports should be included."""
        source = textwrap.dedent("""\
            import java.util.*;
            import static org.junit.Assert.*;
        """)
        full, _ = _extract_imports(source)
        assert "java.util.*" in full
        assert "org.junit.Assert.*" in full

    def test_extract_imports_deduplication(self):
        """Duplicate imports should be deduplicated."""
        source = textwrap.dedent("""\
            import java.util.List;
            import java.util.List;
        """)
        full, _ = _extract_imports(source)
        assert full.count("java.util.List") == 1

    def test_estimate_tokens(self):
        assert _estimate_tokens("abcd") == 1
        assert _estimate_tokens("a" * 400) == 100

    def test_extract_module(self, tmp_path):
        repo_root = str(tmp_path)
        file_path = os.path.join(repo_root, "server", "src", "main", "Foo.java")
        assert _extract_module(file_path, repo_root) == "server"

    def test_extract_module_nested(self, tmp_path):
        repo_root = str(tmp_path)
        file_path = os.path.join(repo_root, "plugins", "transport-nio", "src", "main", "Foo.java")
        assert _extract_module(file_path, repo_root) == "plugins/transport-nio"


# --- JavaChunker ---

class TestJavaChunker:
    @pytest.fixture
    def chunker(self):
        return JavaChunker()

    def test_simple_class(self, chunker, tmp_path):
        source = textwrap.dedent("""\
            package com.example;

            public class Hello {
                private String name;

                public Hello(String name) {
                    this.name = name;
                }

                public String greet() {
                    return "Hello, " + name;
                }
            }
        """)
        file_path = os.path.join(str(tmp_path), "Hello.java")
        symbols, chunks, file_doc = chunker.parse_file(file_path, source, str(tmp_path))

        # Should extract: class, field, constructor, method
        kinds = [s.kind for s in symbols]
        assert "class" in kinds
        assert "field" in kinds
        assert "constructor" in kinds
        assert "method" in kinds
        assert len(symbols) == 4

        # Should have method chunks + class summary
        chunk_kinds = [c.kind for c in chunks]
        assert "class_summary" in chunk_kinds
        assert "method" in chunk_kinds or "constructor" in chunk_kinds

        # File doc
        assert file_doc.filename == "Hello.java"
        assert "Hello" in file_doc.class_names

    def test_interface(self, chunker, tmp_path):
        source = textwrap.dedent("""\
            package com.example;

            public interface Greeter {
                String greet(String name);
            }
        """)
        file_path = os.path.join(str(tmp_path), "Greeter.java")
        symbols, chunks, file_doc = chunker.parse_file(file_path, source, str(tmp_path))

        assert any(s.kind == "interface" for s in symbols)

    def test_inheritance_metadata(self, chunker, tmp_path):
        source = textwrap.dedent("""\
            package com.example;

            public class Dog extends Animal implements Pet, Trainable {
                public void bark() {}
            }
        """)
        file_path = os.path.join(str(tmp_path), "Dog.java")
        symbols, chunks, file_doc = chunker.parse_file(file_path, source, str(tmp_path))

        cls = next(s for s in symbols if s.kind == "class")
        assert cls.extends_class == "Animal"
        assert "Pet" in cls.implements
        assert "Trainable" in cls.implements

    def test_large_method_splits(self, chunker, tmp_path):
        # Create a method with >500 tokens (~2000 chars)
        big_body = "\n".join([f"        int x{i} = {i};" for i in range(200)])
        source = f"""\
package com.example;

public class Big {{
    public void bigMethod() {{
{big_body}
    }}
}}
"""
        file_path = os.path.join(str(tmp_path), "Big.java")
        symbols, chunks, file_doc = chunker.parse_file(file_path, source, str(tmp_path))

        method_chunks = [c for c in chunks if c.kind == "method"]
        # Large method should be split into multiple parts
        assert len(method_chunks) > 1, "Large method should be split"

    def test_symbol_id_format(self, chunker, tmp_path):
        source = textwrap.dedent("""\
            package com.example;
            public class Foo {
                public void bar(String s, int n) {}
            }
        """)
        file_path = os.path.join(str(tmp_path), "Foo.java")
        symbols, _, _ = chunker.parse_file(file_path, source, str(tmp_path))

        method = next(s for s in symbols if s.kind == "method")
        assert "::" in method.id
        assert "#" in method.qualified_name
        assert "bar" in method.name

    def test_empty_file(self, chunker, tmp_path):
        source = ""
        file_path = os.path.join(str(tmp_path), "Empty.java")
        symbols, chunks, file_doc = chunker.parse_file(file_path, source, str(tmp_path))
        assert symbols == []
        assert file_doc.loc == 1  # empty string splits to ['']

    def test_imports_full_in_file_doc(self, chunker, tmp_path):
        """FileDoc.imports_full must contain untruncated import paths."""
        source = textwrap.dedent("""\
            package org.opensearch.cluster.service;

            import org.opensearch.cluster.ClusterManagerMetrics;
            import org.opensearch.common.settings.Setting;
            import org.opensearch.common.settings.Setting.Property;
            import org.opensearch.common.annotation.PublicApi;
            import java.util.Collections;
            import java.util.Map;

            @PublicApi(since = "1.0")
            public class ClusterService {
                public void doWork() {}
            }
        """)
        file_path = os.path.join(str(tmp_path), "ClusterService.java")
        _, _, file_doc = chunker.parse_file(file_path, source, str(tmp_path))

        full_imports = [i.strip() for i in file_doc.imports_full.split(",")]
        assert "org.opensearch.cluster.ClusterManagerMetrics" in full_imports
        assert "org.opensearch.common.settings.Setting" in full_imports
        assert "org.opensearch.common.settings.Setting.Property" in full_imports
        assert "org.opensearch.common.annotation.PublicApi" in full_imports
        assert "java.util.Collections" in full_imports
        assert "java.util.Map" in full_imports
        # Prefixes should still work
        prefix_imports = [i.strip() for i in file_doc.imports_summary.split(",")]
        assert "org.opensearch.cluster" in prefix_imports


# --- TextChunker ---

class TestTextChunker:
    @pytest.fixture
    def chunker(self):
        return TextChunker()

    def test_small_file(self, chunker, tmp_path):
        source = "line1\nline2\nline3\n"
        file_path = os.path.join(str(tmp_path), "small.txt")
        chunks, file_doc = chunker.chunk_file(file_path, source, str(tmp_path))
        # Too small for min chunk size — may be empty
        assert isinstance(chunks, list)
        assert file_doc.extension == "txt"

    def test_gradle_file(self, chunker, tmp_path):
        # Simulate a gradle file with enough content
        lines = [f"task task{i} {{ println 'hello {i}' }}" for i in range(100)]
        source = "\n".join(lines)
        file_path = os.path.join(str(tmp_path), "build.gradle")
        chunks, file_doc = chunker.chunk_file(file_path, source, str(tmp_path))

        assert len(chunks) > 0
        assert file_doc.extension == "gradle"
        assert file_doc.loc == 100


# --- Golden File Regression Tests ---

class TestGoldenFileJava:
    """Regression tests with a realistic Java file — exact known counts."""

    GOLDEN_SOURCE = textwrap.dedent("""\
        package org.opensearch.cluster.service;

        import org.opensearch.cluster.ClusterManagerMetrics;
        import org.opensearch.cluster.ClusterName;
        import org.opensearch.cluster.ClusterState;
        import org.opensearch.cluster.ClusterStateApplier;
        import org.opensearch.cluster.ClusterStateListener;
        import org.opensearch.cluster.ClusterStateTaskConfig;
        import org.opensearch.cluster.ClusterStateTaskExecutor;
        import org.opensearch.cluster.ClusterStateTaskListener;
        import org.opensearch.cluster.LocalNodeClusterManagerListener;
        import org.opensearch.cluster.NodeConnectionsService;
        import org.opensearch.cluster.StreamNodeConnectionsService;
        import org.opensearch.cluster.node.DiscoveryNode;
        import org.opensearch.cluster.routing.OperationRouting;
        import org.opensearch.cluster.routing.RerouteService;
        import org.opensearch.common.annotation.PublicApi;
        import org.opensearch.common.lifecycle.AbstractLifecycleComponent;
        import org.opensearch.common.settings.ClusterSettings;
        import org.opensearch.common.settings.Setting;
        import org.opensearch.common.settings.Setting.Property;
        import org.opensearch.common.settings.Settings;
        import org.opensearch.index.IndexingPressureService;
        import org.opensearch.node.Node;
        import org.opensearch.telemetry.metrics.noop.NoopMetricsRegistry;
        import org.opensearch.threadpool.ThreadPool;

        import java.util.Collections;
        import java.util.Map;

        @PublicApi(since = "1.0")
        public class ClusterService extends AbstractLifecycleComponent {
            private final ClusterSettings clusterSettings;
            private final OperationRouting operationRouting;

            public ClusterService(Settings settings, ClusterSettings clusterSettings,
                                  ThreadPool threadPool) {
                this.clusterSettings = clusterSettings;
                this.operationRouting = new OperationRouting(settings, clusterSettings);
            }

            public ClusterSettings getClusterSettings() {
                return clusterSettings;
            }

            public void submitStateUpdateTask(String source,
                                              ClusterStateTaskListener listener) {
                // submit task
            }
        }
    """)

    EXPECTED_IMPORT_COUNT = 26
    EXPECTED_FULL_IMPORTS = [
        "org.opensearch.cluster.ClusterManagerMetrics",
        "org.opensearch.cluster.ClusterName",
        "org.opensearch.cluster.ClusterState",
        "org.opensearch.cluster.ClusterStateApplier",
        "org.opensearch.cluster.ClusterStateListener",
        "org.opensearch.cluster.ClusterStateTaskConfig",
        "org.opensearch.cluster.ClusterStateTaskExecutor",
        "org.opensearch.cluster.ClusterStateTaskListener",
        "org.opensearch.cluster.LocalNodeClusterManagerListener",
        "org.opensearch.cluster.NodeConnectionsService",
        "org.opensearch.cluster.StreamNodeConnectionsService",
        "org.opensearch.cluster.node.DiscoveryNode",
        "org.opensearch.cluster.routing.OperationRouting",
        "org.opensearch.cluster.routing.RerouteService",
        "org.opensearch.common.annotation.PublicApi",
        "org.opensearch.common.lifecycle.AbstractLifecycleComponent",
        "org.opensearch.common.settings.ClusterSettings",
        "org.opensearch.common.settings.Setting",
        "org.opensearch.common.settings.Setting.Property",
        "org.opensearch.common.settings.Settings",
        "org.opensearch.index.IndexingPressureService",
        "org.opensearch.node.Node",
        "org.opensearch.telemetry.metrics.noop.NoopMetricsRegistry",
        "org.opensearch.threadpool.ThreadPool",
        "java.util.Collections",
        "java.util.Map",
    ]

    @pytest.fixture
    def chunker(self):
        return JavaChunker()

    def test_exact_import_count(self):
        """The golden file has exactly 26 imports — no more, no less."""
        full, _ = _extract_imports(self.GOLDEN_SOURCE)
        assert len(full) == self.EXPECTED_IMPORT_COUNT, (
            f"Expected {self.EXPECTED_IMPORT_COUNT} imports, got {len(full)}: {full}"
        )

    def test_every_import_preserved(self):
        """Every single import must appear in the full list — no silent drops."""
        full, _ = _extract_imports(self.GOLDEN_SOURCE)
        for expected in self.EXPECTED_FULL_IMPORTS:
            assert expected in full, f"Missing import: {expected}"

    def test_no_extra_imports(self):
        """No phantom imports should be added."""
        full, _ = _extract_imports(self.GOLDEN_SOURCE)
        full_set = set(full)
        expected_set = set(self.EXPECTED_FULL_IMPORTS)
        extras = full_set - expected_set
        assert not extras, f"Unexpected extra imports: {extras}"

    def test_file_doc_imports_full(self, chunker, tmp_path):
        """FileDoc.imports_full has all 26 imports after JavaChunker parsing."""
        file_path = os.path.join(str(tmp_path), "ClusterService.java")
        _, _, file_doc = chunker.parse_file(
            file_path, self.GOLDEN_SOURCE, str(tmp_path),
        )
        full = [i.strip() for i in file_doc.imports_full.split(",")]
        assert len(full) == self.EXPECTED_IMPORT_COUNT

    def test_symbol_extraction(self, chunker, tmp_path):
        """Extracts expected symbols: 1 class, 2 fields, 1 constructor, 2 methods."""
        file_path = os.path.join(str(tmp_path), "ClusterService.java")
        symbols, _, _ = chunker.parse_file(
            file_path, self.GOLDEN_SOURCE, str(tmp_path),
        )
        kinds = [s.kind for s in symbols]
        assert kinds.count("class") == 1
        assert kinds.count("field") == 2
        assert kinds.count("constructor") == 1
        assert kinds.count("method") == 2

    def test_class_metadata(self, chunker, tmp_path):
        """Class has correct extends and annotations."""
        file_path = os.path.join(str(tmp_path), "ClusterService.java")
        symbols, _, _ = chunker.parse_file(
            file_path, self.GOLDEN_SOURCE, str(tmp_path),
        )
        cls = next(s for s in symbols if s.kind == "class")
        assert cls.name == "ClusterService"
        assert cls.extends_class == "AbstractLifecycleComponent"
        assert "PublicApi" in cls.annotations


class TestTruncation:
    """Tests for truncation notice."""

    def test_truncation_includes_notice(self):
        from jigyasa_mcp.server.validation import truncate_response
        long_text = "x" * 20000
        result = truncate_response(long_text, max_chars=100)
        assert "[TRUNCATED" in result
        assert "chars omitted" in result

    def test_no_truncation_for_short_text(self):
        from jigyasa_mcp.server.validation import truncate_response
        short_text = "hello world"
        result = truncate_response(short_text)
        assert result == short_text
        assert "[TRUNCATED" not in result


class TestRerankerFilePath:
    """Tests for file path boost in reranker."""

    def test_file_path_match_boosts_score(self):
        from jigyasa_mcp.grpc_client import SearchHit, SearchResult
        from jigyasa_mcp.server.reranker import rerank

        hits = [
            SearchHit(score=1.0, doc_id="1", source={
                "name": "OtherClass",
                "file_path": "server/src/main/ClusterService.java",
            }),
            SearchHit(score=1.0, doc_id="2", source={
                "name": "SomeUtil",
                "file_path": "server/src/main/SomeUtil.java",
            }),
        ]
        result = SearchResult(total_hits=2, hits=hits)
        reranked = rerank(result, "ClusterService")
        # ClusterService should now rank first due to file path + name match
        assert reranked.hits[0].doc_id == "1"
        assert reranked.hits[0].score > reranked.hits[1].score
