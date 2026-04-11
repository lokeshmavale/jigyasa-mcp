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
        imports = _extract_imports(source)
        assert "org.opensearch.cluster" in imports
        assert "org.opensearch.action" in imports
        assert "java.util.List" in imports

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
