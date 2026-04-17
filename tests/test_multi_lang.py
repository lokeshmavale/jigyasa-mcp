"""Tests for multi-language tree-sitter support."""

import textwrap

import pytest

from jigyasa_mcp.indexer.generic_ast_chunker import GenericASTChunker
from jigyasa_mcp.indexer.lang_registry import (
    JAVA_PROFILE,
    PYTHON_PROFILE,
    LanguageRegistry,
    get_registry,
)


class TestLanguageRegistry:
    """Tests for the language registry and auto-detection."""

    def test_detects_python(self):
        registry = LanguageRegistry()
        profile = registry.get_profile("src/main.py")
        assert profile is not None
        assert profile.name == "python"

    def test_detects_java(self):
        registry = LanguageRegistry()
        profile = registry.get_profile("Foo.java")
        assert profile is not None
        assert profile.name == "java"

    def test_detects_javascript(self):
        registry = LanguageRegistry()
        profile = registry.get_profile("app.js")
        assert profile is not None
        assert profile.name == "javascript"

    def test_detects_typescript(self):
        registry = LanguageRegistry()
        ts_profile = registry.get_profile("component.ts")
        assert ts_profile is not None
        assert ts_profile.name == "typescript"

        tsx_profile = registry.get_profile("component.tsx")
        assert tsx_profile is not None
        assert tsx_profile.name == "tsx"

    def test_detects_go(self):
        registry = LanguageRegistry()
        profile = registry.get_profile("main.go")
        assert profile is not None
        assert profile.name == "go"

    def test_detects_rust(self):
        registry = LanguageRegistry()
        profile = registry.get_profile("lib.rs")
        assert profile is not None
        assert profile.name == "rust"

    def test_detects_csharp(self):
        registry = LanguageRegistry()
        profile = registry.get_profile("Program.cs")
        assert profile is not None
        assert profile.name == "c_sharp"

    def test_returns_none_for_unknown(self):
        registry = LanguageRegistry()
        assert registry.get_profile("data.csv") is None
        assert registry.get_profile("image.png") is None

    def test_case_insensitive_extension(self):
        registry = LanguageRegistry()
        profile = registry.get_profile("Main.PY")
        assert profile is not None
        assert profile.name == "python"

    def test_supported_extensions(self):
        registry = LanguageRegistry()
        exts = registry.supported_extensions()
        assert ".py" in exts
        assert ".java" in exts
        assert ".ts" in exts
        assert ".go" in exts

    def test_available_languages_includes_java(self):
        registry = LanguageRegistry()
        available = registry.available_languages()
        assert "java" in available

    def test_status_shows_installed(self):
        registry = LanguageRegistry()
        status = registry.status()
        assert status["java"] == "installed"

    def test_get_parser_returns_parser_for_java(self):
        registry = LanguageRegistry()
        parser, profile = registry.get_parser("Foo.java")
        assert parser is not None
        assert profile is not None
        assert profile.name == "java"

    def test_get_parser_returns_none_for_unknown(self):
        registry = LanguageRegistry()
        parser, profile = registry.get_parser("data.csv")
        assert parser is None
        assert profile is None

    def test_singleton_registry(self):
        r1 = get_registry()
        r2 = get_registry()
        assert r1 is r2


class TestLanguageProfile:
    """Tests for LanguageProfile node type mapping."""

    def test_python_node_to_kind(self):
        assert PYTHON_PROFILE.node_to_kind("class_definition") == "class"
        assert PYTHON_PROFILE.node_to_kind("function_definition") == "function"
        assert PYTHON_PROFILE.node_to_kind("if_statement") is None

    def test_java_node_to_kind(self):
        assert JAVA_PROFILE.node_to_kind("class_declaration") == "class"
        assert JAVA_PROFILE.node_to_kind("method_declaration") == "method"
        assert JAVA_PROFILE.node_to_kind("interface_declaration") == "interface"
        assert JAVA_PROFILE.node_to_kind("enum_declaration") == "enum"

    def test_all_declaration_nodes(self):
        nodes = PYTHON_PROFILE.all_declaration_nodes
        assert "class_definition" in nodes
        assert "function_definition" in nodes


class TestGenericASTChunkerPython:
    """Tests for GenericASTChunker with Python source code."""

    @pytest.fixture()
    def python_chunker(self):
        registry = LanguageRegistry()
        parser, profile = registry.get_parser("test.py")
        if parser is None:
            pytest.skip("tree-sitter-python not installed")
        return GenericASTChunker(parser, profile)

    def test_extracts_class(self, python_chunker, tmp_path):
        source = textwrap.dedent("""\
            class MyService:
                def __init__(self, name):
                    self.name = name

                def process(self, data):
                    return data.upper()
        """)
        repo = str(tmp_path)
        fpath = str(tmp_path / "service.py")
        with open(fpath, "w") as f:
            f.write(source)

        symbols, chunks, file_doc = python_chunker.parse_file(
            fpath, source, repo,
        )

        # Should find class + methods
        class_syms = [s for s in symbols if s.kind == "class"]
        assert len(class_syms) == 1
        assert class_syms[0].name == "MyService"

        method_syms = [s for s in symbols if s.kind == "method"]
        assert len(method_syms) >= 2
        method_names = {s.name for s in method_syms}
        assert "__init__" in method_names
        assert "process" in method_names

    def test_extracts_function(self, python_chunker, tmp_path):
        source = textwrap.dedent("""\
            def calculate_sum(a, b):
                return a + b

            def calculate_product(a, b):
                return a * b
        """)
        repo = str(tmp_path)
        fpath = str(tmp_path / "math_utils.py")
        with open(fpath, "w") as f:
            f.write(source)

        symbols, chunks, file_doc = python_chunker.parse_file(
            fpath, source, repo,
        )

        func_syms = [s for s in symbols if s.kind == "function"]
        assert len(func_syms) == 2
        names = {s.name for s in func_syms}
        assert "calculate_sum" in names
        assert "calculate_product" in names

    def test_file_doc_metadata(self, python_chunker, tmp_path):
        source = textwrap.dedent("""\
            from os.path import join
            import sys

            class App:
                pass
        """)
        repo = str(tmp_path)
        fpath = str(tmp_path / "app.py")
        with open(fpath, "w") as f:
            f.write(source)

        symbols, chunks, file_doc = python_chunker.parse_file(
            fpath, source, repo,
        )

        assert file_doc.filename == "app.py"
        assert file_doc.extension == "py"
        assert "App" in file_doc.class_names

    def test_visibility_conventions(self, python_chunker, tmp_path):
        source = textwrap.dedent("""\
            class Example:
                def public_method(self):
                    pass

                def _protected_method(self):
                    pass

                def __private_method(self):
                    pass

                def __dunder__(self):
                    pass
        """)
        repo = str(tmp_path)
        fpath = str(tmp_path / "example.py")
        with open(fpath, "w") as f:
            f.write(source)

        symbols, _, _ = python_chunker.parse_file(fpath, source, repo)
        by_name = {s.name: s for s in symbols if s.kind == "method"}

        assert by_name["public_method"].visibility == "public"
        assert by_name["_protected_method"].visibility == "protected"
        assert by_name["__private_method"].visibility == "private"
        assert by_name["__dunder__"].visibility == "public"


class TestGenericASTChunkerJavaScript:
    """Tests for GenericASTChunker with JavaScript source code."""

    @pytest.fixture()
    def js_chunker(self):
        registry = LanguageRegistry()
        parser, profile = registry.get_parser("test.js")
        if parser is None:
            pytest.skip("tree-sitter-javascript not installed")
        return GenericASTChunker(parser, profile)

    def test_extracts_function(self, js_chunker, tmp_path):
        source = textwrap.dedent("""\
            function greet(name) {
                return `Hello, ${name}!`;
            }

            function add(a, b) {
                return a + b;
            }
        """)
        repo = str(tmp_path)
        fpath = str(tmp_path / "utils.js")
        with open(fpath, "w") as f:
            f.write(source)

        symbols, chunks, file_doc = js_chunker.parse_file(
            fpath, source, repo,
        )

        func_syms = [s for s in symbols if s.kind == "function"]
        assert len(func_syms) == 2
        names = {s.name for s in func_syms}
        assert "greet" in names
        assert "add" in names

    def test_extracts_class(self, js_chunker, tmp_path):
        source = textwrap.dedent("""\
            class Animal {
                constructor(name) {
                    this.name = name;
                }

                speak() {
                    return this.name + ' makes a noise.';
                }
            }
        """)
        repo = str(tmp_path)
        fpath = str(tmp_path / "animal.js")
        with open(fpath, "w") as f:
            f.write(source)

        symbols, chunks, file_doc = js_chunker.parse_file(
            fpath, source, repo,
        )

        class_syms = [s for s in symbols if s.kind == "class"]
        assert len(class_syms) == 1
        assert class_syms[0].name == "Animal"

        method_syms = [s for s in symbols if s.kind == "method"]
        method_names = {s.name for s in method_syms}
        assert "speak" in method_names
