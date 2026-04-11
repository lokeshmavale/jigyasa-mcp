"""Hybrid code chunker: AST-aware boundaries with sliding window fallback.

Uses tree-sitter for Java AST parsing. Falls back to line-based chunking
for non-Java files or when AST parsing fails.
"""

import logging
import os
import re
from dataclasses import dataclass
from pathlib import Path

import tree_sitter_java as tsjava
from tree_sitter import Language, Parser

logger = logging.getLogger(__name__)

JAVA_LANGUAGE = Language(tsjava.language())

# Approximate token count: ~4 chars per token for code
CHARS_PER_TOKEN = 4
MAX_CHUNK_TOKENS = 500
MIN_CHUNK_TOKENS = 50
OVERLAP_TOKENS = 50

# Files/dirs to skip
SKIP_PATTERNS = {
    "build", "generated", "generated-src", ".gradle", ".idea",
    "node_modules", "__pycache__", ".git",
}
SKIP_EXTENSIONS = {".class", ".jar", ".war", ".zip", ".tar", ".gz", ".png", ".jpg"}


@dataclass
class Symbol:
    """A code symbol extracted from AST."""
    id: str
    name: str
    qualified_name: str
    kind: str  # class, interface, enum, method, constructor, field
    signature: str
    visibility: str
    file_path: str
    package: str
    module: str
    parent_class: str
    implements: str  # comma-separated
    extends_class: str
    annotations: str  # comma-separated
    line_start: int
    line_end: int
    body_preview: str  # first ~200 chars of body
    imports: str = ""  # comma-separated import packages (class-level only)
    type_references: str = ""  # comma-separated types referenced in this symbol


@dataclass
class Chunk:
    """A code chunk for indexing."""
    id: str
    content: str
    file_path: str
    symbol_name: str
    kind: str  # method, class, class_summary, block
    module: str
    language: str
    enclosing_class: str
    enclosing_method: str
    line_start: int
    line_end: int
    token_count: int
    embedding: list[float] | None = None


@dataclass
class FileDoc:
    """File-level metadata document."""
    id: str
    path: str
    filename: str
    extension: str
    module: str
    package: str
    class_names: str  # comma-separated
    imports_summary: str  # comma-separated unique import packages
    loc: int
    last_commit_sha: str


def _estimate_tokens(text: str) -> int:
    return len(text) // CHARS_PER_TOKEN


def _extract_module(file_path: str, repo_root: str) -> str:
    """Extract the Gradle module name from the file path."""
    rel = os.path.relpath(file_path, repo_root).replace("\\", "/")
    parts = rel.split("/")
    # Common patterns: server/src/..., plugins/xyz/src/..., libs/xyz/src/...
    if "src" in parts:
        src_idx = parts.index("src")
        return "/".join(parts[:src_idx])
    return parts[0] if parts else ""


def _extract_package(source: str) -> str:
    """Extract Java package declaration."""
    match = re.search(r"^\s*package\s+([\w.]+)\s*;", source, re.MULTILINE)
    return match.group(1) if match else ""


def _extract_imports(source: str) -> list[str]:
    """Extract unique import package prefixes (first 3 segments)."""
    imports = re.findall(r"^\s*import\s+(?:static\s+)?([\w.]+);", source, re.MULTILINE)
    prefixes = set()
    for imp in imports:
        parts = imp.split(".")
        prefix = ".".join(parts[:min(3, len(parts))])
        prefixes.add(prefix)
    return sorted(prefixes)


def should_skip_file(file_path: str) -> bool:
    """Check if file should be excluded from indexing."""
    path = Path(file_path)
    if path.suffix.lower() in SKIP_EXTENSIONS:
        return True
    parts = set(path.parts)
    return bool(parts & SKIP_PATTERNS)


def _get_visibility(node) -> str:
    """Extract visibility modifier from a Java declaration node."""
    for child in node.children:
        if child.type == "modifiers":
            text = child.text.decode("utf-8") if isinstance(child.text, bytes) else child.text
            for vis in ("public", "protected", "private"):
                if vis in text:
                    return vis
    return "package-private"


def _get_annotations(node) -> list[str]:
    """Extract annotation names from a declaration."""
    annotations = []
    for child in node.children:
        if child.type == "modifiers":
            for mod_child in child.children:
                if mod_child.type in ("marker_annotation", "annotation"):
                    text = mod_child.text
                    name = text.decode("utf-8") if isinstance(text, bytes) else text
                    annotations.append(name.lstrip("@").split("(")[0])
    return annotations


def _get_text(node) -> str:
    return node.text.decode("utf-8") if isinstance(node.text, bytes) else node.text


def _extract_type_references(node) -> list[str]:
    """Extract all type_identifier references from a subtree (for dependency graph)."""
    types = set()
    _walk_types(node, types)
    return sorted(types)


def _walk_types(node, types: set):
    if node.type == "type_identifier":
        name = _get_text(node)
        # Skip common Java types
        if name not in ("String", "Object", "Class", "Integer", "Long", "Boolean",
                        "Double", "Float", "Byte", "Short", "Character", "Void",
                        "List", "Map", "Set", "Collection", "Optional", "Stream",
                        "Override", "Deprecated", "SuppressWarnings"):
            types.add(name)
    for child in node.children:
        _walk_types(child, types)


def _get_superclass(node) -> str:
    """Extract extends clause."""
    for child in node.children:
        if child.type == "superclass":
            return _get_text(child).replace("extends ", "").strip()
    return ""


def _get_interfaces(node) -> list[str]:
    """Extract implements clause."""
    for child in node.children:
        if child.type == "super_interfaces":
            text = _get_text(child).replace("implements ", "").strip()
            return [i.strip() for i in text.split(",")]
    return []


def _get_name(node) -> str:
    """Get the name identifier from a declaration."""
    for child in node.children:
        if child.type == "identifier":
            return _get_text(child)
    return ""


def _get_method_signature(node) -> str:
    """Build method signature: name(ParamType1, ParamType2)"""
    name = _get_name(node)
    params = []
    for child in node.children:
        if child.type == "formal_parameters":
            for param in child.children:
                if param.type == "formal_parameter":
                    # Get type (first type child)
                    for pc in param.children:
                        if pc.type in (
                            "type_identifier", "generic_type", "array_type",
                            "integral_type", "floating_point_type", "boolean_type",
                            "void_type",
                        ):
                            params.append(_get_text(pc))
                            break
    return f"{name}({', '.join(params)})"


class JavaChunker:
    """Extracts symbols and chunks from Java source files using tree-sitter."""

    def __init__(self):
        self.parser = Parser(JAVA_LANGUAGE)

    def parse_file(
        self,
        file_path: str,
        source: str,
        repo_root: str,
        commit_sha: str = "",
    ) -> tuple[list[Symbol], list[Chunk], FileDoc]:
        """Parse a Java file into symbols, chunks, and file metadata."""
        module = _extract_module(file_path, repo_root)
        package = _extract_package(source)
        rel_path = os.path.relpath(file_path, repo_root).replace("\\", "/")

        tree = self.parser.parse(source.encode("utf-8"))
        root = tree.root_node

        symbols: list[Symbol] = []
        chunks: list[Chunk] = []
        class_names: list[str] = []

        # Walk top-level declarations
        for node in root.children:
            if node.type in ("class_declaration", "interface_declaration", "enum_declaration"):
                cls_symbols, cls_chunks = self._process_class(
                    node, rel_path, package, module, source, parent_class=""
                )
                symbols.extend(cls_symbols)
                chunks.extend(cls_chunks)
                class_names.append(_get_name(node))

        # File-level document
        lines = source.split("\n")
        file_doc = FileDoc(
            id=rel_path,
            path=rel_path,
            filename=os.path.basename(file_path),
            extension=Path(file_path).suffix.lstrip("."),
            module=module,
            package=package,
            class_names=", ".join(class_names),
            imports_summary=", ".join(_extract_imports(source)),
            loc=len(lines),
            last_commit_sha=commit_sha,
        )

        return symbols, chunks, file_doc

    def _process_class(
        self,
        node,
        file_path: str,
        package: str,
        module: str,
        source: str,
        parent_class: str,
    ) -> tuple[list[Symbol], list[Chunk]]:
        """Process a class/interface/enum declaration."""
        symbols = []
        chunks = []

        cls_name = _get_name(node)
        qualified = f"{package}.{cls_name}" if package else cls_name
        kind = node.type.replace("_declaration", "")
        body_text = _get_text(node)

        # Class symbol — with type references for dependency graph
        type_refs = _extract_type_references(node)
        # imports are file-level, captured from source
        file_imports = ", ".join(_extract_imports(source)) if not parent_class else ""

        symbols.append(Symbol(
            id=f"{file_path}::{kind}::{qualified}",
            name=cls_name,
            qualified_name=qualified,
            kind=kind,
            signature=f"{kind} {cls_name}",
            visibility=_get_visibility(node),
            file_path=file_path,
            package=package,
            module=module,
            parent_class=parent_class,
            implements=", ".join(_get_interfaces(node)),
            extends_class=_get_superclass(node),
            annotations=", ".join(_get_annotations(node)),
            line_start=node.start_point[0] + 1,
            line_end=node.end_point[0] + 1,
            body_preview=body_text[:200],
            imports=file_imports,
            type_references=", ".join(type_refs[:50]),  # cap at 50 to avoid huge fields
        ))

        # Class summary chunk (Javadoc + method signatures, no bodies)
        summary_lines = [f"// {file_path} :: {qualified}"]
        summary_lines.append(f"{kind} {cls_name}")
        ext = _get_superclass(node)
        if ext:
            summary_lines[-1] += f" extends {ext}"
        impls = _get_interfaces(node)
        if impls:
            summary_lines[-1] += f" implements {', '.join(impls)}"

        # Find class body and extract members
        class_body = None
        for child in node.children:
            if child.type in ("class_body", "interface_body", "enum_body"):
                class_body = child
                break

        if class_body:
            for member in class_body.children:
                if member.type in ("method_declaration", "constructor_declaration"):
                    sig = _get_method_signature(member)
                    vis = _get_visibility(member)
                    summary_lines.append(f"  {vis} {sig}")

                    # Method symbol
                    method_qualified = f"{qualified}#{sig}"
                    method_text = _get_text(member)
                    symbols.append(Symbol(
                        id=f"{file_path}::method::{method_qualified}",
                        name=sig.split("(")[0],
                        qualified_name=method_qualified,
                        kind="method" if member.type == "method_declaration" else "constructor",
                        signature=sig,
                        visibility=vis,
                        file_path=file_path,
                        package=package,
                        module=module,
                        parent_class=cls_name,
                        implements="",
                        extends_class="",
                        annotations=", ".join(_get_annotations(member)),
                        line_start=member.start_point[0] + 1,
                        line_end=member.end_point[0] + 1,
                        body_preview=method_text[:200],
                    ))

                    # Method chunk (with context prefix)
                    method_chunks = self._chunk_method(
                        member, method_qualified, file_path, module, cls_name
                    )
                    chunks.extend(method_chunks)

                elif member.type == "field_declaration":
                    field_text = _get_text(member).strip().rstrip(";")
                    field_name = ""
                    for fc in member.children:
                        if fc.type == "variable_declarator":
                            field_name = _get_name(fc)
                            break
                    if field_name:
                        symbols.append(Symbol(
                            id=f"{file_path}::field::{qualified}.{field_name}",
                            name=field_name,
                            qualified_name=f"{qualified}.{field_name}",
                            kind="field",
                            signature=field_text[:100],
                            visibility=_get_visibility(member),
                            file_path=file_path,
                            package=package,
                            module=module,
                            parent_class=cls_name,
                            implements="",
                            extends_class="",
                            annotations=", ".join(_get_annotations(member)),
                            line_start=member.start_point[0] + 1,
                            line_end=member.end_point[0] + 1,
                            body_preview=field_text[:200],
                        ))

                # Inner classes — recurse
                elif member.type in (
                    "class_declaration", "interface_declaration", "enum_declaration",
                ):
                    inner_syms, inner_chunks = self._process_class(
                        member, file_path, package, module, source,
                        parent_class=cls_name,
                    )
                    symbols.extend(inner_syms)
                    chunks.extend(inner_chunks)

        # Class summary chunk
        summary_text = "\n".join(summary_lines)
        chunks.append(Chunk(
            id=f"{file_path}::class_summary::{qualified}",
            content=summary_text,
            file_path=file_path,
            symbol_name=cls_name,
            kind="class_summary",
            module=module,
            language="java",
            enclosing_class=parent_class,
            enclosing_method="",
            line_start=node.start_point[0] + 1,
            line_end=node.end_point[0] + 1,
            token_count=_estimate_tokens(summary_text),
        ))

        return symbols, chunks

    def _chunk_method(
        self,
        node,
        qualified_name: str,
        file_path: str,
        module: str,
        enclosing_class: str,
    ) -> list[Chunk]:
        """Chunk a method, using sliding window if too large."""
        text = _get_text(node)
        context_prefix = f"// {file_path} :: {enclosing_class} :: {qualified_name}\n"
        full_content = context_prefix + text
        tokens = _estimate_tokens(full_content)

        if tokens <= MAX_CHUNK_TOKENS:
            return [Chunk(
                id=f"{file_path}::method_chunk::{qualified_name}",
                content=full_content,
                file_path=file_path,
                symbol_name=(
                    qualified_name.split("#")[0] if "#" in qualified_name
                    else qualified_name
                ),
                kind="method",
                module=module,
                language="java",
                enclosing_class=enclosing_class,
                enclosing_method=qualified_name,
                line_start=node.start_point[0] + 1,
                line_end=node.end_point[0] + 1,
                token_count=tokens,
            )]

        # Sliding window for large methods
        chunks = []
        lines = text.split("\n")
        max_chars = MAX_CHUNK_TOKENS * CHARS_PER_TOKEN
        overlap_chars = OVERLAP_TOKENS * CHARS_PER_TOKEN
        start_line = 0
        part = 0

        while start_line < len(lines):
            chunk_lines = []
            char_count = 0
            end_line = start_line

            while end_line < len(lines) and char_count < max_chars:
                chunk_lines.append(lines[end_line])
                char_count += len(lines[end_line]) + 1
                end_line += 1

            chunk_text = context_prefix + "\n".join(chunk_lines)
            abs_start = node.start_point[0] + 1 + start_line
            abs_end = node.start_point[0] + 1 + end_line - 1

            chunks.append(Chunk(
                id=f"{file_path}::method_chunk::{qualified_name}::part{part}",
                content=chunk_text,
                file_path=file_path,
                symbol_name=(
                    qualified_name.split("#")[0] if "#" in qualified_name
                    else qualified_name
                ),
                kind="method",
                module=module,
                language="java",
                enclosing_class=enclosing_class,
                enclosing_method=qualified_name,
                line_start=abs_start,
                line_end=abs_end,
                token_count=_estimate_tokens(chunk_text),
            ))

            # Advance with overlap
            overlap_lines = 0
            overlap_count = 0
            for i in range(end_line - 1, start_line, -1):
                overlap_count += len(lines[i]) + 1
                overlap_lines += 1
                if overlap_count >= overlap_chars:
                    break
            start_line = end_line - overlap_lines
            part += 1

        return chunks


class TextChunker:
    """Line-based chunker for non-Java files (gradle, xml, yml, md, etc.)."""

    def chunk_file(
        self,
        file_path: str,
        source: str,
        repo_root: str,
        commit_sha: str = "",
    ) -> tuple[list[Chunk], FileDoc]:
        """Chunk a non-Java file using line-based sliding window."""
        rel_path = os.path.relpath(file_path, repo_root).replace("\\", "/")
        module = _extract_module(file_path, repo_root)
        ext = Path(file_path).suffix.lstrip(".")
        lines = source.split("\n")

        chunks = []
        max_chars = MAX_CHUNK_TOKENS * CHARS_PER_TOKEN
        overlap_chars = OVERLAP_TOKENS * CHARS_PER_TOKEN
        start_line = 0
        part = 0

        while start_line < len(lines):
            chunk_lines = []
            char_count = 0
            end_line = start_line

            while end_line < len(lines) and char_count < max_chars:
                chunk_lines.append(lines[end_line])
                char_count += len(lines[end_line]) + 1
                end_line += 1

            context_prefix = f"// {rel_path}\n"
            chunk_text = context_prefix + "\n".join(chunk_lines)

            if _estimate_tokens(chunk_text) >= MIN_CHUNK_TOKENS:
                chunks.append(Chunk(
                    id=f"{rel_path}::block::part{part}",
                    content=chunk_text,
                    file_path=rel_path,
                    symbol_name="",
                    kind="block",
                    module=module,
                    language=ext,
                    enclosing_class="",
                    enclosing_method="",
                    line_start=start_line + 1,
                    line_end=end_line,
                    token_count=_estimate_tokens(chunk_text),
                ))
                part += 1

            overlap_lines = 0
            overlap_count = 0
            for i in range(end_line - 1, start_line, -1):
                overlap_count += len(lines[i]) + 1
                overlap_lines += 1
                if overlap_count >= overlap_chars:
                    break
            start_line = end_line - overlap_lines
            if start_line == end_line - overlap_lines and end_line >= len(lines):
                break

        file_doc = FileDoc(
            id=rel_path,
            path=rel_path,
            filename=os.path.basename(file_path),
            extension=ext,
            module=module,
            package="",
            class_names="",
            imports_summary="",
            loc=len(lines),
            last_commit_sha=commit_sha,
        )

        return chunks, file_doc
