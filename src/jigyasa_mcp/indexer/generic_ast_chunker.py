"""Generic AST-aware chunker driven by language profiles.

Uses tree-sitter + LanguageProfile to extract symbols and code chunks from
any supported language. Replaces the need for language-specific chunker classes.
"""

import logging
import os
import re
from pathlib import Path

from tree_sitter import Parser

from jigyasa_mcp.indexer.chunker import (
    CHARS_PER_TOKEN,
    MAX_CHUNK_TOKENS,
    OVERLAP_TOKENS,
    Chunk,
    FileDoc,
    Symbol,
    _estimate_tokens,
    _extract_module,
    _get_text,
)
from jigyasa_mcp.indexer.lang_registry import LanguageProfile

logger = logging.getLogger(__name__)


def _find_name(node, profile: LanguageProfile) -> str:
    """Find the name identifier in a declaration node."""
    for child in node.children:
        if child.type in (profile.name_node_type, "property_identifier"):
            return _get_text(child)
        # Some languages nest the name (e.g., Go type_spec inside type_declaration)
        if child.type in ("type_spec", "function_declarator"):
            return _find_name(child, profile)
    return ""


def _find_decorators(node, profile: LanguageProfile) -> list[str]:
    """Extract decorator/annotation names from a declaration."""
    decorators = []
    for child in node.children:
        if child.type in profile.decorator_node_types:
            text = _get_text(child).lstrip("@").split("(")[0]
            decorators.append(text.strip())
    # Also check previous siblings (Python decorators are siblings)
    prev = node.prev_named_sibling
    while prev and prev.type in profile.decorator_node_types:
        text = _get_text(prev).lstrip("@").split("(")[0]
        decorators.insert(0, text.strip())
        prev = prev.prev_named_sibling
    return decorators


def _find_visibility(node, profile: LanguageProfile) -> str:
    """Infer visibility from node modifiers or decorators."""
    text = _get_text(node)
    # Check common visibility keywords in the node text
    first_line = text.split("\n")[0].lower()
    for vis in ("public", "protected", "private", "internal"):
        if vis in first_line:
            return vis
    # Python convention: leading underscore
    if profile.name == "python":
        name = _find_name(node, profile)
        if name.startswith("__") and name.endswith("__"):
            return "public"  # dunder methods
        if name.startswith("__"):
            return "private"
        if name.startswith("_"):
            return "protected"
    # Go convention: uppercase = exported
    if profile.name == "go":
        name = _find_name(node, profile)
        if name and name[0].isupper():
            return "public"
        return "package-private"
    return "public"


def _extract_imports_generic(source: str, profile: LanguageProfile) -> list[str]:
    """Extract import paths using the language profile's regex."""
    if not profile.import_pattern:
        return []
    imports = re.findall(profile.import_pattern, source, re.MULTILINE)
    # Flatten tuples from multi-group patterns
    flat = []
    for imp in imports:
        if isinstance(imp, tuple):
            flat.extend(i for i in imp if i)
        else:
            flat.append(imp)
    # Deduplicate and take first 3 segments
    prefixes = set()
    for imp in flat:
        parts = re.split(r"[./:]", imp)
        prefix = ".".join(parts[:min(3, len(parts))])
        prefixes.add(prefix)
    return sorted(prefixes)


def _extract_package_generic(source: str, profile: LanguageProfile) -> str:
    """Extract package/namespace using the language profile's regex."""
    if not profile.package_pattern:
        return ""
    match = re.search(profile.package_pattern, source, re.MULTILINE)
    return match.group(1) if match else ""


def _find_parent_class_name(node, profile: LanguageProfile) -> str:
    """Walk up the tree to find the enclosing class name."""
    parent = node.parent
    while parent:
        if parent.type in profile.class_nodes:
            return _find_name(parent, profile)
        parent = parent.parent
    return ""


class GenericASTChunker:
    """Language-agnostic AST chunker driven by a LanguageProfile.

    Walks the tree-sitter AST, extracts symbols (classes, functions, methods,
    fields), and creates searchable code chunks with context.
    """

    def __init__(self, parser: Parser, profile: LanguageProfile):
        self.parser = parser
        self.profile = profile

    def parse_file(
        self,
        file_path: str,
        source: str,
        repo_root: str,
        commit_sha: str = "",
    ) -> tuple[list[Symbol], list[Chunk], FileDoc]:
        """Parse a source file into symbols, chunks, and file metadata."""
        rel_path = os.path.relpath(file_path, repo_root).replace("\\", "/")
        module = _extract_module(file_path, repo_root)
        package = _extract_package_generic(source, self.profile)
        imports = _extract_imports_generic(source, self.profile)

        try:
            tree = self.parser.parse(source.encode("utf-8"))
        except Exception as e:
            logger.warning(
                f"tree-sitter parse failed for {rel_path}: {e}"
            )
            return [], [], self._make_file_doc(
                rel_path, file_path, module, package, imports,
                source, commit_sha, [],
            )

        root = tree.root_node
        symbols: list[Symbol] = []
        chunks: list[Chunk] = []
        class_names: list[str] = []

        self._walk_declarations(
            root, rel_path, package, module, source,
            symbols, chunks, class_names,
        )

        file_doc = self._make_file_doc(
            rel_path, file_path, module, package, imports,
            source, commit_sha, class_names,
        )
        return symbols, chunks, file_doc

    def _walk_declarations(
        self,
        node,
        file_path: str,
        package: str,
        module: str,
        source: str,
        symbols: list[Symbol],
        chunks: list[Chunk],
        class_names: list[str],
        parent_class: str = "",
    ):
        """Recursively walk AST and extract symbols + chunks."""
        for child in node.children:
            kind = self.profile.node_to_kind(child.type)
            if kind is None:
                # Recurse into non-declaration nodes
                self._walk_declarations(
                    child, file_path, package, module, source,
                    symbols, chunks, class_names, parent_class,
                )
                continue

            name = _find_name(child, self.profile)
            if not name:
                continue

            # Build qualified name
            if parent_class:
                qualified = (
                    f"{package}.{parent_class}.{name}" if package
                    else f"{parent_class}.{name}"
                )
            elif package:
                qualified = f"{package}.{name}"
            else:
                qualified = name

            # Track class names at top level
            if kind in ("class", "interface", "enum") and not parent_class:
                class_names.append(name)

            visibility = _find_visibility(child, self.profile)
            decorators = _find_decorators(child, self.profile)
            body_text = _get_text(child)

            # Detect if function inside class → method
            actual_kind = kind
            if kind == "function" and parent_class:
                actual_kind = "method"

            symbols.append(Symbol(
                id=f"{file_path}::{actual_kind}::{qualified}",
                name=name,
                qualified_name=qualified,
                kind=actual_kind,
                signature=self._build_signature(child, name),
                visibility=visibility,
                file_path=file_path,
                package=package,
                module=module,
                parent_class=parent_class,
                implements="",
                extends_class="",
                annotations=", ".join(decorators),
                line_start=child.start_point[0] + 1,
                line_end=child.end_point[0] + 1,
                body_preview=body_text[:200],
                imports="",
                type_references="",
            ))

            # Create chunk for the declaration body
            self._create_chunks(
                child, file_path, qualified, actual_kind, module,
                parent_class, source, chunks,
            )

            # Recurse into classes to find methods/fields
            if kind in ("class", "interface", "enum"):
                self._walk_declarations(
                    child, file_path, package, module, source,
                    symbols, chunks, class_names,
                    parent_class=name,
                )

    def _build_signature(self, node, name: str) -> str:
        """Build a simple signature string for a declaration."""
        # Try to extract just the first line (declaration line)
        text = _get_text(node)
        first_line = text.split("\n")[0].strip()
        # Truncate if too long
        if len(first_line) > 120:
            first_line = first_line[:120] + "..."
        return first_line

    def _create_chunks(
        self,
        node,
        file_path: str,
        qualified_name: str,
        kind: str,
        module: str,
        enclosing_class: str,
        source: str,
        chunks: list[Chunk],
    ):
        """Create code chunks for a declaration, splitting large ones."""
        body_text = _get_text(node)
        tokens = _estimate_tokens(body_text)

        if tokens <= MAX_CHUNK_TOKENS:
            chunks.append(Chunk(
                id=f"{file_path}::{kind}_chunk::{qualified_name}",
                content=f"// {file_path}\n{body_text}",
                file_path=file_path,
                symbol_name=(
                    qualified_name.split("#")[0] if "#" in qualified_name
                    else qualified_name
                ),
                kind=kind,
                module=module,
                language=self.profile.name,
                enclosing_class=enclosing_class,
                enclosing_method=(
                    qualified_name if kind in ("method", "function")
                    else ""
                ),
                line_start=node.start_point[0] + 1,
                line_end=node.end_point[0] + 1,
                token_count=tokens,
            ))
        else:
            # Split large declarations with overlap
            self._split_large_node(
                node, file_path, qualified_name, kind, module,
                enclosing_class, body_text, chunks,
            )

    def _split_large_node(
        self,
        node,
        file_path: str,
        qualified_name: str,
        kind: str,
        module: str,
        enclosing_class: str,
        body_text: str,
        chunks: list[Chunk],
    ):
        """Split a large declaration into overlapping chunks."""
        lines = body_text.split("\n")
        max_chars = MAX_CHUNK_TOKENS * CHARS_PER_TOKEN
        overlap_chars = OVERLAP_TOKENS * CHARS_PER_TOKEN
        context = f"// {file_path} — {qualified_name}\n"

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

            chunk_text = context + "\n".join(chunk_lines)
            abs_start = node.start_point[0] + 1 + start_line
            abs_end = node.start_point[0] + 1 + end_line - 1

            chunks.append(Chunk(
                id=(
                    f"{file_path}::{kind}_chunk"
                    f"::{qualified_name}::part{part}"
                ),
                content=chunk_text,
                file_path=file_path,
                symbol_name=(
                    qualified_name.split("#")[0] if "#" in qualified_name
                    else qualified_name
                ),
                kind=kind,
                module=module,
                language=self.profile.name,
                enclosing_class=enclosing_class,
                enclosing_method=(
                    qualified_name if kind in ("method", "function")
                    else ""
                ),
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

    def _make_file_doc(
        self,
        rel_path: str,
        file_path: str,
        module: str,
        package: str,
        imports: list[str],
        source: str,
        commit_sha: str,
        class_names: list[str],
    ) -> FileDoc:
        return FileDoc(
            id=rel_path,
            path=rel_path,
            filename=os.path.basename(file_path),
            extension=Path(file_path).suffix.lstrip("."),
            module=module,
            package=package,
            class_names=", ".join(class_names),
            imports_summary=", ".join(imports),
            loc=len(source.split("\n")),
            last_commit_sha=commit_sha,
        )
