"""Code intelligence: find_implementations, find_references, dependency_graph.

Builds on the indexed symbol data (implements, extends_class, type_references,
imports) to answer structural code questions that BM25 text search can't:

  - "What classes implement this interface?"
  - "Who references this class?"
  - "What does this file depend on?"

These queries use Jigyasa's filter API on the already-indexed fields,
combined with in-memory graph traversal for multi-hop queries.
"""

import logging
from dataclasses import dataclass, field

from jigyasa_mcp.grpc_client import JigyasaClient

logger = logging.getLogger(__name__)


@dataclass
class SymbolRef:
    """A reference to a symbol with location context."""
    name: str
    qualified_name: str
    kind: str
    file_path: str
    line_start: int
    line_end: int
    relationship: str  # "implements", "extends", "references", "imports"


@dataclass
class DependencyNode:
    """A node in the dependency graph."""
    file_path: str
    imports: list[str] = field(default_factory=list)
    imported_by: list[str] = field(default_factory=list)
    classes: list[str] = field(default_factory=list)


def find_implementations(
    client: JigyasaClient,
    symbol_name: str,
    collection: str,
) -> list[SymbolRef]:
    """Find all classes that implement an interface or extend a class.

    Searches the 'implements' and 'extends_class' fields in the symbols
    collection. Works for both Java interfaces and abstract classes, and
    Python/TS class inheritance.

    Example: find_implementations("ActionFilter") →
      [AutoTaggingActionFilter, ActionFilter.Simple, ...]
    """
    refs: list[SymbolRef] = []

    # Search implements field
    result = client.query(
        collection,
        text_query=symbol_name,
        filters=[],
        top_k=100,
    )

    for hit in result.hits:
        src = hit.source
        implements = src.get("implements", "")
        extends = src.get("extends_class", "")
        name = src.get("name", "")
        qualified = src.get("qualified_name", "")

        # Check if this symbol implements/extends the target
        target_lower = symbol_name.lower()
        relationship = None

        if implements:
            impl_list = [
                i.strip().lower() for i in implements.split(",")
            ]
            # Match full name or simple name
            for impl in impl_list:
                if (target_lower == impl
                        or target_lower == impl.rsplit(".", 1)[-1]):
                    relationship = "implements"
                    break

        if not relationship and extends:
            ext_lower = extends.lower().strip()
            if (target_lower == ext_lower
                    or target_lower == ext_lower.rsplit(".", 1)[-1]):
                relationship = "extends"

        if relationship:
            refs.append(SymbolRef(
                name=name,
                qualified_name=qualified,
                kind=src.get("kind", ""),
                file_path=src.get("file_path", ""),
                line_start=src.get("line_start", 0),
                line_end=src.get("line_end", 0),
                relationship=relationship,
            ))

    return refs


def find_references(
    client: JigyasaClient,
    symbol_name: str,
    collection: str,
) -> list[SymbolRef]:
    """Find all symbols that reference a given type.

    Searches 'type_references' field to find symbols that use a class/type,
    and 'imports' field to find files that import it.

    Example: find_references("ClusterService") →
      [ClusterStateListener.onClusterChanged, TransportAction.execute, ...]
    """
    refs: list[SymbolRef] = []
    seen = set()

    # Search type_references field
    result = client.query(
        collection,
        text_query=symbol_name,
        filters=[],
        top_k=100,
    )

    target_lower = symbol_name.lower()

    for hit in result.hits:
        src = hit.source
        qualified = src.get("qualified_name", "")
        name = src.get("name", "")

        # Skip self-references
        if name.lower() == target_lower:
            continue

        type_refs = src.get("type_references", "")
        imports = src.get("imports", "")
        ref_key = f"{src.get('file_path', '')}::{qualified}"

        if ref_key in seen:
            continue

        # Check type_references
        if type_refs:
            ref_list = [r.strip().lower() for r in type_refs.split(",")]
            if target_lower in ref_list:
                seen.add(ref_key)
                refs.append(SymbolRef(
                    name=name,
                    qualified_name=qualified,
                    kind=src.get("kind", ""),
                    file_path=src.get("file_path", ""),
                    line_start=src.get("line_start", 0),
                    line_end=src.get("line_end", 0),
                    relationship="references",
                ))
                continue

        # Check imports
        if imports and ref_key not in seen:
            import_list = [i.strip().lower() for i in imports.split(",")]
            for imp in import_list:
                if target_lower in imp:
                    seen.add(ref_key)
                    refs.append(SymbolRef(
                        name=name,
                        qualified_name=qualified,
                        kind=src.get("kind", ""),
                        file_path=src.get("file_path", ""),
                        line_start=src.get("line_start", 0),
                        line_end=src.get("line_end", 0),
                        relationship="imports",
                    ))
                    break

    return refs


def _classify_import(
    import_path: str,
    symbols_by_class: dict[str, dict],
) -> str:
    """Classify an import into a dependency edge type.

    Uses the target file's symbol data to determine the relationship:
      - EXTENDS: this file extends the imported type
      - IMPLEMENTS: this file implements the imported interface
      - ANNOTATION: the import is used as an annotation
      - IMPORT: generic import (default)
    """
    # Extract the simple class name from the import path
    parts = import_path.split(".")
    simple_name = parts[-1] if parts else import_path

    # Check if any symbol in the current file extends/implements this
    for _cls_name, sym_data in symbols_by_class.items():
        extends = sym_data.get("extends_class", "")
        implements = sym_data.get("implements", "")
        annotations = sym_data.get("annotations", "")

        if extends and simple_name in extends:
            return "EXTENDS"
        if implements and simple_name in implements:
            return "IMPLEMENTS"
        if annotations and simple_name in annotations:
            return "ANNOTATION"

    return "IMPORT"


def _resolve_import_to_file(
    client: JigyasaClient,
    import_path: str,
    files_collection: str,
) -> str:
    """Resolve an import path to an actual file path in the index.

    E.g., 'org.opensearch.cluster.ClusterState' → 'server/src/.../ClusterState.java'
    Uses exact TermFilter on package field + BM25 on class name for precision.
    Returns the file path if found, empty string otherwise.
    """
    parts = import_path.split(".")
    if not parts or not import_path.strip():
        return ""
    # The class name is usually the last segment (or second-to-last for inner classes)
    class_name = parts[-1]
    if class_name == "*":
        return ""  # Can't resolve wildcard imports

    # Strategy: use exact package filter + class name text query
    # This is much more precise than BM25-only search
    expected_package = ".".join(parts[:-1])

    # Try exact package + class name match first
    if expected_package:
        result = client.query(
            files_collection,
            text_query=class_name,
            filters=[{"field": "package", "value": expected_package}],
            top_k=5,
        )
        for hit in result.hits:
            src = hit.source
            file_classes = src.get("class_names", "")
            class_list = [c.strip() for c in file_classes.split(",")]
            if class_name in class_list:
                return src.get("path", "")

    # Inner class fallback: org.opensearch.common.settings.Setting.Property
    # → package=org.opensearch.common.settings, class=Setting
    if len(parts) >= 3:
        parent_name = parts[-2]
        parent_package = ".".join(parts[:-2])
        if parent_package:
            result = client.query(
                files_collection,
                text_query=parent_name,
                filters=[{"field": "package", "value": parent_package}],
                top_k=5,
            )
            for hit in result.hits:
                src = hit.source
                file_classes = src.get("class_names", "")
                class_list = [c.strip() for c in file_classes.split(",")]
                if parent_name in class_list:
                    return src.get("path", "")

    return ""


def dependency_graph(
    client: JigyasaClient,
    file_path: str,
    files_collection: str,
    symbols_collection: str,
    depth: int = 1,
) -> dict:
    """Build a dependency graph for a file.

    Shows what a file depends on (imports/type references) and what depends
    on it (reverse references). Supports multi-hop traversal via depth param.

    Returns a dict with:
      - target: the queried file
      - depends_on: list of {import, type, file} dicts with typed edges
      - depended_by: files that reference types defined in this file
      - classes_defined: classes/interfaces/enums in this file
      - completeness: "full" or "truncated"
    """
    # Get file metadata — try exact path filter first, fall back to BM25
    file_result = client.query(
        files_collection,
        text_query="",
        filters=[{"field": "path", "value": file_path}],
        top_k=1,
    )

    target_info = None
    if file_result.hits:
        target_info = file_result.hits[0].source

    if not target_info:
        # Fallback: BM25 search for partial path match
        file_result = client.query(
            files_collection,
            text_query=file_path,
            filters=[],
            top_k=5,
        )
        for hit in file_result.hits:
            if hit.source.get("path", "") == file_path:
                target_info = hit.source
                break
        if not target_info:
            for hit in file_result.hits:
                if file_path in hit.source.get("path", ""):
                    target_info = hit.source
                    file_path = hit.source["path"]
                    break

    if not target_info:
        return {"error": f"File not found: {file_path}"}

    imports_raw = target_info.get("imports_full", "") or target_info.get("imports_summary", "")
    classes_raw = target_info.get("class_names", "")

    import_list = [
        i.strip() for i in imports_raw.split(",") if i.strip()
    ]
    classes_defined = [
        c.strip() for c in classes_raw.split(",") if c.strip()
    ]

    # Get symbols for this file to classify edge types
    # Use exact TermFilter on file_path for precision (not BM25 text search)
    symbols_by_class: dict[str, dict] = {}
    if classes_defined:
        sym_result = client.query(
            symbols_collection,
            text_query="",
            filters=[{"field": "file_path", "value": file_path}],
            top_k=50,
        )
        for hit in sym_result.hits:
            src = hit.source
            if src.get("kind", "") in ("class", "interface", "enum"):
                symbols_by_class[src.get("name", "")] = src

    # Build typed dependency edges
    depends_on: list[dict] = []
    for imp in import_list:
        edge_type = _classify_import(imp, symbols_by_class)
        resolved_file = _resolve_import_to_file(client, imp, files_collection)
        dep = {"import": imp, "type": edge_type}
        if resolved_file:
            dep["file"] = resolved_file
        depends_on.append(dep)

    # Find files that depend on this file's classes
    depended_by: list[dict] = []
    seen_files = set()

    for class_name in classes_defined:
        refs = find_references(client, class_name, symbols_collection)
        for ref in refs:
            if ref.file_path != file_path and ref.file_path not in seen_files:
                seen_files.add(ref.file_path)
                depended_by.append({
                    "file": ref.file_path,
                    "references": class_name,
                    "via": f"{ref.qualified_name} ({ref.relationship})",
                })

    # Multi-hop: if depth > 1, recurse into depended_by files
    transitive_deps: list[dict] = []
    if depth > 1:
        for dep in depended_by[:10]:  # cap to prevent explosion
            sub_result = dependency_graph(
                client, dep["file"],
                files_collection, symbols_collection,
                depth=depth - 1,
            )
            if "error" not in sub_result:
                transitive_deps.append({
                    "file": dep["file"],
                    "further_depended_by": len(
                        sub_result.get("depended_by", [])
                    ),
                })

    result = {
        "target": file_path,
        "classes_defined": classes_defined,
        "depends_on": depends_on,
        "depends_on_count": len(depends_on),
        "depended_by": depended_by,
        "depended_by_count": len(depended_by),
        "completeness": "full",
    }
    if transitive_deps:
        result["transitive"] = transitive_deps
    return result


# ---------------------------------------------------------------------------
# Formatters for MCP tool responses
# ---------------------------------------------------------------------------


def format_implementations(
    refs: list[SymbolRef], symbol_name: str,
) -> str:
    if not refs:
        return f"No implementations found for '{symbol_name}'. (0 results, completeness: full)"
    lines = [
        f"Found {len(refs)} implementation(s) of '{symbol_name}':",
    ]
    for ref in refs:
        lines.append(
            f"  [{ref.relationship}] {ref.qualified_name}"
            f"  ({ref.kind}, {ref.file_path}:{ref.line_start})"
        )
    lines.append(f"\nTotal: {len(refs)}, completeness: full")
    return "\n".join(lines)


def format_references(
    refs: list[SymbolRef], symbol_name: str,
) -> str:
    if not refs:
        return f"No references found for '{symbol_name}'. (0 results, completeness: full)"
    lines = [
        f"Found {len(refs)} reference(s) to '{symbol_name}':",
    ]
    for ref in refs:
        lines.append(
            f"  [{ref.relationship}] {ref.qualified_name}"
            f"  ({ref.kind}, {ref.file_path}:{ref.line_start})"
        )
    lines.append(f"\nTotal: {len(refs)}, completeness: full")
    return "\n".join(lines)


def format_dependency_graph(graph: dict) -> str:
    if "error" in graph:
        return f"ERROR: {graph['error']}"

    lines = [f"Dependency graph for: {graph['target']}"]

    classes = graph.get("classes_defined", [])
    if classes:
        lines.append(f"\nDefines: {', '.join(classes)}")

    deps = graph.get("depends_on", [])
    dep_count = graph.get("depends_on_count", len(deps))
    if deps:
        lines.append(f"\nDepends on ({dep_count}):")
        for d in deps:
            if isinstance(d, dict):
                edge_type = d.get("type", "IMPORT")
                imp = d.get("import", "?")
                resolved = d.get("file", "")
                marker = f"[{edge_type}]"
                if resolved:
                    lines.append(f"  {marker} {imp} → {resolved}")
                else:
                    lines.append(f"  {marker} {imp}")
            else:
                # Backward compat: old-style string list
                lines.append(f"  → {d}")
    else:
        lines.append("\nDepends on: (none)")

    dep_by = graph.get("depended_by", [])
    dep_by_count = graph.get("depended_by_count", len(dep_by))
    if dep_by:
        lines.append(f"\nDepended by ({dep_by_count}):")
        for d in dep_by:
            lines.append(f"  ← {d['file']}  (via {d['via']})")
    else:
        lines.append("\nDepended by: (none)")

    transitive = graph.get("transitive", [])
    if transitive:
        lines.append("\nTransitive impact:")
        for t in transitive:
            lines.append(
                f"  {t['file']} → "
                f"{t['further_depended_by']} further dependents"
            )

    completeness = graph.get("completeness", "unknown")
    lines.append(f"\nCompleteness: {completeness}")

    return "\n".join(lines)
