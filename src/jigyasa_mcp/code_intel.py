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
      - depends_on: files/packages this file imports
      - depended_by: files that reference types defined in this file
      - classes_defined: classes/interfaces/enums in this file
    """
    # Get file metadata
    file_result = client.query(
        files_collection,
        text_query=file_path,
        filters=[],
        top_k=5,
    )

    target_info = None
    for hit in file_result.hits:
        if hit.source.get("path", "") == file_path:
            target_info = hit.source
            break

    if not target_info:
        # Try partial match
        for hit in file_result.hits:
            if file_path in hit.source.get("path", ""):
                target_info = hit.source
                file_path = hit.source["path"]
                break

    if not target_info:
        return {"error": f"File not found: {file_path}"}

    imports_raw = target_info.get("imports_full", "") or target_info.get("imports_summary", "")
    classes_raw = target_info.get("class_names", "")

    depends_on = [
        i.strip() for i in imports_raw.split(",") if i.strip()
    ]
    classes_defined = [
        c.strip() for c in classes_raw.split(",") if c.strip()
    ]

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
        "depended_by": depended_by,
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
        return f"No implementations found for '{symbol_name}'."
    lines = [
        f"Found {len(refs)} implementation(s) of '{symbol_name}':",
    ]
    for ref in refs:
        lines.append(
            f"  [{ref.relationship}] {ref.qualified_name}"
            f"  ({ref.kind}, {ref.file_path}:{ref.line_start})"
        )
    return "\n".join(lines)


def format_references(
    refs: list[SymbolRef], symbol_name: str,
) -> str:
    if not refs:
        return f"No references found for '{symbol_name}'."
    lines = [
        f"Found {len(refs)} reference(s) to '{symbol_name}':",
    ]
    for ref in refs:
        lines.append(
            f"  [{ref.relationship}] {ref.qualified_name}"
            f"  ({ref.kind}, {ref.file_path}:{ref.line_start})"
        )
    return "\n".join(lines)


def format_dependency_graph(graph: dict) -> str:
    if "error" in graph:
        return f"ERROR: {graph['error']}"

    lines = [f"Dependency graph for: {graph['target']}"]

    classes = graph.get("classes_defined", [])
    if classes:
        lines.append(f"\nDefines: {', '.join(classes)}")

    deps = graph.get("depends_on", [])
    if deps:
        lines.append(f"\nDepends on ({len(deps)}):")
        for d in deps:
            lines.append(f"  → {d}")
    else:
        lines.append("\nDepends on: (none)")

    dep_by = graph.get("depended_by", [])
    if dep_by:
        lines.append(f"\nDepended by ({len(dep_by)}):")
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

    return "\n".join(lines)
