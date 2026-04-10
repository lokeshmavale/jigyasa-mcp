"""Jigyasa collection schemas for code search."""

import json

SYMBOLS_SCHEMA = {
    "fields": [
        {"name": "id", "type": "STRING", "key": True},
        {"name": "name", "type": "STRING", "searchable": True},
        {"name": "qualified_name", "type": "STRING", "searchable": True},
        {"name": "signature", "type": "STRING", "searchable": True},
        {"name": "kind", "type": "STRING", "filterable": True},
        {"name": "visibility", "type": "STRING", "filterable": True},
        {"name": "file_path", "type": "STRING", "searchable": True, "filterable": True},
        {"name": "package", "type": "STRING", "searchable": True, "filterable": True},
        {"name": "module", "type": "STRING", "filterable": True},
        {"name": "parent_class", "type": "STRING", "filterable": True},
        {"name": "implements", "type": "STRING", "searchable": True},
        {"name": "extends_class", "type": "STRING", "searchable": True},
        {"name": "annotations", "type": "STRING", "searchable": True},
        {"name": "line_start", "type": "INT32"},
        {"name": "line_end", "type": "INT32"},
        {"name": "body_preview", "type": "STRING", "searchable": True},
        {"name": "imports", "type": "STRING", "searchable": True},
        {"name": "type_references", "type": "STRING", "searchable": True},
    ]
}

CHUNKS_SCHEMA_BM25 = {
    "fields": [
        {"name": "id", "type": "STRING", "key": True},
        {"name": "content", "type": "STRING", "searchable": True},
        {"name": "file_path", "type": "STRING", "searchable": True, "filterable": True},
        {"name": "symbol_name", "type": "STRING", "searchable": True},
        {"name": "kind", "type": "STRING", "filterable": True},
        {"name": "module", "type": "STRING", "filterable": True},
        {"name": "language", "type": "STRING", "filterable": True},
        {"name": "enclosing_class", "type": "STRING", "searchable": True, "filterable": True},
        {"name": "enclosing_method", "type": "STRING", "filterable": True},
        {"name": "line_start", "type": "INT32"},
        {"name": "line_end", "type": "INT32"},
        {"name": "token_count", "type": "INT32"},
    ]
}

# Phase 2: Same schema but with a VECTOR field for hybrid search
CHUNKS_SCHEMA_HYBRID = {
    "fields": [
        *CHUNKS_SCHEMA_BM25["fields"],
        {"name": "embedding", "type": "VECTOR", "dimensions": 384},
    ]
}

FILES_SCHEMA = {
    "fields": [
        {"name": "id", "type": "STRING", "key": True},
        {"name": "path", "type": "STRING", "searchable": True},
        {"name": "filename", "type": "STRING", "searchable": True},
        {"name": "extension", "type": "STRING", "filterable": True},
        {"name": "module", "type": "STRING", "searchable": True, "filterable": True},
        {"name": "package", "type": "STRING", "searchable": True},
        {"name": "class_names", "type": "STRING", "searchable": True},
        {"name": "imports_summary", "type": "STRING", "searchable": True},
        {"name": "loc", "type": "INT32"},
        {"name": "last_commit_sha", "type": "STRING"},
    ]
}


def get_schema_json(name: str, use_embeddings: bool = False) -> str:
    schemas = {
        "symbols": SYMBOLS_SCHEMA,
        "chunks": CHUNKS_SCHEMA_HYBRID if use_embeddings else CHUNKS_SCHEMA_BM25,
        "files": FILES_SCHEMA,
    }
    return json.dumps(schemas[name])
