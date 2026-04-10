"""Static ranking boosts applied to search results.

Boosts are applied post-retrieval to re-rank Jigyasa BM25 results
based on signals that BM25 doesn't capture:
  - Exact name match
  - Path proximity to the query context
  - Production code over test code
  - Recently modified files
"""

import os
import re
from typing import Optional

from jigyasa_mcp.grpc_client import SearchHit, SearchResult


# Boost multipliers (applied to BM25 score)
EXACT_NAME_BOOST = 2.0      # query exactly matches symbol name
PROD_OVER_TEST_BOOST = 1.3  # src/main/ ranked above src/test/
TEST_PENALTY = 0.7           # test files ranked lower unless explicitly searching tests
MAIN_CLASS_BOOST = 1.2       # file's primary class (not inner classes)


def _is_test_file(file_path: str) -> bool:
    parts = file_path.replace("\\", "/").lower().split("/")
    return "test" in parts or "tests" in parts


def _is_exact_name_match(query: str, hit: SearchHit) -> bool:
    """Check if query is an exact match for the symbol name."""
    query_lower = query.lower().strip()
    source = hit.source
    for field in ("name", "qualified_name", "filename"):
        val = source.get(field, "")
        if val and val.lower() == query_lower:
            return True
        # Also match unqualified: "ClusterService" matches "org...ClusterService"
        if val and val.rsplit(".", 1)[-1].lower() == query_lower:
            return True
    return False


def _is_primary_class(hit: SearchHit) -> bool:
    """Check if this is the file's primary class (not an inner class)."""
    source = hit.source
    parent = source.get("parent_class", "")
    kind = source.get("kind", "")
    return kind in ("class", "interface", "enum") and not parent


def rerank(result: SearchResult, query: str, exclude_tests: bool = False) -> SearchResult:
    """Apply static ranking boosts to search results."""
    if not result.hits:
        return result

    boosted_hits: list[tuple[float, SearchHit]] = []

    for hit in result.hits:
        score = hit.score
        file_path = hit.source.get("file_path", hit.source.get("path", ""))

        # Exact name match boost
        if _is_exact_name_match(query, hit):
            score *= EXACT_NAME_BOOST

        # Prod vs test
        if _is_test_file(file_path):
            if exclude_tests:
                continue  # Skip entirely
            score *= TEST_PENALTY
        else:
            score *= PROD_OVER_TEST_BOOST

        # Primary class boost
        if _is_primary_class(hit):
            score *= MAIN_CLASS_BOOST

        boosted_hits.append((score, hit))

    # Sort by boosted score descending
    boosted_hits.sort(key=lambda x: x[0], reverse=True)

    reranked = SearchResult(
        total_hits=result.total_hits,
        hits=[SearchHit(score=s, doc_id=h.doc_id, source=h.source)
              for s, h in boosted_hits],
        latency_ms=result.latency_ms,
    )
    return reranked
