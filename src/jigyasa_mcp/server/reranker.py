"""Static ranking boosts applied to search results.

Boosts are applied post-retrieval to re-rank Jigyasa BM25 results
based on signals that BM25 doesn't capture:
  - Exact name match
  - Path proximity to the query context
  - Production code over test code
  - Recently modified files
"""

from dataclasses import dataclass

from jigyasa_mcp.grpc_client import SearchHit, SearchResult


@dataclass
class RankingConfig:
    """Configurable boost multipliers for search result reranking."""
    exact_name_boost: float = 2.0       # query exactly matches symbol name
    prod_over_test_boost: float = 1.3   # src/main/ ranked above src/test/
    test_penalty: float = 0.7           # test files ranked lower unless searching tests
    main_class_boost: float = 1.2       # file's primary class (not inner classes)
    recent_boost: float = 1.15          # files modified within recent_days
    recent_days: int = 7                # how many days counts as "recent"
    min_score: float = 0.0001           # minimum score floor to prevent zero-out


# Default config — importable for override
DEFAULT_RANKING_CONFIG = RankingConfig()


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


def _is_recently_modified(hit: SearchHit, recent_days: int = 7) -> bool:
    """Check if the file was modified within the last N days.

    Uses the 'last_modified' or 'last_commit_date' field if present in source.
    """
    for date_field in ("last_modified", "last_commit_date", "modified_date"):
        date_str = hit.source.get(date_field, "")
        if not date_str:
            continue
        try:
            # Parse ISO format: 2026-04-10T12:00:00Z
            from datetime import datetime, timezone
            dt = datetime.fromisoformat(date_str.replace("Z", "+00:00"))
            age_days = (datetime.now(timezone.utc) - dt).days
            return age_days <= recent_days
        except (ValueError, TypeError):
            continue
    return False


def rerank(
    result: SearchResult,
    query: str,
    exclude_tests: bool = False,
    config: RankingConfig | None = None,
) -> SearchResult:
    """Apply static ranking boosts to search results."""
    if not result.hits:
        return result

    cfg = config or DEFAULT_RANKING_CONFIG
    boosted_hits: list[tuple[float, SearchHit]] = []

    for hit in result.hits:
        score = hit.score
        file_path = hit.source.get("file_path", hit.source.get("path", ""))

        # Exact name match boost
        if _is_exact_name_match(query, hit):
            score *= cfg.exact_name_boost

        # Prod vs test
        if _is_test_file(file_path):
            if exclude_tests:
                continue  # Skip entirely
            score *= cfg.test_penalty
        else:
            score *= cfg.prod_over_test_boost

        # Primary class boost
        if _is_primary_class(hit):
            score *= cfg.main_class_boost

        # Recently modified boost
        if _is_recently_modified(hit, cfg.recent_days):
            score *= cfg.recent_boost

        # Enforce score floor
        score = max(cfg.min_score, score)
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
