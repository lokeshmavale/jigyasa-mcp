"""Client-side search result highlighting.

Applies query term highlighting to search results without requiring
a Jigyasa-side highlight API. Uses simple regex matching on returned
source documents.

When Jigyasa adds a native Highlight RPC (Lucene UnifiedHighlighter),
this module can be swapped for server-side highlighting.
"""

import re
from functools import lru_cache


@lru_cache(maxsize=128)
def _compile_pattern(pattern_key: str) -> re.Pattern:
    """Cache compiled regex patterns to avoid recompilation per search."""
    return re.compile(f"({pattern_key})", re.IGNORECASE)


def highlight_matches(
    text: str,
    query: str,
    max_snippet_length: int = 300,
    context_chars: int = 50,
    marker_start: str = "**",
    marker_end: str = "**",
) -> list[str]:
    """Extract highlighted snippets from text matching query terms.

    Returns a list of snippets with query terms wrapped in markers.
    """
    terms = _tokenize_query(query)
    if not terms:
        return [text[:max_snippet_length]]

    # Build regex pattern matching any query term (cached)
    pattern = "|".join(re.escape(t) for t in terms)
    regex = _compile_pattern(pattern)

    matches = list(regex.finditer(text))
    if not matches:
        return [text[:max_snippet_length]]

    snippets = []
    seen_positions = set()

    for match in matches:
        start = max(0, match.start() - context_chars)
        end = min(len(text), match.end() + context_chars)

        # Skip overlapping snippets
        pos_key = start // context_chars
        if pos_key in seen_positions:
            continue
        seen_positions.add(pos_key)

        snippet = text[start:end]
        # Highlight all terms in this snippet
        snippet = regex.sub(f"{marker_start}\\1{marker_end}", snippet)

        # Clean up: trim to word boundaries
        if start > 0:
            snippet = "..." + snippet
        if end < len(text):
            snippet = snippet + "..."

        snippets.append(snippet)

        if len(snippets) >= 3:
            break

    return snippets


def highlight_search_result(source: dict, query: str) -> str:
    """Format a search result with highlighted matching snippets."""
    content = source.get("content", source.get("body_preview", ""))
    if not content:
        return ""

    snippets = highlight_matches(content, query)
    return "\n    ".join(snippets)


def _tokenize_query(query: str) -> list[str]:
    """Split query into meaningful search terms (drop stopwords)."""
    stopwords = {
        "the", "a", "an", "is", "are", "was", "were", "be", "been",
        "in", "on", "at", "to", "for", "of", "with", "by", "from",
        "and", "or", "not", "this", "that", "it", "how", "what",
        "where", "when", "why", "which", "do", "does", "did",
    }
    words = re.findall(r"\w+", query.lower())
    return [w for w in words if w not in stopwords and len(w) > 1]
