"""Phase 2: Local embedding generation using all-MiniLM-L6-v2.

This module is lazily loaded — the 23MB model is only downloaded
on first use. Falls back gracefully if sentence-transformers is
not installed.
"""

import logging
import threading

logger = logging.getLogger(__name__)

_model = None
_available: bool | None = None
_model_lock = threading.Lock()


def is_available() -> bool:
    """Check if embedding model can be loaded."""
    global _available
    if _available is not None:
        return _available
    try:
        from sentence_transformers import SentenceTransformer  # noqa: F401
        _available = True
    except ImportError:
        _available = False
        logger.info(
            "sentence-transformers not installed — embeddings disabled. "
            "Install with: pip install sentence-transformers"
        )
    return _available


def _ensure_model():
    global _model
    if _model is not None:
        return
    with _model_lock:
        if _model is not None:  # double-check after acquiring lock
            return
        if not is_available():
            raise RuntimeError("sentence-transformers not installed")
        from sentence_transformers import SentenceTransformer
        logger.info("Loading embedding model: all-MiniLM-L6-v2 (23MB, first time may download)")
        _model = SentenceTransformer("all-MiniLM-L6-v2")
        logger.info("Embedding model loaded (384 dimensions)")


def embed_texts(texts: list[str], batch_size: int = 64) -> list[list[float]]:
    """Generate embeddings for a list of texts. Returns list of 384-dim vectors."""
    _ensure_model()
    embeddings = _model.encode(texts, batch_size=batch_size, show_progress_bar=False)
    return [e.tolist() for e in embeddings]


def embed_single(text: str) -> list[float]:
    """Generate embedding for a single text."""
    return embed_texts([text])[0]


def get_dimensions() -> int:
    return 384
