"""gRPC client wrapper for Jigyasa search engine.

Thread-safe, with circuit breaker, exponential backoff retries,
channel health verification, and automatic reconnection.
"""

import json
import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Optional

import grpc

logger = logging.getLogger(__name__)

_stubs_loaded = False
_stubs_lock = threading.Lock()
_pb2 = None
_pb2_grpc = None

# gRPC status codes that are safe to retry
_RETRYABLE_CODES = frozenset({
    grpc.StatusCode.UNAVAILABLE,
    grpc.StatusCode.DEADLINE_EXCEEDED,
    grpc.StatusCode.RESOURCE_EXHAUSTED,
})

MAX_RETRIES = 3
INITIAL_BACKOFF_S = 0.2
CHANNEL_READY_TIMEOUT_S = 5.0


def _load_stubs():
    """Load generated protobuf stubs (thread-safe)."""
    global _stubs_loaded, _pb2, _pb2_grpc
    if _stubs_loaded:
        return
    with _stubs_lock:
        if _stubs_loaded:
            return
        try:
            from jigyasa_mcp import dpSearch_pb2 as pb2
            from jigyasa_mcp import dpSearch_pb2_grpc as pb2_grpc
            _pb2 = pb2
            _pb2_grpc = pb2_grpc
            _stubs_loaded = True
        except ImportError:
            raise RuntimeError(
                "gRPC stubs not generated. Run: python build_proto.py"
            )


@dataclass
class SearchHit:
    score: float
    doc_id: str
    source: dict = field(default_factory=dict)


@dataclass
class SearchResult:
    total_hits: int
    hits: list[SearchHit]
    latency_ms: float = 0.0


class CircuitBreaker:
    """Thread-safe circuit breaker for gRPC connection failures."""

    def __init__(self, failure_threshold: int = 3, reset_timeout: float = 30.0):
        self._lock = threading.Lock()
        self.failure_threshold = failure_threshold
        self.reset_timeout = reset_timeout
        self._failures = 0
        self._last_failure_time = 0.0
        self._is_open = False

    def record_failure(self):
        with self._lock:
            self._failures += 1
            self._last_failure_time = time.time()
            if self._failures >= self.failure_threshold:
                self._is_open = True
                logger.warning("Circuit breaker OPEN — Jigyasa unreachable")

    def record_success(self):
        with self._lock:
            self._failures = 0
            self._is_open = False

    def allow_request(self) -> bool:
        with self._lock:
            if not self._is_open:
                return True
            if time.time() - self._last_failure_time > self.reset_timeout:
                self._is_open = False
                self._failures = 0
                logger.info("Circuit breaker HALF-OPEN — retrying")
                return True
            return False

    @property
    def is_open(self) -> bool:
        with self._lock:
            return self._is_open


class JigyasaClient:
    """Thread-safe gRPC client with lazy connect, circuit breaker, retries, and auto-reconnect."""

    def __init__(self, endpoint: str = "localhost:50051", timeout: float = 10.0):
        self.endpoint = endpoint
        self.timeout = timeout
        self._lock = threading.Lock()
        self._channel: Optional[grpc.Channel] = None
        self._stub = None
        self._breaker = CircuitBreaker()

    def _ensure_connected(self):
        if not self._breaker.allow_request():
            raise ConnectionError(
                f"Jigyasa at {self.endpoint} is unreachable (circuit breaker open)"
            )
        with self._lock:
            if self._channel is not None:
                return
            _load_stubs()
            self._channel = grpc.insecure_channel(self.endpoint)
            # Verify channel is actually reachable
            try:
                grpc.channel_ready_future(self._channel).result(
                    timeout=CHANNEL_READY_TIMEOUT_S
                )
            except grpc.FutureTimeoutError:
                self._channel.close()
                self._channel = None
                self._breaker.record_failure()
                raise ConnectionError(
                    f"Jigyasa at {self.endpoint} not reachable within "
                    f"{CHANNEL_READY_TIMEOUT_S}s"
                )
            self._stub = _pb2_grpc.JigyasaDataPlaneServiceStub(self._channel)

    def _reconnect(self):
        """Force close and reconnect on next call."""
        with self._lock:
            if self._channel is not None:
                try:
                    self._channel.close()
                except Exception:
                    pass
                self._channel = None
                self._stub = None

    def _call(self, method_name: str, request):
        """Execute a gRPC call with timeout, retries with exponential backoff,
        circuit breaker, and auto-reconnect on transient failures."""
        self._ensure_connected()
        method = getattr(self._stub, method_name)

        last_error = None
        for attempt in range(MAX_RETRIES + 1):
            try:
                response = method(request, timeout=self.timeout)
                self._breaker.record_success()
                return response
            except grpc.RpcError as e:
                last_error = e
                code = e.code()

                if code == grpc.StatusCode.UNAVAILABLE:
                    # Server may have restarted — force reconnect
                    self._reconnect()

                if code in _RETRYABLE_CODES and attempt < MAX_RETRIES:
                    backoff = INITIAL_BACKOFF_S * (2 ** attempt)
                    logger.warning(
                        f"gRPC {method_name} failed ({code}), "
                        f"retry {attempt + 1}/{MAX_RETRIES} in {backoff:.1f}s"
                    )
                    time.sleep(backoff)
                    self._ensure_connected()
                    method = getattr(self._stub, method_name)
                    continue

                # Non-retryable or exhausted retries
                self._breaker.record_failure()
                raise ConnectionError(
                    f"Jigyasa gRPC error ({method_name}): {code} — {e.details()}"
                ) from e

        # Should not reach here, but safety net
        self._breaker.record_failure()
        raise ConnectionError(
            f"Jigyasa gRPC {method_name} failed after {MAX_RETRIES} retries"
        ) from last_error

    def health(self) -> dict:
        self._ensure_connected()
        resp = self._call("Health", _pb2.HealthRequest())
        return {
            "status": "SERVING" if resp.status == 0 else "NOT_SERVING",
            "collections": [
                {
                    "name": c.name,
                    "doc_count": c.doc_count,
                    "segment_count": c.segment_count,
                    "writer_open": c.writer_open,
                    "searcher_available": c.searcher_available,
                }
                for c in resp.collections
            ],
        }

    def create_collection(self, name: str, schema_json: str):
        self._call(
            "CreateCollection",
            _pb2.CreateCollectionRequest(collection=name, indexSchema=schema_json),
        )
        logger.info(f"Created collection: {name}")

    def open_collection(self, name: str, schema_json: str = ""):
        """Reopen a persisted collection after Jigyasa restart."""
        self._call(
            "OpenCollection",
            _pb2.OpenCollectionRequest(collection=name, indexSchema=schema_json),
        )
        logger.info(f"Opened collection: {name}")

    def index_batch(
        self,
        collection: str,
        documents: list[dict],
        refresh: str = "NONE",
    ) -> int:
        """Index a batch of documents. Returns count indexed."""
        _load_stubs()
        refresh_policy = {
            "NONE": _pb2.NONE,
            "WAIT_FOR": _pb2.WAIT_FOR,
            "IMMEDIATE": _pb2.IMMEDIATE,
        }.get(refresh, _pb2.NONE)

        items = [
            _pb2.IndexItem(document=json.dumps(doc)) for doc in documents
        ]
        self._call(
            "Index",
            _pb2.IndexRequest(
                collection=collection, item=items, refresh=refresh_policy
            ),
        )
        return len(documents)

    @staticmethod
    def _build_filters(filters: list[dict]) -> list:
        """Build gRPC FilterClause list from dict filters."""
        _load_stubs()
        clauses = []
        for f in filters:
            clause = _pb2.FilterClause(field=f["field"])
            if "value" in f:
                clause.term_filter.CopyFrom(_pb2.TermFilter(value=f["value"]))
            elif "min" in f or "max" in f:
                clause.range_filter.CopyFrom(
                    _pb2.RangeFilter(min=f.get("min", ""), max=f.get("max", ""))
                )
            clauses.append(clause)
        return clauses

    def query(
        self,
        collection: str,
        text_query: str = "",
        filters: Optional[list[dict]] = None,
        top_k: int = 20,
        include_source: bool = True,
        vector: Optional[list[float]] = None,
        vector_field: str = "embedding",
        text_weight: float = 0.5,
    ) -> SearchResult:
        """Search a collection with BM25 and optional KNN vector search."""
        start = time.time()

        _load_stubs()
        request = _pb2.QueryRequest(
            collection=collection,
            text_query=text_query,
            top_k=top_k,
            include_source=include_source,
        )

        if filters:
            request.filters.extend(self._build_filters(filters))

        if vector:
            request.vector_query.CopyFrom(
                _pb2.VectorQuery(field=vector_field, vector=vector, k=top_k)
            )
            request.text_weight = text_weight

        resp = self._call("Query", request)
        elapsed = (time.time() - start) * 1000

        hits = []
        for h in resp.hits:
            source = {}
            if h.source:
                try:
                    source = json.loads(h.source)
                except json.JSONDecodeError:
                    source = {"_raw": h.source}
            hits.append(SearchHit(score=h.score, doc_id=h.doc_id, source=source))

        return SearchResult(
            total_hits=resp.total_hits, hits=hits, latency_ms=elapsed
        )

    def delete_by_query(self, collection: str, filters: list[dict]):
        """Delete documents matching filters."""
        _load_stubs()
        request = _pb2.DeleteByQueryRequest(collection=collection)
        request.filters.extend(self._build_filters(filters))
        self._call("DeleteByQuery", request)

    def count(self, collection: str) -> int:
        self._ensure_connected()
        resp = self._call("Count", _pb2.CountRequest(collection=collection))
        return resp.count

    def close(self):
        """Gracefully close the gRPC channel."""
        with self._lock:
            if self._channel:
                try:
                    self._channel.close()
                except Exception:
                    pass
                self._channel = None
                self._stub = None
        logger.info("JigyasaClient closed")

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
