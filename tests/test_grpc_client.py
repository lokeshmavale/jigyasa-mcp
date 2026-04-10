"""Unit tests for the gRPC client — circuit breaker and retry logic."""

import threading
import time

import pytest

from jigyasa_mcp.grpc_client import CircuitBreaker


class TestCircuitBreaker:
    def test_starts_closed(self):
        cb = CircuitBreaker(failure_threshold=3)
        assert cb.allow_request() is True
        assert cb.is_open is False

    def test_opens_after_threshold(self):
        cb = CircuitBreaker(failure_threshold=3)
        cb.record_failure()
        cb.record_failure()
        assert cb.allow_request() is True  # not yet
        cb.record_failure()
        assert cb.is_open is True
        assert cb.allow_request() is False

    def test_resets_on_success(self):
        cb = CircuitBreaker(failure_threshold=2)
        cb.record_failure()
        cb.record_success()
        assert cb.is_open is False
        cb.record_failure()
        assert cb.allow_request() is True  # counter reset

    def test_half_open_after_timeout(self):
        cb = CircuitBreaker(failure_threshold=1, reset_timeout=0.1)
        cb.record_failure()
        assert cb.is_open is True
        time.sleep(0.15)
        assert cb.allow_request() is True  # half-open

    def test_thread_safety(self):
        """Concurrent record_failure calls shouldn't corrupt state."""
        cb = CircuitBreaker(failure_threshold=100)
        errors = []

        def hammer():
            try:
                for _ in range(1000):
                    cb.record_failure()
                    cb.record_success()
                    cb.allow_request()
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=hammer) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert errors == [], f"Thread safety violated: {errors}"


class TestJigyasaClientInit:
    """Tests that don't require a running Jigyasa instance."""

    def test_default_endpoint(self):
        from jigyasa_mcp.grpc_client import JigyasaClient
        client = JigyasaClient()
        assert client.endpoint == "localhost:50051"
        assert client.timeout == 10.0

    def test_custom_endpoint(self):
        from jigyasa_mcp.grpc_client import JigyasaClient
        client = JigyasaClient(endpoint="remote:9090", timeout=5.0)
        assert client.endpoint == "remote:9090"
        assert client.timeout == 5.0

    def test_context_manager(self):
        from jigyasa_mcp.grpc_client import JigyasaClient
        with JigyasaClient() as client:
            assert client.endpoint == "localhost:50051"
        # Channel should be cleaned up

    def test_connection_error_on_unreachable(self):
        from jigyasa_mcp.grpc_client import JigyasaClient
        client = JigyasaClient(endpoint="localhost:99999", timeout=1.0)
        with pytest.raises((ConnectionError, RuntimeError)):
            client.health()
