"""
Test suite for the main FastAPI application.

Tests the core application functionality including health endpoints,
middleware, and error handling.
"""

import pytest
from fastapi.testclient import TestClient
from httpx import AsyncClient

from main import app


class TestHealthEndpoints:
    """Test health check endpoints."""

    def test_root_endpoint(self, client: TestClient):
        """Test the root endpoint returns expected response."""
        response = client.get("/")
        assert response.status_code == 200

        data = response.json()
        assert data["message"] == "Cerby Identity Automation Platform API"
        assert data["status"] == "operational"
        assert data["version"] == "1.0.0"
        assert "docs" in data
        assert "health" in data

    def test_health_check_endpoint(self, client: TestClient):
        """Test the health check endpoint."""
        response = client.get("/health")
        assert response.status_code == 200

        data = response.json()
        assert data["status"] == "healthy"
        assert "timestamp" in data
        assert data["service"] == "cerby-api"
        assert data["environment"] == "testing"  # Set in conftest.py
        assert data["version"] == "1.0.0"

        # Check all health checks
        assert "checks" in data
        assert data["checks"]["api"] == "operational"
        assert data["checks"]["database"] == "operational"
        assert data["checks"]["cache"] == "operational"
        assert data["checks"]["genetic_algorithm"] == "operational"

    @pytest.mark.asyncio
    async def test_root_endpoint_async(self, async_client: AsyncClient):
        """Test the root endpoint with async client."""
        response = await async_client.get("/")
        assert response.status_code == 200

        data = response.json()
        assert data["message"] == "Cerby Identity Automation Platform API"
        assert data["status"] == "operational"

    @pytest.mark.asyncio
    async def test_health_check_async(self, async_client: AsyncClient):
        """Test the health check endpoint with async client."""
        response = await async_client.get("/health")
        assert response.status_code == 200

        data = response.json()
        assert data["status"] == "healthy"


class TestMiddleware:
    """Test application middleware."""

    def test_cors_headers(self, client: TestClient):
        """Test CORS headers are properly set."""
        response = client.get("/")

        # Check CORS headers
        assert "access-control-allow-origin" in response.headers
        assert response.headers["access-control-allow-origin"] == "*"

    def test_process_time_header(self, client: TestClient):
        """Test that process time header is added to responses."""
        response = client.get("/")

        assert "x-process-time" in response.headers
        process_time = float(response.headers["x-process-time"])
        assert process_time > 0
        assert process_time < 1  # Should be fast

    def test_request_id_header(self, client: TestClient):
        """Test that request ID header is added to responses."""
        response = client.get("/")

        assert "x-request-id" in response.headers
        request_id = response.headers["x-request-id"]
        assert len(request_id) == 36  # UUID4 format with dashes

    def test_options_request(self, client: TestClient):
        """Test OPTIONS request for CORS preflight."""
        response = client.options("/")
        assert response.status_code == 200


class TestErrorHandling:
    """Test error handling and exception middleware."""

    def test_404_not_found(self, client: TestClient):
        """Test 404 error handling."""
        response = client.get("/non-existent-endpoint")
        assert response.status_code == 404

        data = response.json()
        assert "error" in data
        assert "timestamp" in data

    def test_method_not_allowed(self, client: TestClient):
        """Test 405 method not allowed."""
        response = client.post("/health")  # Health endpoint only accepts GET
        assert response.status_code == 405

    def test_invalid_json_body(self, client: TestClient):
        """Test handling of invalid JSON in request body."""
        # This test would be more relevant once we have POST endpoints
        # For now, we'll skip it
        pass


class TestApplicationConfiguration:
    """Test application configuration and metadata."""

    def test_openapi_schema(self, client: TestClient):
        """Test OpenAPI schema generation."""
        response = client.get("/api/openapi.json")
        assert response.status_code == 200

        schema = response.json()
        assert schema["info"]["title"] == "Cerby Identity Automation Platform"
        assert schema["info"]["version"] == "1.0.0"
        assert "description" in schema["info"]

    def test_docs_endpoint(self, client: TestClient):
        """Test API documentation endpoint."""
        response = client.get("/api/docs")
        assert response.status_code == 200
        assert "text/html" in response.headers["content-type"]

    def test_redoc_endpoint(self, client: TestClient):
        """Test ReDoc documentation endpoint."""
        response = client.get("/api/redoc")
        assert response.status_code == 200
        assert "text/html" in response.headers["content-type"]


class TestPerformance:
    """Test performance-related aspects."""

    def test_health_endpoint_performance(self, client: TestClient, performance_timer):
        """Test health endpoint responds quickly."""
        performance_timer.start()
        response = client.get("/health")
        performance_timer.stop()

        assert response.status_code == 200
        assert performance_timer.elapsed() < 0.1  # Should respond in less than 100ms

    def test_concurrent_requests(self, client: TestClient):
        """Test handling of concurrent requests."""
        import concurrent.futures

        def make_request():
            return client.get("/health")

        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(make_request) for _ in range(10)]
            results = [future.result() for future in concurrent.futures.as_completed(futures)]

        assert len(results) == 10
        assert all(r.status_code == 200 for r in results)


class TestLogfireIntegration:
    """Test Logfire observability integration."""

    def test_logfire_instrumentation(self, client: TestClient, mock_logfire):
        """Test that Logfire is properly instrumenting requests."""
        # Make a request
        response = client.get("/health")
        assert response.status_code == 200

        # Verify Logfire was called (this is mocked in tests)
        # In real tests, we'd check for proper span creation
        # For now, we just ensure the app doesn't crash with Logfire enabled

    def test_error_logging(self, client: TestClient, mock_logfire):
        """Test that errors are properly logged to Logfire."""
        # Trigger a 404 error
        response = client.get("/non-existent")
        assert response.status_code == 404

        # In production, this would log to Logfire
        # Here we just ensure it doesn't crash
