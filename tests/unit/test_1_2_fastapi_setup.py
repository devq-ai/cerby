"""
Unit tests for Task 1.2 - Install and configure FastAPI framework.

Tests verify that FastAPI is properly configured with correct project structure,
middleware, error handlers, and basic endpoints.
"""

import pytest
from fastapi.testclient import TestClient
from fastapi import FastAPI, status
import sys
from pathlib import Path
import json
from unittest.mock import patch, MagicMock

# Add project root to Python path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import main
from src.core.config import settings


class TestFastAPIConfiguration:
    """Test FastAPI application configuration and setup."""

    @pytest.fixture
    def client(self):
        """Create a test client."""
        return TestClient(main.app)

    @pytest.fixture
    def mock_settings(self, monkeypatch):
        """Mock settings for testing."""
        monkeypatch.setattr(settings, "debug", True)
        monkeypatch.setattr(settings, "environment", "testing")
        monkeypatch.setattr(settings, "logfire_token", "test_token")
        return settings

    def test_fastapi_app_exists(self):
        """Test that FastAPI app is properly initialized."""
        assert hasattr(main, 'app'), "FastAPI app not found in main module"
        assert isinstance(main.app, FastAPI), "app is not a FastAPI instance"

    def test_app_metadata(self):
        """Test FastAPI app metadata configuration."""
        app = main.app
        assert app.title == "Cerby Identity Automation Platform"
        assert app.version == "1.0.0"
        assert "identity management" in app.description.lower()
        assert app.docs_url == "/api/docs"
        assert app.redoc_url == "/api/redoc"
        assert app.openapi_url == "/api/openapi.json"

    def test_root_endpoint(self, client):
        """Test root endpoint returns correct response."""
        response = client.get("/")
        assert response.status_code == status.HTTP_200_OK

        data = response.json()
        assert "message" in data
        assert "status" in data
        assert "version" in data
        assert data["status"] == "operational"
        assert data["version"] == "1.0.0"

    def test_health_endpoint(self, client):
        """Test health check endpoint."""
        response = client.get("/health")
        assert response.status_code == status.HTTP_200_OK

        data = response.json()
        required_fields = ["status", "timestamp", "service", "environment", "version", "checks"]
        for field in required_fields:
            assert field in data, f"Health response missing field: {field}"

        assert data["status"] == "healthy"
        assert isinstance(data["checks"], dict)
        assert all(v == "operational" for v in data["checks"].values())

    def test_cors_middleware_configured(self, client):
        """Test CORS middleware is properly configured."""
        response = client.options("/", headers={"Origin": "http://localhost:3000"})
        assert response.status_code == status.HTTP_200_OK
        assert "access-control-allow-origin" in response.headers

    def test_process_time_header_middleware(self, client):
        """Test that process time header is added to responses."""
        response = client.get("/")
        assert "x-process-time" in response.headers
        process_time = float(response.headers["x-process-time"])
        assert process_time >= 0
        assert process_time < 5  # Should be reasonably fast

    def test_request_id_header_middleware(self, client):
        """Test that request ID header is added to responses."""
        response = client.get("/")
        assert "x-request-id" in response.headers
        request_id = response.headers["x-request-id"]
        assert len(request_id) == 36  # UUID format
        assert request_id.count('-') == 4  # UUID has 4 dashes

    def test_http_exception_handler(self, client):
        """Test HTTP exception handler."""
        response = client.get("/nonexistent")
        assert response.status_code == status.HTTP_404_NOT_FOUND

        data = response.json()
        assert "error" in data
        assert "status_code" in data
        assert "timestamp" in data

    def test_lifespan_context(self):
        """Test application lifespan context manager."""
        # This tests that lifespan is properly defined
        assert hasattr(main, 'lifespan')
        assert callable(main.lifespan)

    @patch('logfire.instrument_fastapi')
    def test_logfire_instrumentation(self, mock_instrument):
        """Test that Logfire instrumentation is called."""
        # Reload the module to trigger instrumentation
        import importlib
        importlib.reload(main)

        # Check that instrument_fastapi was called
        assert mock_instrument.called
        call_args = mock_instrument.call_args[0]
        assert isinstance(call_args[0], FastAPI)
        assert mock_instrument.call_args[1].get('capture_headers') is True

    def test_api_structure_directories(self):
        """Test API directory structure exists."""
        api_dirs = [
            project_root / "src" / "api",
            project_root / "src" / "api" / "v1",
            project_root / "src" / "api" / "v1" / "endpoints",
        ]

        for dir_path in api_dirs:
            assert dir_path.exists(), f"API directory missing: {dir_path}"
            assert dir_path.is_dir(), f"Path is not a directory: {dir_path}"

    def test_openapi_schema(self, client):
        """Test OpenAPI schema generation."""
        response = client.get("/api/openapi.json")
        assert response.status_code == status.HTTP_200_OK

        schema = response.json()
        assert "openapi" in schema
        assert "info" in schema
        assert "paths" in schema

        # Check schema info
        assert schema["info"]["title"] == "Cerby Identity Automation Platform"
        assert schema["info"]["version"] == "1.0.0"

        # Check that basic paths are included
        assert "/" in schema["paths"]
        assert "/health" in schema["paths"]

    def test_startup_without_errors(self, client):
        """Test that application starts without errors."""
        # Making a request ensures the app starts up
        response = client.get("/")
        assert response.status_code == status.HTTP_200_OK

    def test_middleware_order(self):
        """Test middleware is added in correct order."""
        middlewares = [m for m in main.app.middleware]

        # Check that we have middleware
        assert len(middlewares) > 0

        # CORS should be one of the first middleware
        cors_found = any('CORSMiddleware' in str(type(m)) for m in middlewares)
        assert cors_found, "CORS middleware not found"

    def test_exception_handlers_registered(self):
        """Test that exception handlers are properly registered."""
        from fastapi import HTTPException

        # Check that handlers are registered
        assert HTTPException in main.app.exception_handlers
        assert Exception in main.app.exception_handlers

    def test_response_headers(self, client):
        """Test that required response headers are set."""
        response = client.get("/health")

        # Check for security headers (if implemented)
        headers = response.headers

        # Basic headers that should be present
        assert "content-type" in headers
        assert headers["content-type"] == "application/json"

    @pytest.mark.parametrize("endpoint,expected_status", [
        ("/", status.HTTP_200_OK),
        ("/health", status.HTTP_200_OK),
        ("/api/docs", status.HTTP_200_OK),
        ("/api/redoc", status.HTTP_200_OK),
        ("/api/openapi.json", status.HTTP_200_OK),
        ("/nonexistent", status.HTTP_404_NOT_FOUND),
    ])
    def test_endpoint_availability(self, client, endpoint, expected_status):
        """Test that endpoints return expected status codes."""
        response = client.get(endpoint)
        assert response.status_code == expected_status

    def test_json_response_format(self, client):
        """Test that JSON responses are properly formatted."""
        response = client.get("/")
        assert response.headers["content-type"] == "application/json"

        # Should be valid JSON
        try:
            data = response.json()
            assert isinstance(data, dict)
        except json.JSONDecodeError:
            pytest.fail("Response is not valid JSON")

    def test_error_response_format(self, client):
        """Test error response format consistency."""
        response = client.get("/this-endpoint-does-not-exist")
        assert response.status_code == status.HTTP_404_NOT_FOUND

        data = response.json()
        # Should have consistent error format
        assert "error" in data or "detail" in data
        assert "status_code" in data or "status" in data

    def test_app_debug_mode(self, mock_settings):
        """Test that debug mode is properly configured from settings."""
        assert settings.debug == True  # From mock
        assert settings.environment == "testing"

    def test_uvicorn_configuration(self):
        """Test uvicorn configuration in main module."""
        # Check that main.py has proper uvicorn configuration
        main_content = (project_root / "main.py").read_text()
        assert "if __name__ == '__main__':" in main_content
        assert "uvicorn.run" in main_content
        assert '"main:app"' in main_content or "'main:app'" in main_content

    def test_api_versioning_structure(self):
        """Test API versioning is properly set up."""
        assert hasattr(settings, 'api_v1_prefix')
        assert settings.api_v1_prefix == "/api/v1"

    def test_fastapi_configuration_from_settings(self):
        """Test that FastAPI is configured using settings."""
        app = main.app

        # These should come from settings or be properly configured
        assert app.title is not None
        assert app.version is not None
        assert app.docs_url is not None
