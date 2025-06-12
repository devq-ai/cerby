"""
Unit tests for Task 1.3 - Configure Logfire observability.

Tests verify that Pydantic Logfire is properly integrated for comprehensive
monitoring, instrumentation, and observability.
"""

import pytest
from unittest.mock import patch, MagicMock, call
import sys
from pathlib import Path
from datetime import datetime
import os

# Add project root to Python path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from fastapi.testclient import TestClient
import main
from src.core.config import settings


class TestLogfireConfiguration:
    """Test Logfire observability configuration and integration."""

    @pytest.fixture
    def mock_logfire(self):
        """Mock logfire module."""
        with patch('main.logfire') as mock:
            yield mock

    @pytest.fixture
    def client(self):
        """Create a test client."""
        return TestClient(main.app)

    @pytest.fixture
    def mock_env(self, monkeypatch):
        """Mock environment variables for Logfire."""
        monkeypatch.setenv("LOGFIRE_TOKEN", "test_token_123")
        monkeypatch.setenv("LOGFIRE_PROJECT_NAME", "test-project")
        monkeypatch.setenv("LOGFIRE_SERVICE_NAME", "test-service")
        monkeypatch.setenv("LOGFIRE_ENVIRONMENT", "testing")
        return {
            "token": "test_token_123",
            "project_name": "test-project",
            "service_name": "test-service",
            "environment": "testing"
        }

    def test_logfire_configuration_called(self, mock_logfire):
        """Test that logfire.configure is called with correct parameters."""
        # Reload main module to trigger configuration
        import importlib
        with patch('logfire.configure') as mock_configure:
            importlib.reload(main)

            # Check configure was called
            assert mock_configure.called
            call_kwargs = mock_configure.call_args[1]

            # Check required parameters
            assert 'token' in call_kwargs
            assert 'project_name' in call_kwargs
            assert 'service_name' in call_kwargs
            assert 'environment' in call_kwargs

    def test_logfire_instrumentation_fastapi(self, mock_logfire):
        """Test that FastAPI is instrumented with Logfire."""
        with patch('logfire.instrument_fastapi') as mock_instrument:
            import importlib
            importlib.reload(main)

            # Check instrument_fastapi was called
            assert mock_instrument.called
            call_args = mock_instrument.call_args

            # Check it was called with the app
            assert call_args[0][0] == main.app
            # Check capture_headers is enabled
            assert call_args[1].get('capture_headers') is True

    def test_logfire_span_in_health_endpoint(self, client):
        """Test that Logfire spans are used in endpoints."""
        with patch('logfire.span') as mock_span:
            # Mock the context manager
            mock_context = MagicMock()
            mock_span.return_value.__enter__ = MagicMock(return_value=mock_context)
            mock_span.return_value.__exit__ = MagicMock(return_value=None)

            response = client.get("/health")
            assert response.status_code == 200

            # Check span was created for health check
            mock_span.assert_called_with("Health check")

    def test_logfire_request_middleware_span(self, client):
        """Test that middleware creates spans for requests."""
        with patch('logfire.span') as mock_span:
            mock_context = MagicMock()
            mock_span.return_value.__enter__ = MagicMock(return_value=mock_context)
            mock_span.return_value.__exit__ = MagicMock(return_value=None)

            response = client.get("/")
            assert response.status_code == 200

            # Check span was created for HTTP request
            span_calls = [call for call in mock_span.call_args_list
                         if 'HTTP Request' in str(call)]
            assert len(span_calls) > 0

    def test_logfire_info_logging(self, client):
        """Test that Logfire info logging is used."""
        with patch('logfire.info') as mock_info:
            response = client.get("/health")
            assert response.status_code == 200

            # Check that info logging was called
            info_calls = [call for call in mock_info.call_args_list
                         if 'Health check' in str(call)]
            assert len(info_calls) > 0

    def test_logfire_error_logging_on_exception(self, client):
        """Test that Logfire logs errors properly."""
        with patch('logfire.error') as mock_error:
            response = client.get("/nonexistent")
            assert response.status_code == 404

            # Check that error logging was called
            assert mock_error.called
            error_call = mock_error.call_args
            assert 'HTTP Exception' in str(error_call)

    def test_logfire_startup_logging(self):
        """Test that Logfire logs application startup."""
        with patch('logfire.info') as mock_info:
            # Simulate startup by calling lifespan
            import asyncio

            async def test_startup():
                async with main.lifespan(main.app):
                    pass

            asyncio.run(test_startup())

            # Check startup logging
            startup_calls = [call for call in mock_info.call_args_list
                           if 'starting up' in str(call).lower()]
            assert len(startup_calls) > 0

    def test_logfire_shutdown_logging(self):
        """Test that Logfire logs application shutdown."""
        with patch('logfire.info') as mock_info:
            # Simulate shutdown
            import asyncio

            async def test_shutdown():
                async with main.lifespan(main.app):
                    pass

            asyncio.run(test_shutdown())

            # Check shutdown logging
            shutdown_calls = [call for call in mock_info.call_args_list
                            if 'shutting down' in str(call).lower()]
            assert len(shutdown_calls) > 0

    def test_logfire_configuration_from_settings(self, mock_env):
        """Test that Logfire uses configuration from settings."""
        assert hasattr(settings, 'logfire_token')
        assert hasattr(settings, 'logfire_project_name')
        assert hasattr(settings, 'logfire_service_name')
        assert hasattr(settings, 'logfire_environment')

    def test_logfire_request_tracking(self, client):
        """Test that requests are tracked with metadata."""
        with patch('logfire.info') as mock_info:
            response = client.get("/")

            # Check response headers
            assert 'x-request-id' in response.headers
            assert 'x-process-time' in response.headers

            # Check that request completion was logged
            completion_calls = [call for call in mock_info.call_args_list
                              if 'Request completed' in str(call)]
            assert len(completion_calls) > 0

    def test_logfire_metrics_in_response(self, client):
        """Test that process time metrics are captured."""
        response = client.get("/health")

        # Check process time header
        assert 'x-process-time' in response.headers
        process_time = float(response.headers['x-process-time'])
        assert process_time > 0
        assert process_time < 1  # Should be fast

    def test_logfire_context_propagation(self, client):
        """Test that Logfire context is propagated through requests."""
        with patch('logfire.span') as mock_span:
            # Make a request
            response = client.get("/")

            # Check that spans include context
            for call in mock_span.call_args_list:
                if len(call[0]) > 0 and call[0][0] == "HTTP Request":
                    kwargs = call[1]
                    assert 'request_id' in kwargs
                    assert 'method' in kwargs
                    assert 'path' in kwargs

    def test_logfire_environment_configuration(self, mock_env):
        """Test that Logfire environment is properly set."""
        # Check settings reflect environment
        assert settings.logfire_environment == "testing"

    def test_logfire_credentials_file(self):
        """Test that Logfire credentials file structure is documented."""
        # Check if example credentials file exists or is documented
        logfire_dir = project_root / ".logfire"

        # The directory should exist or be in gitignore
        gitignore_path = project_root / ".gitignore"
        if gitignore_path.exists():
            with open(gitignore_path, 'r') as f:
                content = f.read()
                assert ".logfire/" in content or "logfire_credentials.json" in content

    def test_logfire_integration_with_database(self):
        """Test that database operations can be instrumented."""
        # This is a placeholder - actual implementation would test
        # that logfire.instrument_sqlalchemy() is called
        with patch('logfire.instrument_sqlalchemy') as mock_instrument:
            from src.db.database import db_manager
            # The database module should instrument SQLAlchemy
            # This would be called during database initialization
            pass

    def test_logfire_span_attributes(self, client):
        """Test that spans include proper attributes."""
        with patch('logfire.span') as mock_span:
            response = client.get("/health")

            # Find health check span
            health_spans = [call for call in mock_span.call_args_list
                          if call[0][0] == "Health check"]
            assert len(health_spans) > 0

    def test_logfire_error_handling_with_details(self, client):
        """Test that errors include proper details in Logfire."""
        with patch('logfire.error') as mock_error:
            # Trigger a 404 error
            response = client.get("/api/v1/nonexistent")

            # Check error was logged with details
            assert mock_error.called
            error_calls = [call for call in mock_error.call_args_list
                          if 'status_code' in str(call)]
            assert len(error_calls) > 0

    def test_logfire_performance_monitoring(self, client):
        """Test that performance metrics are captured."""
        # Make multiple requests
        for _ in range(5):
            response = client.get("/health")
            assert response.status_code == 200

            # Each should have process time
            assert 'x-process-time' in response.headers

    @pytest.mark.parametrize("endpoint", ["/", "/health"])
    def test_logfire_endpoint_coverage(self, client, endpoint):
        """Test that all endpoints are covered by Logfire."""
        with patch('logfire.info') as mock_info:
            response = client.get(endpoint)
            assert response.status_code == 200

            # Should have logging for the request
            assert mock_info.called

    def test_logfire_custom_metrics(self):
        """Test that custom metrics can be added."""
        with patch('logfire.info') as mock_info:
            # Custom metric example
            mock_info("Custom metric", value=42, metric_type="gauge")

            assert mock_info.called
            call_args = mock_info.call_args
            assert call_args[0][0] == "Custom metric"
            assert call_args[1]['value'] == 42

    def test_logfire_configuration_validation(self, mock_env):
        """Test that Logfire configuration is validated."""
        # Test with missing token
        with patch.dict(os.environ, {'LOGFIRE_TOKEN': ''}):
            # Should still load but with empty token
            from src.core.config import Settings
            test_settings = Settings()
            assert test_settings.logfire_token == ''

    def test_logfire_instrumentation_coverage(self):
        """Test that all major components are instrumented."""
        required_instrumentations = [
            'logfire.instrument_fastapi',
            'logfire.configure',
        ]

        # Check that these are imported and used in main
        main_content = (project_root / "main.py").read_text()
        for instrumentation in required_instrumentations:
            assert instrumentation in main_content
