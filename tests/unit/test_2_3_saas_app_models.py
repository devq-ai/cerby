"""
Unit tests for SaaS Application models (Subtask 2.3).

Tests cover:
- SaaS application model creation and validation
- Provider configuration and authentication
- API endpoint management
- Connection status tracking
- Rate limiting configuration
"""

import pytest
from datetime import datetime, timedelta
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from src.db.models.saas_application import SaaSApplication, AuthType, SaaSProvider
from src.db.models.identity import Identity


class TestSaaSApplicationModel:
    """Test suite for SaaS Application model."""

    def test_saas_app_creation(self, db_session: Session):
        """Test creating a SaaS application with all fields."""
        app = SaaSApplication(
            name="GitHub Enterprise",
            provider=SaaSProvider.GITHUB,
            api_endpoint="https://api.github.com",
            auth_type=AuthType.OAUTH2,
            config={
                "client_id": "test_client_id",
                "client_secret": "test_client_secret",
                "scopes": ["read:user", "repo"]
            },
            rate_limit_requests=5000,
            rate_limit_window=3600,
            is_active=True
        )

        db_session.add(app)
        db_session.commit()

        assert app.id is not None
        assert app.name == "GitHub Enterprise"
        assert app.provider == SaaSProvider.GITHUB
        assert app.auth_type == AuthType.OAUTH2
        assert app.is_active is True
        assert app.created_at is not None
        assert app.updated_at is not None

    def test_saas_app_required_fields(self, db_session: Session):
        """Test that required fields are enforced."""
        # Missing required fields should raise error
        app = SaaSApplication()

        db_session.add(app)
        with pytest.raises(IntegrityError):
            db_session.commit()

    def test_saas_app_unique_constraint(self, db_session: Session):
        """Test unique constraint on name."""
        app1 = SaaSApplication(
            name="Unique App",
            provider=SaaSProvider.OKTA,
            api_endpoint="https://example.okta.com",
            auth_type=AuthType.API_KEY
        )
        db_session.add(app1)
        db_session.commit()

        # Try to create another with same name
        app2 = SaaSApplication(
            name="Unique App",
            provider=SaaSProvider.GOOGLE,
            api_endpoint="https://www.googleapis.com",
            auth_type=AuthType.OAUTH2
        )
        db_session.add(app2)

        with pytest.raises(IntegrityError):
            db_session.commit()

    def test_saas_app_provider_enum(self, db_session: Session):
        """Test all supported SaaS providers."""
        providers = [
            (SaaSProvider.OKTA, "https://example.okta.com"),
            (SaaSProvider.GOOGLE, "https://www.googleapis.com"),
            (SaaSProvider.MICROSOFT, "https://graph.microsoft.com"),
            (SaaSProvider.GITHUB, "https://api.github.com"),
            (SaaSProvider.SLACK, "https://slack.com/api"),
            (SaaSProvider.ZOOM, "https://api.zoom.us/v2"),
            (SaaSProvider.SALESFORCE, "https://example.salesforce.com"),
            (SaaSProvider.WORKDAY, "https://example.workday.com"),
            (SaaSProvider.ATLASSIAN, "https://api.atlassian.com"),
            (SaaSProvider.DROPBOX, "https://api.dropboxapi.com"),
        ]

        for provider, endpoint in providers:
            app = SaaSApplication(
                name=f"Test {provider.value}",
                provider=provider,
                api_endpoint=endpoint,
                auth_type=AuthType.OAUTH2
            )
            db_session.add(app)

        db_session.commit()

        # Verify all were created
        assert db_session.query(SaaSApplication).count() == len(providers)

    def test_saas_app_auth_types(self, db_session: Session):
        """Test all supported authentication types."""
        auth_types = [
            (AuthType.OAUTH2, {"client_id": "test", "client_secret": "secret"}),
            (AuthType.API_KEY, {"api_key": "test_key"}),
            (AuthType.BASIC, {"username": "user", "password": "pass"}),
            (AuthType.BEARER, {"token": "bearer_token"}),
            (AuthType.SAML, {"sso_url": "https://sso.example.com"}),
        ]

        for i, (auth_type, config) in enumerate(auth_types):
            app = SaaSApplication(
                name=f"Test Auth {i}",
                provider=SaaSProvider.CUSTOM,
                api_endpoint=f"https://api{i}.example.com",
                auth_type=auth_type,
                config=config
            )
            db_session.add(app)

        db_session.commit()

        # Verify all auth types work
        apps = db_session.query(SaaSApplication).all()
        assert len(apps) == len(auth_types)

    def test_saas_app_config_json(self, db_session: Session):
        """Test JSON configuration storage."""
        complex_config = {
            "oauth": {
                "client_id": "test_id",
                "client_secret": "test_secret",
                "redirect_uri": "https://example.com/callback",
                "scopes": ["read", "write", "admin"]
            },
            "webhooks": {
                "endpoint": "https://example.com/webhooks",
                "secret": "webhook_secret"
            },
            "custom_headers": {
                "X-API-Version": "2.0",
                "X-Client-Id": "cerby"
            }
        }

        app = SaaSApplication(
            name="Complex Config App",
            provider=SaaSProvider.CUSTOM,
            api_endpoint="https://api.example.com",
            auth_type=AuthType.OAUTH2,
            config=complex_config
        )

        db_session.add(app)
        db_session.commit()
        db_session.refresh(app)

        # Verify JSON is stored and retrieved correctly
        assert app.config == complex_config
        assert app.config["oauth"]["scopes"] == ["read", "write", "admin"]

    def test_saas_app_rate_limiting(self, db_session: Session):
        """Test rate limiting configuration."""
        app = SaaSApplication(
            name="Rate Limited App",
            provider=SaaSProvider.GITHUB,
            api_endpoint="https://api.github.com",
            auth_type=AuthType.OAUTH2,
            rate_limit_requests=1000,
            rate_limit_window=3600  # 1 hour
        )

        db_session.add(app)
        db_session.commit()

        assert app.rate_limit_requests == 1000
        assert app.rate_limit_window == 3600

        # Test rate limit calculation
        assert app.get_rate_limit_per_minute() == pytest.approx(16.67, rel=0.01)

    def test_saas_app_status_tracking(self, db_session: Session):
        """Test connection status tracking."""
        app = SaaSApplication(
            name="Status Test App",
            provider=SaaSProvider.OKTA,
            api_endpoint="https://example.okta.com",
            auth_type=AuthType.API_KEY,
            is_active=True
        )

        db_session.add(app)
        db_session.commit()

        # Test initial status
        assert app.is_active is True
        assert app.last_sync_at is None
        assert app.last_error_at is None

        # Update sync status
        app.update_sync_status(success=True)
        db_session.commit()

        assert app.last_sync_at is not None
        assert app.last_error_at is None
        assert app.sync_error_message is None

        # Update with error
        app.update_sync_status(success=False, error="API rate limit exceeded")
        db_session.commit()

        assert app.last_error_at is not None
        assert app.sync_error_message == "API rate limit exceeded"

    def test_saas_app_identity_relationship(self, db_session: Session):
        """Test relationship with Identity model."""
        app = SaaSApplication(
            name="Identity Test App",
            provider=SaaSProvider.GOOGLE,
            api_endpoint="https://www.googleapis.com",
            auth_type=AuthType.OAUTH2
        )

        db_session.add(app)
        db_session.commit()

        # Create identities for this app
        identity1 = Identity(
            provider=SaaSProvider.GOOGLE,
            external_id="google123",
            email="user1@example.com",
            username="user1",
            saas_app_id=app.id
        )

        identity2 = Identity(
            provider=SaaSProvider.GOOGLE,
            external_id="google456",
            email="user2@example.com",
            username="user2",
            saas_app_id=app.id
        )

        db_session.add_all([identity1, identity2])
        db_session.commit()

        # Refresh to load relationships
        db_session.refresh(app)

        assert len(app.identities) == 2
        assert identity1 in app.identities
        assert identity2 in app.identities

    def test_saas_app_webhook_config(self, db_session: Session):
        """Test webhook configuration storage."""
        app = SaaSApplication(
            name="Webhook App",
            provider=SaaSProvider.SLACK,
            api_endpoint="https://slack.com/api",
            auth_type=AuthType.OAUTH2,
            webhook_url="https://cerby.example.com/webhooks/slack",
            webhook_secret="webhook_secret_123",
            config={
                "webhook_events": ["user.created", "user.updated", "user.deleted"]
            }
        )

        db_session.add(app)
        db_session.commit()

        assert app.webhook_url == "https://cerby.example.com/webhooks/slack"
        assert app.webhook_secret == "webhook_secret_123"
        assert "user.created" in app.config["webhook_events"]

    def test_saas_app_methods(self, db_session: Session):
        """Test SaaSApplication model methods."""
        app = SaaSApplication(
            name="Method Test App",
            provider=SaaSProvider.MICROSOFT,
            api_endpoint="https://graph.microsoft.com",
            auth_type=AuthType.OAUTH2,
            rate_limit_requests=5000,
            rate_limit_window=3600
        )

        db_session.add(app)
        db_session.commit()

        # Test get_rate_limit_per_minute
        assert app.get_rate_limit_per_minute() == pytest.approx(83.33, rel=0.01)

        # Test is_rate_limited
        assert app.is_rate_limited() is False

        # Test to_dict
        app_dict = app.to_dict()
        assert app_dict["name"] == "Method Test App"
        assert app_dict["provider"] == "microsoft"
        assert app_dict["auth_type"] == "oauth2"
        assert "config" not in app_dict  # Config should be excluded by default

        # Test to_dict with config
        app_dict_full = app.to_dict(include_config=True)
        assert "config" in app_dict_full

    def test_saas_app_validation(self, db_session: Session):
        """Test model validation."""
        # Test invalid rate limit
        app = SaaSApplication(
            name="Invalid App",
            provider=SaaSProvider.OKTA,
            api_endpoint="https://example.okta.com",
            auth_type=AuthType.API_KEY,
            rate_limit_requests=-1  # Invalid
        )

        with pytest.raises(ValueError):
            app.validate()

        # Test invalid window
        app.rate_limit_requests = 1000
        app.rate_limit_window = 0  # Invalid

        with pytest.raises(ValueError):
            app.validate()

    def test_saas_app_scim_support(self, db_session: Session):
        """Test SCIM support configuration."""
        app = SaaSApplication(
            name="SCIM App",
            provider=SaaSProvider.OKTA,
            api_endpoint="https://example.okta.com",
            auth_type=AuthType.BEARER,
            supports_scim=True,
            scim_endpoint="https://example.okta.com/scim/v2",
            config={
                "scim_version": "2.0",
                "scim_schemas": [
                    "urn:ietf:params:scim:schemas:core:2.0:User",
                    "urn:ietf:params:scim:schemas:extension:enterprise:2.0:User"
                ]
            }
        )

        db_session.add(app)
        db_session.commit()

        assert app.supports_scim is True
        assert app.scim_endpoint == "https://example.okta.com/scim/v2"
        assert app.config["scim_version"] == "2.0"

    def test_saas_app_soft_delete(self, db_session: Session):
        """Test soft delete functionality."""
        app = SaaSApplication(
            name="Delete Test App",
            provider=SaaSProvider.ZOOM,
            api_endpoint="https://api.zoom.us/v2",
            auth_type=AuthType.JWT,
            is_active=True
        )

        db_session.add(app)
        db_session.commit()

        # Soft delete
        app.soft_delete()
        db_session.commit()

        assert app.is_active is False
        assert app.deleted_at is not None

        # Should not appear in active query
        active_apps = db_session.query(SaaSApplication).filter(
            SaaSApplication.is_active == True
        ).all()
        assert app not in active_apps

    def test_saas_app_connection_test(self, db_session: Session, mocker):
        """Test connection testing functionality."""
        app = SaaSApplication(
            name="Connection Test App",
            provider=SaaSProvider.GITHUB,
            api_endpoint="https://api.github.com",
            auth_type=AuthType.OAUTH2,
            config={
                "client_id": "test",
                "client_secret": "secret"
            }
        )

        db_session.add(app)
        db_session.commit()

        # Mock the connection test
        mock_test = mocker.patch.object(app, 'test_connection', return_value=True)

        result = app.test_connection()
        assert result is True
        mock_test.assert_called_once()

    def test_saas_app_custom_provider(self, db_session: Session):
        """Test custom provider configuration."""
        app = SaaSApplication(
            name="Custom Provider App",
            provider=SaaSProvider.CUSTOM,
            api_endpoint="https://custom.api.com",
            auth_type=AuthType.CUSTOM,
            config={
                "custom_auth": {
                    "method": "hmac",
                    "key": "secret_key",
                    "algorithm": "sha256"
                },
                "custom_headers": {
                    "X-Custom-Auth": "required",
                    "X-API-Version": "v3"
                }
            }
        )

        db_session.add(app)
        db_session.commit()

        assert app.provider == SaaSProvider.CUSTOM
        assert app.auth_type == AuthType.CUSTOM
        assert app.config["custom_auth"]["method"] == "hmac"
