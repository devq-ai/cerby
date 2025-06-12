"""
Unit tests for Webhook Receivers (Subtask 3.3).

Tests cover:
- Webhook endpoint creation
- Provider-specific webhook handling
- Event parsing and validation
- Security (signature verification)
- Error handling and retries
"""

import pytest
from datetime import datetime
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session
import json
import hmac
import hashlib
import base64

from main import app
from src.ingestion.webhook import WebhookRouter, WebhookHandler, WebhookSecurity
from src.db.models.identity import Identity
from src.db.models.saas_application import SaaSApplication, SaaSProvider
from src.db.models.audit import IdentityEvent, EventType


class TestWebhookReceivers:
    """Test suite for webhook receivers."""

    @pytest.fixture
    def webhook_secret(self):
        """Webhook secret for testing."""
        return "test_webhook_secret_123"

    @pytest.fixture
    def saas_app(self, db_session: Session, webhook_secret):
        """Create a SaaS application with webhook configuration."""
        app = SaaSApplication(
            name="Webhook Test App",
            provider=SaaSProvider.OKTA,
            api_endpoint="https://test.okta.com",
            auth_type="oauth2",
            webhook_url="https://cerby.example.com/webhooks/okta",
            webhook_secret=webhook_secret
        )
        db_session.add(app)
        db_session.commit()
        return app

    def test_okta_webhook_user_created(self, client: TestClient, saas_app, webhook_secret):
        """Test Okta webhook for user creation event."""
        webhook_data = {
            "eventId": "event123",
            "eventTime": "2024-01-01T10:00:00.000Z",
            "eventType": "user.lifecycle.create",
            "displayMessage": "User created",
            "outcome": {
                "result": "SUCCESS"
            },
            "target": [
                {
                    "id": "00u1234567890",
                    "type": "User",
                    "alternateId": "testuser@example.com",
                    "displayName": "Test User"
                }
            ]
        }

        # Calculate signature
        signature = self._calculate_okta_signature(webhook_data, webhook_secret)

        response = client.post(
            "/webhooks/okta",
            json=webhook_data,
            headers={"X-Okta-Verification-Challenge": signature}
        )

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "accepted"
        assert data["event_id"] == "event123"

    def test_google_webhook_user_updated(self, client: TestClient, saas_app, webhook_secret):
        """Test Google Workspace webhook for user update event."""
        webhook_data = {
            "kind": "admin#reports#activity",
            "id": {"time": "2024-01-01T10:00:00.000Z"},
            "etag": "etag123",
            "actor": {
                "email": "admin@example.com",
                "profileId": "admin123"
            },
            "events": [
                {
                    "type": "USER_SETTINGS",
                    "name": "CHANGE_PASSWORD",
                    "parameters": [
                        {
                            "name": "USER_EMAIL",
                            "value": "user@example.com"
                        }
                    ]
                }
            ]
        }

        # Google uses different signature method
        signature = self._calculate_google_signature(webhook_data, webhook_secret)

        response = client.post(
            "/webhooks/google",
            json=webhook_data,
            headers={"X-Goog-Channel-Token": signature}
        )

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "accepted"

    def test_microsoft_webhook_user_deleted(self, client: TestClient, saas_app, webhook_secret):
        """Test Microsoft Graph webhook for user deletion event."""
        webhook_data = {
            "value": [
                {
                    "subscriptionId": "sub123",
                    "changeType": "deleted",
                    "resourceData": {
                        "@odata.type": "#Microsoft.Graph.User",
                        "@odata.id": "Users/user123",
                        "id": "user123"
                    },
                    "subscriptionExpirationDateTime": "2024-12-31T23:59:59.000Z"
                }
            ]
        }

        # Microsoft uses bearer token validation
        response = client.post(
            "/webhooks/microsoft",
            json=webhook_data,
            headers={"Authorization": f"Bearer {webhook_secret}"}
        )

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "accepted"

    def test_slack_webhook_user_joined(self, client: TestClient, saas_app, webhook_secret):
        """Test Slack webhook for user joined event."""
        webhook_data = {
            "token": webhook_secret,
            "team_id": "T123456",
            "api_app_id": "A123456",
            "event": {
                "type": "team_join",
                "user": {
                    "id": "U123456789",
                    "team_id": "T123456",
                    "name": "newuser",
                    "real_name": "New User",
                    "email": "newuser@example.com"
                }
            },
            "type": "event_callback",
            "event_id": "Ev123456",
            "event_time": 1234567890
        }

        response = client.post("/webhooks/slack", json=webhook_data)

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "accepted"

    def test_webhook_signature_verification_failure(self, client: TestClient, saas_app):
        """Test webhook with invalid signature."""
        webhook_data = {
            "eventId": "event456",
            "eventType": "user.lifecycle.create",
            "target": [{"id": "00u987654321"}]
        }

        # Use wrong signature
        wrong_signature = "invalid_signature"

        response = client.post(
            "/webhooks/okta",
            json=webhook_data,
            headers={"X-Okta-Verification-Challenge": wrong_signature}
        )

        assert response.status_code == 401
        data = response.json()
        assert data["error"] == "Invalid signature"

    def test_webhook_duplicate_event_handling(self, client: TestClient, db_session: Session, saas_app, webhook_secret):
        """Test handling of duplicate webhook events."""
        # Create an existing event
        existing_event = IdentityEvent(
            event_id="duplicate123",
            event_type=EventType.USER_CREATED,
            provider=SaaSProvider.OKTA,
            external_id="okta123",
            timestamp=datetime.utcnow(),
            data={"test": True}
        )
        db_session.add(existing_event)
        db_session.commit()

        # Send webhook with same event ID
        webhook_data = {
            "eventId": "duplicate123",
            "eventType": "user.lifecycle.create",
            "target": [{"id": "okta123"}]
        }

        signature = self._calculate_okta_signature(webhook_data, webhook_secret)

        response = client.post(
            "/webhooks/okta",
            json=webhook_data,
            headers={"X-Okta-Verification-Challenge": signature}
        )

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "duplicate"

    def test_webhook_retry_mechanism(self, client: TestClient, mocker):
        """Test webhook retry handling."""
        # Mock internal processing to fail first time
        mock_process = mocker.patch(
            'src.ingestion.webhook.WebhookHandler.process_event',
            side_effect=[Exception("Temporary failure"), {"status": "success"}]
        )

        webhook_data = {
            "eventId": "retry123",
            "eventType": "user.lifecycle.create"
        }

        # First attempt should fail but return 200 (queued for retry)
        response = client.post("/webhooks/okta", json=webhook_data)
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "queued_for_retry"

    def test_webhook_event_parsing(self, client: TestClient, db_session: Session):
        """Test webhook event parsing and storage."""
        webhook_data = {
            "eventId": "parse123",
            "eventType": "user.lifecycle.activate",
            "eventTime": "2024-01-01T10:00:00.000Z",
            "target": [
                {
                    "id": "00uparse123",
                    "type": "User",
                    "alternateId": "parseuser@example.com",
                    "displayName": "Parse User"
                }
            ],
            "actor": {
                "id": "admin123",
                "type": "User",
                "alternateId": "admin@example.com"
            }
        }

        response = client.post("/webhooks/okta", json=webhook_data)

        assert response.status_code == 200

        # Verify event was stored
        event = db_session.query(IdentityEvent).filter(
            IdentityEvent.event_id == "parse123"
        ).first()

        assert event is not None
        assert event.event_type == EventType.USER_ACTIVATED
        assert event.external_id == "00uparse123"
        assert event.data["email"] == "parseuser@example.com"

    def test_webhook_bulk_events(self, client: TestClient, saas_app, webhook_secret):
        """Test webhook with multiple events in single request."""
        webhook_data = {
            "events": [
                {
                    "eventId": "bulk1",
                    "eventType": "user.lifecycle.create",
                    "target": [{"id": "user1"}]
                },
                {
                    "eventId": "bulk2",
                    "eventType": "user.lifecycle.update",
                    "target": [{"id": "user2"}]
                },
                {
                    "eventId": "bulk3",
                    "eventType": "user.lifecycle.delete",
                    "target": [{"id": "user3"}]
                }
            ]
        }

        signature = self._calculate_okta_signature(webhook_data, webhook_secret)

        response = client.post(
            "/webhooks/okta/bulk",
            json=webhook_data,
            headers={"X-Okta-Verification-Challenge": signature}
        )

        assert response.status_code == 200
        data = response.json()
        assert data["processed"] == 3
        assert len(data["results"]) == 3

    def test_webhook_rate_limiting(self, client: TestClient):
        """Test webhook rate limiting."""
        # Send many requests quickly
        responses = []
        for i in range(20):
            response = client.post(
                "/webhooks/okta",
                json={"eventId": f"rate{i}", "eventType": "test"}
            )
            responses.append(response)

        # Some requests should be rate limited
        rate_limited = [r for r in responses if r.status_code == 429]
        assert len(rate_limited) > 0

        # Check rate limit headers
        limited_response = rate_limited[0]
        assert "X-RateLimit-Limit" in limited_response.headers
        assert "X-RateLimit-Remaining" in limited_response.headers

    def test_webhook_custom_provider(self, client: TestClient, db_session: Session):
        """Test webhook for custom provider."""
        # Create custom provider app
        custom_app = SaaSApplication(
            name="Custom App",
            provider=SaaSProvider.CUSTOM,
            api_endpoint="https://custom.example.com",
            auth_type="custom",
            webhook_url="https://cerby.example.com/webhooks/custom/customapp",
            webhook_secret="custom_secret"
        )
        db_session.add(custom_app)
        db_session.commit()

        webhook_data = {
            "event": "user.created",
            "data": {
                "user_id": "custom123",
                "email": "custom@example.com",
                "attributes": {
                    "department": "Engineering"
                }
            },
            "timestamp": "2024-01-01T10:00:00Z"
        }

        # Custom provider might use HMAC-SHA256
        signature = hmac.new(
            "custom_secret".encode(),
            json.dumps(webhook_data).encode(),
            hashlib.sha256
        ).hexdigest()

        response = client.post(
            "/webhooks/custom/customapp",
            json=webhook_data,
            headers={"X-Custom-Signature": signature}
        )

        assert response.status_code == 200

    def test_webhook_validation_endpoint(self, client: TestClient):
        """Test webhook validation endpoint for provider setup."""
        # Some providers require validation during webhook setup
        response = client.get(
            "/webhooks/okta/validate",
            params={"challenge": "validation_challenge_123"}
        )

        assert response.status_code == 200
        assert response.text == "validation_challenge_123"

    def _calculate_okta_signature(self, data: dict, secret: str) -> str:
        """Calculate Okta webhook signature."""
        # Simplified for testing - real implementation would be more complex
        return base64.b64encode(
            hmac.new(
                secret.encode(),
                json.dumps(data).encode(),
                hashlib.sha256
            ).digest()
        ).decode()

    def _calculate_google_signature(self, data: dict, secret: str) -> str:
        """Calculate Google webhook signature."""
        # Google uses different signature method
        return hashlib.sha1(
            f"{secret}{json.dumps(data)}".encode()
        ).hexdigest()
