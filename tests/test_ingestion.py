"""
Test suite for the ingestion pipeline in Cerby Identity Automation Platform.

This module tests synthetic data generation, SCIM endpoints, webhook processing,
batch imports, and streaming functionality.
"""

import pytest
import asyncio
import json
from datetime import datetime, timedelta
from pathlib import Path
import tempfile
from unittest.mock import Mock, patch, AsyncMock
from fastapi import Request
from fastapi.testclient import TestClient
import pandas as pd

from src.ingestion.synthetic import SyntheticDataGenerator
from src.ingestion.scim import SCIMHandler, SCIMError
from src.ingestion.webhook import WebhookHandler, WebhookAuthType
from src.ingestion.batch import BatchImporter, BatchImportConfig
from src.ingestion.streaming import StreamProcessor, StreamStatus
from src.ingestion.base import IngestionStatus, IngestionError
from src.db.models.identity import Identity, IdentityProvider
from src.db.models.saas_app import SaaSApplication, SaaSAppType, SyncMethod
from src.db.models.audit import IdentityEvent, EventType


class TestSyntheticDataGenerator:
    """Test synthetic data generation."""

    @pytest.fixture
    async def generator(self, async_db_session):
        """Create synthetic data generator."""
        saas_app = SaaSApplication(
            name="Test Okta",
            provider=IdentityProvider.OKTA.value,
            app_type=SaaSAppType.IDENTITY_PROVIDER.value,
            sync_method=SyncMethod.REST_API.value
        )
        async_db_session.add(saas_app)
        await async_db_session.commit()

        return SyntheticDataGenerator(async_db_session, saas_app)

    @pytest.mark.asyncio
    async def test_generate_identities(self, generator):
        """Test generating synthetic identities."""
        # Generate identities
        result = await generator.ingest(count=10)

        assert result.status == IngestionStatus.COMPLETED
        assert result.total_records == 10
        assert result.processed_records == 10
        assert result.created_records == 10
        assert result.failed_records == 0

    @pytest.mark.asyncio
    async def test_generate_with_department_filter(self, generator):
        """Test generating identities with department filter."""
        # Generate only Engineering identities
        result = await generator.ingest(count=5, department_filter="Engineering")

        assert result.status == IngestionStatus.COMPLETED
        assert result.processed_records == 5

        # Verify all generated identities are in Engineering
        from sqlalchemy import select
        stmt = select(Identity).where(Identity.provider == IdentityProvider.OKTA.value)
        result = await generator.db_session.execute(stmt)
        identities = result.scalars().all()

        for identity in identities:
            assert identity.department == "Engineering"

    @pytest.mark.asyncio
    async def test_provider_specific_attributes(self, generator):
        """Test that provider-specific attributes are generated."""
        # Generate one identity
        await generator.ingest(count=1)

        # Get the generated identity
        from sqlalchemy import select
        stmt = select(Identity).where(Identity.provider == IdentityProvider.OKTA.value)
        result = await generator.db_session.execute(stmt)
        identity = result.scalar_one()

        # Check Okta-specific attributes
        assert "provider_attributes" in identity.__dict__
        okta_attrs = identity.provider_attributes
        assert "login" in okta_attrs
        assert "status" in okta_attrs
        assert "profile" in okta_attrs
        assert okta_attrs["profile"]["firstName"] == identity.first_name
        assert okta_attrs["profile"]["lastName"] == identity.last_name


class TestSCIMHandler:
    """Test SCIM 2.0 protocol handler."""

    @pytest.fixture
    async def scim_handler(self, async_db_session):
        """Create SCIM handler."""
        saas_app = SaaSApplication(
            name="Test SCIM App",
            provider=IdentityProvider.OKTA.value,
            app_type=SaaSAppType.IDENTITY_PROVIDER.value,
            sync_method=SyncMethod.SCIM.value
        )
        async_db_session.add(saas_app)
        await async_db_session.commit()

        return SCIMHandler(async_db_session, saas_app)

    @pytest.mark.asyncio
    async def test_create_user(self, scim_handler):
        """Test creating user via SCIM."""
        user_data = {
            "schemas": ["urn:ietf:params:scim:schemas:core:2.0:User"],
            "userName": "john.doe",
            "name": {
                "givenName": "John",
                "familyName": "Doe"
            },
            "emails": [
                {
                    "value": "john.doe@example.com",
                    "type": "work",
                    "primary": True
                }
            ],
            "displayName": "John Doe",
            "active": True
        }

        # Create user
        result = await scim_handler.create_user(user_data)

        assert result["id"] == "john.doe"
        assert result["userName"] == "john.doe"
        assert result["name"]["givenName"] == "John"
        assert result["meta"]["resourceType"] == "User"

    @pytest.mark.asyncio
    async def test_get_user(self, scim_handler):
        """Test retrieving user via SCIM."""
        # First create a user
        user_data = {
            "schemas": ["urn:ietf:params:scim:schemas:core:2.0:User"],
            "userName": "jane.doe",
            "emails": [{"value": "jane.doe@example.com", "primary": True}]
        }
        await scim_handler.create_user(user_data)

        # Get the user
        result = await scim_handler.get_user("jane.doe")

        assert result["id"] == "jane.doe"
        assert result["userName"] == "jane.doe"

    @pytest.mark.asyncio
    async def test_update_user(self, scim_handler):
        """Test updating user via SCIM."""
        # Create user
        user_data = {
            "schemas": ["urn:ietf:params:scim:schemas:core:2.0:User"],
            "userName": "update.test",
            "emails": [{"value": "update.test@example.com", "primary": True}]
        }
        await scim_handler.create_user(user_data)

        # Update user
        update_data = user_data.copy()
        update_data["displayName"] = "Updated User"
        update_data["active"] = False

        result = await scim_handler.update_user("update.test", update_data)

        assert result["displayName"] == "Updated User"
        assert not result["active"]

    @pytest.mark.asyncio
    async def test_list_users_with_filter(self, scim_handler):
        """Test listing users with SCIM filter."""
        # Create multiple users
        for i in range(5):
            user_data = {
                "schemas": ["urn:ietf:params:scim:schemas:core:2.0:User"],
                "userName": f"user{i}",
                "emails": [{"value": f"user{i}@example.com", "primary": True}],
                "active": i % 2 == 0  # Even users are active
            }
            await scim_handler.create_user(user_data)

        # List with filter
        result = await scim_handler.list_users(
            filter_query='active eq true',
            start_index=1,
            count=10
        )

        assert result["schemas"] == ["urn:ietf:params:scim:api:messages:2.0:ListResponse"]
        assert result["totalResults"] == 3  # 0, 2, 4 are active
        assert len(result["Resources"]) == 3


class TestWebhookHandler:
    """Test webhook processing."""

    @pytest.fixture
    async def webhook_handler(self, async_db_session):
        """Create webhook handler."""
        saas_app = SaaSApplication(
            name="Test Webhook App",
            provider=IdentityProvider.OKTA.value,
            app_type=SaaSAppType.IDENTITY_PROVIDER.value,
            sync_method=SyncMethod.WEBHOOK.value,
            webhook_secret="test_secret",
            provider_config={
                "webhook_auth_type": WebhookAuthType.HMAC_SHA256.value,
                "signature_header": "X-Webhook-Signature"
            }
        )
        async_db_session.add(saas_app)
        await async_db_session.commit()

        return WebhookHandler(async_db_session, saas_app)

    @pytest.mark.asyncio
    async def test_process_okta_webhook(self, webhook_handler):
        """Test processing Okta webhook event."""
        # Mock request
        event_data = {
            "eventType": "user.lifecycle.create",
            "target": [
                {
                    "type": "User",
                    "id": "00u123456",
                    "alternateId": "new.user@example.com",
                    "displayName": "New User"
                }
            ]
        }

        request = Mock(spec=Request)
        request.json = AsyncMock(return_value={"data": {"events": [event_data]}})
        request.headers = {"X-Webhook-Signature": "valid_signature"}
        request.body = AsyncMock(return_value=b'{"data": {"events": []}}')

        # Mock signature verification
        with patch.object(webhook_handler, '_verify_webhook', return_value=True):
            result = await webhook_handler.ingest(request=request)

        assert result.status == IngestionStatus.COMPLETED
        assert result.processed_records == 1

    @pytest.mark.asyncio
    async def test_webhook_deduplication(self, webhook_handler):
        """Test webhook event deduplication."""
        # Create duplicate events
        event = {
            "id": "duplicate-event-123",
            "eventType": "user.lifecycle.update",
            "target": [{
                "type": "User",
                "id": "00u123456",
                "alternateId": "user@example.com"
            }]
        }

        request = Mock(spec=Request)
        request.json = AsyncMock(return_value={"data": {"events": [event, event]}})
        request.headers = {}
        request.body = AsyncMock(return_value=b'{}')

        with patch.object(webhook_handler, '_verify_webhook', return_value=True):
            result = await webhook_handler.ingest(request=request)

        # Should only process one event
        assert result.processed_records == 1
        assert result.skipped_records == 1


class TestBatchImporter:
    """Test batch import functionality."""

    @pytest.fixture
    async def batch_importer(self, async_db_session):
        """Create batch importer."""
        saas_app = SaaSApplication(
            name="Test Batch App",
            provider=IdentityProvider.OKTA.value,
            app_type=SaaSAppType.IDENTITY_PROVIDER.value,
            sync_method=SyncMethod.CSV_IMPORT.value
        )
        async_db_session.add(saas_app)
        await async_db_session.commit()

        config = BatchImportConfig(
            chunk_size=2,
            skip_invalid_rows=True
        )

        return BatchImporter(async_db_session, saas_app, config)

    @pytest.mark.asyncio
    async def test_import_csv_file(self, batch_importer):
        """Test importing identities from CSV file."""
        # Create temporary CSV file
        csv_content = """email,firstName,lastName,department
john.doe@example.com,John,Doe,Engineering
jane.smith@example.com,Jane,Smith,Sales
invalid-email,Invalid,User,HR
"""

        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            f.write(csv_content)
            csv_path = Path(f.name)

        try:
            # Import CSV
            result = await batch_importer.ingest(csv_path)

            assert result.status == IngestionStatus.COMPLETED
            assert result.total_records == 3
            assert result.processed_records == 2  # Two valid records
            assert result.skipped_records == 1  # One invalid email

        finally:
            csv_path.unlink()

    @pytest.mark.asyncio
    async def test_import_json_file(self, batch_importer):
        """Test importing identities from JSON file."""
        # Create temporary JSON file
        json_data = {
            "users": [
                {
                    "email": "user1@example.com",
                    "firstName": "User",
                    "lastName": "One",
                    "department": "IT"
                },
                {
                    "email": "user2@example.com",
                    "firstName": "User",
                    "lastName": "Two",
                    "department": "HR"
                }
            ]
        }

        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(json_data, f)
            json_path = Path(f.name)

        try:
            # Import JSON
            result = await batch_importer.ingest(json_path)

            assert result.status == IngestionStatus.COMPLETED
            assert result.total_records == 2
            assert result.processed_records == 2

        finally:
            json_path.unlink()


class TestStreamProcessor:
    """Test streaming processor."""

    @pytest.fixture
    async def stream_processor(self, async_db_session):
        """Create stream processor."""
        saas_app = SaaSApplication(
            name="Test Stream App",
            provider=IdentityProvider.OKTA.value,
            app_type=SaaSAppType.IDENTITY_PROVIDER.value,
            sync_method=SyncMethod.REST_API.value
        )
        async_db_session.add(saas_app)
        await async_db_session.commit()

        return StreamProcessor(async_db_session, saas_app, num_partitions=2)

    @pytest.mark.asyncio
    async def test_stream_processing(self, stream_processor):
        """Test basic stream processing."""
        # Start processor
        await stream_processor.ingest()
        assert stream_processor.status == StreamStatus.RUNNING

        # Produce events
        for i in range(5):
            await stream_processor.produce_event({
                "external_id": f"user{i}",
                "email": f"user{i}@example.com",
                "display_name": f"User {i}"
            })

        # Wait for processing
        await asyncio.sleep(1)

        # Stop processor
        result = await stream_processor.stop()

        assert result.status == IngestionStatus.COMPLETED
        assert stream_processor.metrics["events_received"] == 5
        assert stream_processor.metrics["events_processed"] == 5

    @pytest.mark.asyncio
    async def test_stream_partitioning(self, stream_processor):
        """Test that events are distributed across partitions."""
        await stream_processor.ingest()

        # Produce events with different partition keys
        events = [
            {"external_id": "user1", "email": "user1@example.com"},
            {"external_id": "user2", "email": "user2@example.com"},
            {"external_id": "user3", "email": "user3@example.com"},
            {"external_id": "user4", "email": "user4@example.com"},
        ]

        for event in events:
            await stream_processor.produce_event(event)

        # Check partition distribution
        partition_stats = await stream_processor.get_partition_stats()

        # Should have events distributed across partitions
        events_in_partitions = sum(
            stats["events_processed"] + len(stream_processor.event_buffers[pid])
            for pid, stats in partition_stats.items()
        )

        await stream_processor.stop()

    @pytest.mark.asyncio
    async def test_stream_error_handling(self, stream_processor):
        """Test stream error handling and DLQ."""
        await stream_processor.ingest()

        # Produce invalid event
        await stream_processor.produce_event({
            "external_id": "invalid",
            # Missing required email field
        })

        await asyncio.sleep(0.5)
        await stream_processor.stop()

        # Check DLQ
        dlq_events = await stream_processor.get_dlq_events()
        assert len(dlq_events) > 0


@pytest.mark.integration
class TestIngestionIntegration:
    """Integration tests for the complete ingestion pipeline."""

    @pytest.mark.asyncio
    async def test_synthetic_to_stream_pipeline(self, async_db_session):
        """Test generating synthetic data and processing through stream."""
        # Create SaaS app
        saas_app = SaaSApplication(
            name="Integration Test App",
            provider=IdentityProvider.OKTA.value,
            app_type=SaaSAppType.IDENTITY_PROVIDER.value,
            sync_method=SyncMethod.REST_API.value
        )
        async_db_session.add(saas_app)
        await async_db_session.commit()

        # Generate synthetic data
        generator = SyntheticDataGenerator(async_db_session, saas_app)
        gen_result = await generator.ingest(count=10)
        assert gen_result.status == IngestionStatus.COMPLETED

        # Create stream processor
        processor = StreamProcessor(async_db_session, saas_app)
        await processor.ingest()

        # Get generated identities and stream them
        from sqlalchemy import select
        stmt = select(Identity).where(Identity.provider == IdentityProvider.OKTA.value)
        result = await async_db_session.execute(stmt)
        identities = result.scalars().all()

        for identity in identities:
            await processor.produce_event(identity.to_dict())

        await asyncio.sleep(1)
        stream_result = await processor.stop()

        assert stream_result.status == IngestionStatus.COMPLETED
        assert processor.metrics["events_processed"] == len(identities)
