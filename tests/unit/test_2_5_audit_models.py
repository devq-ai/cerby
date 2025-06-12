"""
Unit tests for Audit and Event models (Subtask 2.5).

Tests cover:
- Audit log creation and tracking
- Identity event recording
- Event correlation and analysis
- Compliance tracking
- Event searching and filtering
"""

import pytest
from datetime import datetime, timedelta
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session
import json
import uuid

from src.db.models.audit import AuditLog, IdentityEvent, EventType, EventStatus
from src.db.models.user import User
from src.db.models.identity import Identity
from src.db.models.saas_application import SaaSApplication, SaaSProvider


class TestAuditLogModel:
    """Test suite for Audit Log model."""

    def test_audit_log_creation(self, db_session: Session):
        """Test creating an audit log with all fields."""
        # Create a user for the audit log
        user = User(
            email="auditor@example.com",
            full_name="Test Auditor",
            is_active=True
        )
        db_session.add(user)
        db_session.commit()

        audit_log = AuditLog(
            user_id=user.id,
            action="user.login",
            resource_type="authentication",
            resource_id=str(user.id),
            ip_address="192.168.1.100",
            user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
            success=True,
            changes={
                "before": {"last_login": None},
                "after": {"last_login": datetime.utcnow().isoformat()}
            },
            action_metadata={"method": "password", "mfa": True},
            reason="Regular user login"
        )

        db_session.add(audit_log)
        db_session.commit()

        assert audit_log.id is not None
        assert audit_log.user_id == user.id
        assert audit_log.action == "user.login"
        assert audit_log.success is True
        assert audit_log.ip_address == "192.168.1.100"
        assert audit_log.created_at is not None

    def test_audit_log_required_fields(self, db_session: Session):
        """Test that required fields are enforced."""
        # Missing required action field
        audit_log = AuditLog(
            resource_type="test"
        )

        db_session.add(audit_log)
        with pytest.raises(IntegrityError):
            db_session.commit()

    def test_audit_log_compliance_flags(self, db_session: Session):
        """Test compliance-related audit logging."""
        audit_log = AuditLog(
            action="data.export",
            resource_type="user_data",
            resource_id="user123",
            success=True,
            sox_relevant=True,
            gdpr_relevant=True,
            action_metadata={
                "export_format": "csv",
                "data_types": ["personal_info", "access_logs"],
                "purpose": "compliance_audit"
            }
        )

        db_session.add(audit_log)
        db_session.commit()

        assert audit_log.sox_relevant is True
        assert audit_log.gdpr_relevant is True
        assert "personal_info" in audit_log.action_metadata["data_types"]

    def test_audit_log_failure_tracking(self, db_session: Session):
        """Test tracking failed actions."""
        audit_log = AuditLog(
            action="permission.grant",
            resource_type="application",
            resource_id="app123",
            success=False,
            error_message="Insufficient privileges to grant admin access",
            changes=None,  # No changes since it failed
            action_metadata={
                "requested_permission": "admin",
                "requestor_role": "user"
            }
        )

        db_session.add(audit_log)
        db_session.commit()

        assert audit_log.success is False
        assert "Insufficient privileges" in audit_log.error_message
        assert audit_log.changes is None

    def test_audit_log_user_relationship(self, db_session: Session):
        """Test relationship with User model."""
        user = User(
            email="test@example.com",
            full_name="Test User",
            is_active=True
        )
        db_session.add(user)
        db_session.commit()

        # Create multiple audit logs for the user
        for i in range(3):
            audit_log = AuditLog(
                user_id=user.id,
                action=f"action.{i}",
                resource_type="test",
                resource_id=str(i)
            )
            db_session.add(audit_log)

        db_session.commit()
        db_session.refresh(user)

        # Check relationship
        assert len(user.audit_logs) == 3
        assert all(log.user_id == user.id for log in user.audit_logs)

    def test_audit_log_search(self, db_session: Session):
        """Test searching and filtering audit logs."""
        # Create various audit logs
        actions = ["user.login", "user.logout", "data.export", "permission.change"]

        for action in actions:
            audit_log = AuditLog(
                action=action,
                resource_type="test",
                resource_id="123",
                success=True
            )
            db_session.add(audit_log)

        db_session.commit()

        # Search by action pattern
        login_logs = db_session.query(AuditLog).filter(
            AuditLog.action.like("user.%")
        ).all()
        assert len(login_logs) == 2

        # Search by success status
        successful_logs = db_session.query(AuditLog).filter(
            AuditLog.success == True
        ).all()
        assert len(successful_logs) == 4

    def test_audit_log_retention(self, db_session: Session):
        """Test audit log retention policies."""
        # Create old and new audit logs
        old_date = datetime.utcnow() - timedelta(days=400)
        new_date = datetime.utcnow() - timedelta(days=30)

        old_log = AuditLog(
            action="old.action",
            resource_type="test",
            resource_id="old",
            created_at=old_date
        )

        new_log = AuditLog(
            action="new.action",
            resource_type="test",
            resource_id="new",
            created_at=new_date
        )

        db_session.add_all([old_log, new_log])
        db_session.commit()

        # Query logs older than 365 days
        retention_date = datetime.utcnow() - timedelta(days=365)
        old_logs = db_session.query(AuditLog).filter(
            AuditLog.created_at < retention_date
        ).all()

        assert len(old_logs) == 1
        assert old_logs[0].action == "old.action"


class TestIdentityEventModel:
    """Test suite for Identity Event model."""

    def test_identity_event_creation(self, db_session: Session):
        """Test creating an identity event with all fields."""
        # Create related models
        app = SaaSApplication(
            name="Test App",
            provider=SaaSProvider.OKTA,
            api_endpoint="https://test.okta.com",
            auth_type="oauth2"
        )
        db_session.add(app)
        db_session.commit()

        identity = Identity(
            provider=SaaSProvider.OKTA,
            external_id="okta123",
            email="user@example.com",
            username="testuser",
            saas_app_id=app.id
        )
        db_session.add(identity)
        db_session.commit()

        event = IdentityEvent(
            event_id=str(uuid.uuid4()),
            event_type=EventType.USER_CREATED,
            provider=SaaSProvider.OKTA,
            external_id="okta123",
            identity_id=identity.id,
            saas_app_id=app.id,
            timestamp=datetime.utcnow(),
            data={
                "email": "user@example.com",
                "department": "Engineering",
                "manager": "manager@example.com"
            },
            status=EventStatus.PENDING
        )

        db_session.add(event)
        db_session.commit()

        assert event.id is not None
        assert event.event_type == EventType.USER_CREATED
        assert event.status == EventStatus.PENDING
        assert event.data["department"] == "Engineering"

    def test_identity_event_types(self, db_session: Session):
        """Test all supported event types."""
        event_types = [
            EventType.USER_CREATED,
            EventType.USER_UPDATED,
            EventType.USER_DELETED,
            EventType.USER_SUSPENDED,
            EventType.USER_ACTIVATED,
            EventType.PERMISSION_GRANTED,
            EventType.PERMISSION_REVOKED,
            EventType.LOGIN_SUCCESS,
            EventType.LOGIN_FAILURE,
            EventType.SYNC_STARTED,
            EventType.SYNC_COMPLETED,
            EventType.SYNC_FAILED
        ]

        for event_type in event_types:
            event = IdentityEvent(
                event_id=str(uuid.uuid4()),
                event_type=event_type,
                provider=SaaSProvider.OKTA,
                external_id=f"test_{event_type.value}",
                timestamp=datetime.utcnow(),
                data={"test": True}
            )
            db_session.add(event)

        db_session.commit()

        # Verify all event types work
        assert db_session.query(IdentityEvent).count() == len(event_types)

    def test_identity_event_processing(self, db_session: Session):
        """Test event processing workflow."""
        event = IdentityEvent(
            event_id=str(uuid.uuid4()),
            event_type=EventType.USER_UPDATED,
            provider=SaaSProvider.GOOGLE,
            external_id="google123",
            timestamp=datetime.utcnow(),
            data={"email": "updated@example.com"},
            status=EventStatus.PENDING
        )

        db_session.add(event)
        db_session.commit()

        # Process event
        event.status = EventStatus.PROCESSING
        event.processed_at = datetime.utcnow()
        db_session.commit()

        assert event.status == EventStatus.PROCESSING
        assert event.processed_at is not None

        # Complete processing
        event.status = EventStatus.PROCESSED
        event.processing_result = {"updated_fields": ["email"]}
        db_session.commit()

        assert event.status == EventStatus.PROCESSED
        assert "updated_fields" in event.processing_result

    def test_identity_event_error_handling(self, db_session: Session):
        """Test event error handling."""
        event = IdentityEvent(
            event_id=str(uuid.uuid4()),
            event_type=EventType.USER_CREATED,
            provider=SaaSProvider.SLACK,
            external_id="slack123",
            timestamp=datetime.utcnow(),
            data={"email": "test@example.com"},
            status=EventStatus.PROCESSING
        )

        db_session.add(event)
        db_session.commit()

        # Simulate processing error
        event.status = EventStatus.FAILED
        event.error_message = "User already exists in target system"
        event.retry_count = 1
        db_session.commit()

        assert event.status == EventStatus.FAILED
        assert "already exists" in event.error_message
        assert event.retry_count == 1

    def test_identity_event_correlation(self, db_session: Session):
        """Test event correlation functionality."""
        correlation_id = str(uuid.uuid4())

        # Create correlated events
        events = []
        for i in range(3):
            event = IdentityEvent(
                event_id=str(uuid.uuid4()),
                event_type=EventType.USER_UPDATED,
                provider=SaaSProvider.OKTA,
                external_id="okta123",
                timestamp=datetime.utcnow() + timedelta(seconds=i),
                data={"field": f"value{i}"},
                correlation_id=correlation_id
            )
            events.append(event)
            db_session.add(event)

        db_session.commit()

        # Query correlated events
        correlated = db_session.query(IdentityEvent).filter(
            IdentityEvent.correlation_id == correlation_id
        ).all()

        assert len(correlated) == 3
        assert all(e.correlation_id == correlation_id for e in correlated)

    def test_identity_event_duplicate_detection(self, db_session: Session):
        """Test duplicate event detection."""
        event_id = str(uuid.uuid4())

        # Create first event
        event1 = IdentityEvent(
            event_id=event_id,
            event_type=EventType.USER_CREATED,
            provider=SaaSProvider.GITHUB,
            external_id="github123",
            timestamp=datetime.utcnow(),
            data={"test": True}
        )
        db_session.add(event1)
        db_session.commit()

        # Try to create duplicate
        event2 = IdentityEvent(
            event_id=event_id,  # Same event ID
            event_type=EventType.USER_CREATED,
            provider=SaaSProvider.GITHUB,
            external_id="github123",
            timestamp=datetime.utcnow(),
            data={"test": True}
        )
        db_session.add(event2)

        with pytest.raises(IntegrityError):
            db_session.commit()

    def test_identity_event_batch_operations(self, db_session: Session):
        """Test batch event operations."""
        batch_id = str(uuid.uuid4())
        events = []

        # Create batch of events
        for i in range(10):
            event = IdentityEvent(
                event_id=str(uuid.uuid4()),
                event_type=EventType.USER_CREATED,
                provider=SaaSProvider.OKTA,
                external_id=f"okta{i}",
                timestamp=datetime.utcnow(),
                data={"index": i},
                batch_id=batch_id
            )
            events.append(event)
            db_session.add(event)

        db_session.commit()

        # Query batch
        batch_events = db_session.query(IdentityEvent).filter(
            IdentityEvent.batch_id == batch_id
        ).all()

        assert len(batch_events) == 10
        assert all(e.batch_id == batch_id for e in batch_events)

    def test_identity_event_metrics(self, db_session: Session):
        """Test event metrics and statistics."""
        # Create events with different statuses
        statuses = [
            (EventStatus.PROCESSED, 5),
            (EventStatus.FAILED, 2),
            (EventStatus.PENDING, 3)
        ]

        for status, count in statuses:
            for i in range(count):
                event = IdentityEvent(
                    event_id=str(uuid.uuid4()),
                    event_type=EventType.USER_UPDATED,
                    provider=SaaSProvider.GOOGLE,
                    external_id=f"google_{status.value}_{i}",
                    timestamp=datetime.utcnow(),
                    data={"test": True},
                    status=status
                )
                db_session.add(event)

        db_session.commit()

        # Calculate metrics
        metrics = {}
        for status in EventStatus:
            count = db_session.query(IdentityEvent).filter(
                IdentityEvent.status == status
            ).count()
            metrics[status.value] = count

        assert metrics["processed"] == 5
        assert metrics["failed"] == 2
        assert metrics["pending"] == 3

    def test_identity_event_to_dict(self, db_session: Session):
        """Test event serialization."""
        event = IdentityEvent(
            event_id=str(uuid.uuid4()),
            event_type=EventType.PERMISSION_GRANTED,
            provider=SaaSProvider.SALESFORCE,
            external_id="sf123",
            timestamp=datetime.utcnow(),
            data={
                "permission": "admin",
                "resource": "opportunity"
            },
            status=EventStatus.PROCESSED
        )

        db_session.add(event)
        db_session.commit()

        event_dict = event.to_dict()

        assert event_dict["event_type"] == "permission_granted"
        assert event_dict["provider"] == "salesforce"
        assert event_dict["status"] == "processed"
        assert event_dict["data"]["permission"] == "admin"
