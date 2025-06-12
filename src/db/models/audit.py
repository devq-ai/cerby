"""
Audit and Event models for Cerby Identity Automation Platform.

This module defines the AuditLog and IdentityEvent models which track
all system activities for compliance and monitoring purposes.
"""

from datetime import datetime
from typing import Optional, Dict, Any, TYPE_CHECKING
from enum import Enum
from sqlalchemy import Column, String, Boolean, DateTime, Text, JSON, Integer, ForeignKey, Index
from sqlalchemy.orm import relationship, Mapped
from sqlalchemy.dialects.postgresql import UUID
import uuid

from src.db.database import BaseModel

if TYPE_CHECKING:
    from src.db.models.user import User
    from src.db.models.identity import Identity
    from src.db.models.policy import AccessPolicy


class AuditAction(str, Enum):
    """Types of auditable actions."""
    # Identity actions
    IDENTITY_CREATED = "identity.created"
    IDENTITY_UPDATED = "identity.updated"
    IDENTITY_DELETED = "identity.deleted"
    IDENTITY_SUSPENDED = "identity.suspended"
    IDENTITY_REACTIVATED = "identity.reactivated"
    IDENTITY_SYNC = "identity.sync"

    # Policy actions
    POLICY_CREATED = "policy.created"
    POLICY_UPDATED = "policy.updated"
    POLICY_DELETED = "policy.deleted"
    POLICY_EVALUATED = "policy.evaluated"
    POLICY_ACTIVATED = "policy.activated"
    POLICY_DEACTIVATED = "policy.deactivated"

    # Access actions
    ACCESS_GRANTED = "access.granted"
    ACCESS_DENIED = "access.denied"
    ACCESS_REVOKED = "access.revoked"
    ACCESS_REQUESTED = "access.requested"

    # User actions
    USER_LOGIN = "user.login"
    USER_LOGOUT = "user.logout"
    USER_CREATED = "user.created"
    USER_UPDATED = "user.updated"
    USER_DELETED = "user.deleted"
    USER_PASSWORD_CHANGED = "user.password_changed"

    # System actions
    SYSTEM_CONFIG_CHANGED = "system.config_changed"
    SYSTEM_ERROR = "system.error"
    SYSTEM_MAINTENANCE = "system.maintenance"

    # Compliance actions
    COMPLIANCE_REPORT_GENERATED = "compliance.report_generated"
    COMPLIANCE_VIOLATION_DETECTED = "compliance.violation_detected"
    COMPLIANCE_REMEDIATION = "compliance.remediation"


class EventType(str, Enum):
    """Types of identity events."""
    # Lifecycle events
    USER_CREATED = "user.created"
    USER_UPDATED = "user.updated"
    USER_DELETED = "user.deleted"
    USER_SUSPENDED = "user.suspended"
    USER_ACTIVATED = "user.activated"

    # Authentication events
    LOGIN_SUCCESS = "auth.login_success"
    LOGIN_FAILURE = "auth.login_failure"
    LOGOUT = "auth.logout"
    PASSWORD_CHANGED = "auth.password_changed"
    PASSWORD_RESET = "auth.password_reset"
    MFA_ENABLED = "auth.mfa_enabled"
    MFA_DISABLED = "auth.mfa_disabled"

    # Access events
    ACCESS_GRANTED = "access.granted"
    ACCESS_REVOKED = "access.revoked"
    PERMISSION_ADDED = "permission.added"
    PERMISSION_REMOVED = "permission.removed"
    ROLE_ASSIGNED = "role.assigned"
    ROLE_REMOVED = "role.removed"

    # Group events
    GROUP_JOINED = "group.joined"
    GROUP_LEFT = "group.left"
    GROUP_CREATED = "group.created"
    GROUP_DELETED = "group.deleted"

    # Sync events
    SYNC_STARTED = "sync.started"
    SYNC_COMPLETED = "sync.completed"
    SYNC_FAILED = "sync.failed"
    SYNC_CONFLICT = "sync.conflict"


class AuditLog(BaseModel):
    """
    AuditLog model for tracking all system activities.

    This model provides comprehensive audit trail for compliance requirements
    including SOX and GDPR. All significant actions are logged here.
    """

    __tablename__ = "audit_logs"
    __table_args__ = (
        Index('ix_audit_action', 'action'),
        Index('ix_audit_user', 'user_id'),
        Index('ix_audit_timestamp', 'created_at'),
        Index('ix_audit_entity', 'entity_type', 'entity_id'),
        Index('ix_audit_compliance', 'compliance_relevant'),
    )

    # Audit identifiers
    uuid = Column(UUID(as_uuid=True), default=uuid.uuid4, unique=True, nullable=False)
    action = Column(String(100), nullable=False)

    # Actor information
    user_id = Column(Integer, ForeignKey("users.id"), nullable=True)
    user_email = Column(String(255), nullable=True)  # Denormalized for history
    user_ip = Column(String(45), nullable=True)  # Support IPv6
    user_agent = Column(String(500), nullable=True)

    # Entity information
    entity_type = Column(String(50), nullable=True)  # e.g., "identity", "policy", "user"
    entity_id = Column(Integer, nullable=True)
    entity_uuid = Column(UUID(as_uuid=True), nullable=True)
    entity_name = Column(String(255), nullable=True)  # Denormalized for history

    # Action details
    changes = Column(JSON, nullable=True)  # Before/after values for updates
    action_metadata = Column(JSON, default=dict, nullable=False)  # Additional context
    reason = Column(Text, nullable=True)  # Business justification

    # Compliance flags
    compliance_relevant = Column(Boolean, default=False, nullable=False)
    compliance_frameworks = Column(JSON, default=list, nullable=False)  # ["SOX", "GDPR"]
    data_classification = Column(String(50), nullable=True)  # e.g., "PII", "SENSITIVE"

    # Request context
    request_id = Column(UUID(as_uuid=True), nullable=True)
    session_id = Column(String(255), nullable=True)
    api_endpoint = Column(String(255), nullable=True)
    http_method = Column(String(10), nullable=True)

    # Result information
    success = Column(Boolean, default=True, nullable=False)
    error_message = Column(Text, nullable=True)
    duration_ms = Column(Integer, nullable=True)

    # Retention and archival
    retention_until = Column(DateTime, nullable=True)
    is_archived = Column(Boolean, default=False, nullable=False)
    archived_at = Column(DateTime, nullable=True)

    # Relationships
    user: Mapped[Optional["User"]] = relationship(
        "User",
        back_populates="audit_logs",
        foreign_keys=[user_id]
    )

    # Methods
    def is_high_risk_action(self) -> bool:
        """Check if this is a high-risk action requiring additional scrutiny."""
        high_risk_actions = [
            AuditAction.USER_DELETED,
            AuditAction.POLICY_DELETED,
            AuditAction.SYSTEM_CONFIG_CHANGED,
            AuditAction.COMPLIANCE_VIOLATION_DETECTED,
            AuditAction.ACCESS_GRANTED,  # When granting privileged access
        ]
        return self.action in high_risk_actions

    def requires_retention(self) -> bool:
        """Check if this audit log requires extended retention."""
        return (
            self.compliance_relevant or
            self.is_high_risk_action() or
            not self.success
        )

    def anonymize_pii(self) -> None:
        """Anonymize PII data for GDPR compliance while maintaining audit integrity."""
        # Hash email addresses
        if self.user_email:
            import hashlib
            self.user_email = hashlib.sha256(self.user_email.encode()).hexdigest()[:16] + "@anonymized"

        # Remove IP address
        if self.user_ip:
            self.user_ip = "0.0.0.0"

        # Clean metadata
        if self.metadata and isinstance(self.metadata, dict):
            pii_fields = ["email", "name", "phone", "address", "ssn", "employee_id"]
            for field in pii_fields:
                if field in self.metadata:
                    self.metadata[field] = "[REDACTED]"

    def to_dict(self, include_sensitive: bool = False) -> Dict[str, Any]:
        """Convert audit log to dictionary representation."""
        data = {
            "id": self.id,
            "uuid": str(self.uuid),
            "action": self.action,
            "user_id": self.user_id,
            "user_email": self.user_email,
            "entity_type": self.entity_type,
            "entity_id": self.entity_id,
            "entity_name": self.entity_name,
            "success": self.success,
            "compliance_relevant": self.compliance_relevant,
            "compliance_frameworks": self.compliance_frameworks,
            "created_at": self.created_at.isoformat(),
        }

        if include_sensitive:
            data.update({
                "user_ip": self.user_ip,
                "user_agent": self.user_agent,
                "changes": self.changes,
                "metadata": self.metadata,
                "reason": self.reason,
                "error_message": self.error_message,
                "request_id": str(self.request_id) if self.request_id else None,
                "session_id": self.session_id,
                "api_endpoint": self.api_endpoint,
                "duration_ms": self.duration_ms,
            })

        return data

    def __repr__(self) -> str:
        """String representation of AuditLog."""
        return f"<AuditLog(id={self.id}, action='{self.action}', user='{self.user_email}', entity='{self.entity_type}:{self.entity_id}')>"


class IdentityEvent(BaseModel):
    """
    IdentityEvent model for tracking identity-specific events.

    This model captures events from various identity providers and
    tracks the lifecycle of identities across different systems.
    """

    __tablename__ = "identity_events"
    __table_args__ = (
        Index('ix_event_type', 'event_type'),
        Index('ix_event_identity', 'identity_id'),
        Index('ix_event_provider', 'provider'),
        Index('ix_event_occurred', 'occurred_at'),
        Index('ix_event_provider_time', 'provider', 'occurred_at'),
    )

    # Event identifiers
    uuid = Column(UUID(as_uuid=True), default=uuid.uuid4, unique=True, nullable=False)
    event_type = Column(String(50), nullable=False)
    event_id = Column(String(255), nullable=True)  # Provider's event ID

    # Identity reference
    identity_id = Column(Integer, ForeignKey("identities.id"), nullable=False)
    external_id = Column(String(255), nullable=False)  # Denormalized

    # Provider information
    provider = Column(String(50), nullable=False)
    provider_event_id = Column(String(255), nullable=True)

    # Event details
    occurred_at = Column(DateTime, nullable=False)
    received_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    processed_at = Column(DateTime, nullable=True)

    # Event data
    event_data = Column(JSON, default=dict, nullable=False)
    changed_fields = Column(JSON, default=list, nullable=False)  # List of fields that changed
    previous_values = Column(JSON, default=dict, nullable=False)  # Previous field values

    # Actor information (who triggered the event)
    actor_id = Column(String(255), nullable=True)
    actor_email = Column(String(255), nullable=True)
    actor_name = Column(String(255), nullable=True)
    actor_type = Column(String(50), nullable=True)  # "user", "system", "admin"

    # Processing information
    is_processed = Column(Boolean, default=False, nullable=False)
    processing_errors = Column(JSON, default=list, nullable=False)
    retry_count = Column(Integer, default=0, nullable=False)

    # Risk and anomaly detection
    risk_score = Column(Integer, nullable=True)
    anomaly_detected = Column(Boolean, default=False, nullable=False)
    anomaly_details = Column(JSON, nullable=True)

    # Compliance and security
    requires_approval = Column(Boolean, default=False, nullable=False)
    approval_status = Column(String(20), nullable=True)  # "pending", "approved", "rejected"
    approved_by = Column(Integer, ForeignKey("users.id"), nullable=True)
    approved_at = Column(DateTime, nullable=True)

    # Correlation
    correlation_id = Column(UUID(as_uuid=True), nullable=True)  # Group related events
    parent_event_id = Column(Integer, ForeignKey("identity_events.id"), nullable=True)

    # Relationships
    identity: Mapped["Identity"] = relationship(
        "Identity",
        back_populates="events"
    )

    parent_event: Mapped[Optional["IdentityEvent"]] = relationship(
        "IdentityEvent",
        remote_side="IdentityEvent.id",
        backref="child_events"
    )

    # Methods
    def is_high_risk(self) -> bool:
        """Check if this event represents a high-risk activity."""
        high_risk_events = [
            EventType.USER_DELETED,
            EventType.PASSWORD_CHANGED,
            EventType.MFA_DISABLED,
            EventType.ACCESS_GRANTED,
            EventType.PERMISSION_ADDED,
        ]
        return (
            self.event_type in high_risk_events or
            self.anomaly_detected or
            (self.risk_score and self.risk_score >= 70)
        )

    def calculate_risk_score(self) -> int:
        """Calculate risk score based on event type and context."""
        base_scores = {
            EventType.USER_DELETED: 80,
            EventType.PASSWORD_CHANGED: 60,
            EventType.MFA_DISABLED: 70,
            EventType.ACCESS_GRANTED: 50,
            EventType.LOGIN_FAILURE: 30,
            EventType.USER_CREATED: 40,
        }

        score = base_scores.get(self.event_type, 20)

        # Adjust based on context
        if self.anomaly_detected:
            score += 30

        # Check for unusual timing
        if self.occurred_at:
            hour = self.occurred_at.hour
            if hour < 6 or hour > 22:  # Outside business hours
                score += 10

        # Check actor
        if self.actor_type == "system":
            score -= 10
        elif self.actor_type != "user":
            score += 10

        return min(100, max(0, score))

    def requires_immediate_action(self) -> bool:
        """Check if this event requires immediate attention."""
        return (
            self.is_high_risk() and
            not self.is_processed and
            self.retry_count < 3
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert event to dictionary representation."""
        return {
            "id": self.id,
            "uuid": str(self.uuid),
            "event_type": self.event_type,
            "identity_id": self.identity_id,
            "external_id": self.external_id,
            "provider": self.provider,
            "occurred_at": self.occurred_at.isoformat(),
            "received_at": self.received_at.isoformat(),
            "processed_at": self.processed_at.isoformat() if self.processed_at else None,
            "is_processed": self.is_processed,
            "event_data": self.event_data,
            "changed_fields": self.changed_fields,
            "actor_email": self.actor_email,
            "actor_type": self.actor_type,
            "risk_score": self.risk_score or self.calculate_risk_score(),
            "anomaly_detected": self.anomaly_detected,
            "requires_approval": self.requires_approval,
            "approval_status": self.approval_status,
            "is_high_risk": self.is_high_risk(),
        }

    def __repr__(self) -> str:
        """String representation of IdentityEvent."""
        return f"<IdentityEvent(id={self.id}, type='{self.event_type}', identity={self.identity_id}, provider='{self.provider}')>"
