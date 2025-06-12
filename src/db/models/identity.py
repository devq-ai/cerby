"""
Identity model for Cerby Identity Automation Platform.

This module defines the Identity model which represents external user identities
from various SaaS applications and identity providers.
"""

from datetime import datetime
from typing import Optional, List, Dict, Any, TYPE_CHECKING
from enum import Enum
from sqlalchemy import Column, String, Boolean, DateTime, Text, JSON, Integer, ForeignKey, Index, UniqueConstraint
from sqlalchemy.orm import relationship, Mapped
from sqlalchemy.dialects.postgresql import UUID
import uuid

from src.db.database import BaseModel

if TYPE_CHECKING:
    from src.db.models.user import User
    from src.db.models.saas_app import SaaSApplication
    from src.db.models.audit import IdentityEvent
    from src.db.models.policy import AccessPolicy


class IdentityProvider(str, Enum):
    """Enumeration of supported identity providers."""
    OKTA = "okta"
    AZURE_AD = "azure_ad"
    GOOGLE_WORKSPACE = "google_workspace"
    SLACK = "slack"
    GITHUB = "github"
    JIRA = "jira"
    CONFLUENCE = "confluence"
    SALESFORCE = "salesforce"
    BOX = "box"
    DROPBOX = "dropbox"
    CUSTOM = "custom"


class IdentityStatus(str, Enum):
    """Identity lifecycle status."""
    ACTIVE = "active"
    SUSPENDED = "suspended"
    DEPROVISIONED = "deprovisioned"
    PENDING = "pending"
    FAILED = "failed"


class Identity(BaseModel):
    """
    Identity model representing external user identities from SaaS applications.

    This model stores identity information synchronized from various identity
    providers and tracks the lifecycle of identities across platforms.
    """

    __tablename__ = "identities"
    __table_args__ = (
        UniqueConstraint('provider', 'external_id', name='uq_provider_external_id'),
        Index('ix_identity_email', 'email'),
        Index('ix_identity_username', 'username'),
        Index('ix_identity_status', 'status'),
        Index('ix_identity_provider_status', 'provider', 'status'),
    )

    # Identity identifiers
    uuid = Column(UUID(as_uuid=True), default=uuid.uuid4, unique=True, nullable=False)
    provider = Column(String(50), nullable=False)
    external_id = Column(String(255), nullable=False)

    # Identity attributes
    email = Column(String(255), nullable=False)
    username = Column(String(100), nullable=True)
    display_name = Column(String(255), nullable=True)
    first_name = Column(String(100), nullable=True)
    last_name = Column(String(100), nullable=True)

    # Identity metadata
    status = Column(String(20), default=IdentityStatus.ACTIVE.value, nullable=False)
    is_privileged = Column(Boolean, default=False, nullable=False)
    is_service_account = Column(Boolean, default=False, nullable=False)

    # Organization information
    department = Column(String(100), nullable=True)
    job_title = Column(String(100), nullable=True)
    manager_email = Column(String(255), nullable=True)
    employee_id = Column(String(50), nullable=True)
    cost_center = Column(String(50), nullable=True)
    location = Column(String(100), nullable=True)

    # Identity lifecycle dates
    provisioned_at = Column(DateTime, nullable=True)
    last_sync_at = Column(DateTime, nullable=True)
    last_login_at = Column(DateTime, nullable=True)
    deprovisioned_at = Column(DateTime, nullable=True)
    password_changed_at = Column(DateTime, nullable=True)

    # Risk and compliance
    risk_score = Column(Integer, default=0, nullable=False)
    compliance_flags = Column(JSON, default=list, nullable=False)
    anomaly_flags = Column(JSON, default=list, nullable=False)

    # Provider-specific attributes
    provider_attributes = Column(JSON, default=dict, nullable=False)

    # Sync metadata
    sync_enabled = Column(Boolean, default=True, nullable=False)
    sync_errors = Column(JSON, default=list, nullable=False)
    last_sync_error = Column(Text, nullable=True)
    sync_retry_count = Column(Integer, default=0, nullable=False)

    # Version tracking for changes
    version = Column(Integer, default=1, nullable=False)
    previous_versions = Column(JSON, default=list, nullable=False)

    # Foreign keys
    saas_app_id = Column(Integer, ForeignKey("saas_applications.id"), nullable=True)

    # Relationships
    managing_users: Mapped[List["User"]] = relationship(
        "User",
        secondary="user_identities",
        back_populates="managed_identities",
        lazy="dynamic"
    )

    saas_application: Mapped[Optional["SaaSApplication"]] = relationship(
        "SaaSApplication",
        back_populates="identities"
    )

    events: Mapped[List["IdentityEvent"]] = relationship(
        "IdentityEvent",
        back_populates="identity",
        lazy="dynamic",
        order_by="desc(IdentityEvent.occurred_at)"
    )

    assigned_policies: Mapped[List["AccessPolicy"]] = relationship(
        "AccessPolicy",
        secondary="policy_assignments",
        back_populates="assigned_identities",
        lazy="dynamic"
    )

    # Methods
    def is_active(self) -> bool:
        """Check if identity is currently active."""
        return self.status == IdentityStatus.ACTIVE.value

    def is_high_risk(self) -> bool:
        """Check if identity is considered high risk."""
        return self.risk_score >= 70

    def has_compliance_issues(self) -> bool:
        """Check if identity has any compliance flags."""
        return len(self.compliance_flags) > 0

    def update_risk_score(self, new_score: int) -> None:
        """Update the risk score with bounds checking."""
        self.risk_score = max(0, min(100, new_score))

    def add_compliance_flag(self, flag: Dict[str, Any]) -> None:
        """Add a compliance flag with timestamp."""
        flag['timestamp'] = datetime.utcnow().isoformat()
        self.compliance_flags = self.compliance_flags + [flag]

    def add_anomaly_flag(self, anomaly: Dict[str, Any]) -> None:
        """Add an anomaly flag with timestamp."""
        anomaly['timestamp'] = datetime.utcnow().isoformat()
        self.anomaly_flags = self.anomaly_flags + [anomaly]

    def record_sync_error(self, error: str) -> None:
        """Record a sync error and increment retry count."""
        self.last_sync_error = error
        self.sync_retry_count += 1
        error_record = {
            'timestamp': datetime.utcnow().isoformat(),
            'error': error,
            'retry_count': self.sync_retry_count
        }
        self.sync_errors = self.sync_errors + [error_record]

    def clear_sync_errors(self) -> None:
        """Clear sync errors after successful sync."""
        self.sync_errors = []
        self.last_sync_error = None
        self.sync_retry_count = 0
        self.last_sync_at = datetime.utcnow()

    def create_version_snapshot(self) -> Dict[str, Any]:
        """Create a snapshot of current identity state for versioning."""
        return {
            'version': self.version,
            'timestamp': datetime.utcnow().isoformat(),
            'email': self.email,
            'username': self.username,
            'display_name': self.display_name,
            'department': self.department,
            'job_title': self.job_title,
            'status': self.status,
            'provider_attributes': self.provider_attributes
        }

    def increment_version(self) -> None:
        """Increment version and save current state."""
        snapshot = self.create_version_snapshot()
        self.previous_versions = self.previous_versions + [snapshot]
        self.version += 1

    def get_full_name(self) -> str:
        """Get full name from first and last name."""
        if self.first_name and self.last_name:
            return f"{self.first_name} {self.last_name}"
        return self.display_name or self.username or self.email

    def to_dict(self, include_sensitive: bool = False) -> Dict[str, Any]:
        """Convert identity to dictionary representation."""
        data = {
            "id": self.id,
            "uuid": str(self.uuid),
            "provider": self.provider,
            "external_id": self.external_id,
            "email": self.email,
            "username": self.username,
            "display_name": self.display_name,
            "full_name": self.get_full_name(),
            "status": self.status,
            "is_privileged": self.is_privileged,
            "is_service_account": self.is_service_account,
            "department": self.department,
            "job_title": self.job_title,
            "location": self.location,
            "risk_score": self.risk_score,
            "has_compliance_issues": self.has_compliance_issues(),
            "provisioned_at": self.provisioned_at.isoformat() if self.provisioned_at else None,
            "last_sync_at": self.last_sync_at.isoformat() if self.last_sync_at else None,
            "last_login_at": self.last_login_at.isoformat() if self.last_login_at else None,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
        }

        if include_sensitive:
            data.update({
                "manager_email": self.manager_email,
                "employee_id": self.employee_id,
                "cost_center": self.cost_center,
                "compliance_flags": self.compliance_flags,
                "anomaly_flags": self.anomaly_flags,
                "provider_attributes": self.provider_attributes,
                "sync_enabled": self.sync_enabled,
                "sync_errors": self.sync_errors,
                "version": self.version,
            })

        return data

    def to_scim_resource(self) -> Dict[str, Any]:
        """Convert identity to SCIM 2.0 resource format."""
        return {
            "schemas": ["urn:ietf:params:scim:schemas:core:2.0:User"],
            "id": self.external_id,
            "externalId": str(self.uuid),
            "userName": self.username or self.email,
            "name": {
                "formatted": self.get_full_name(),
                "givenName": self.first_name,
                "familyName": self.last_name,
            },
            "displayName": self.display_name,
            "emails": [
                {
                    "value": self.email,
                    "type": "work",
                    "primary": True
                }
            ],
            "active": self.is_active(),
            "enterprise": {
                "department": self.department,
                "employeeNumber": self.employee_id,
                "manager": {
                    "value": self.manager_email
                } if self.manager_email else None
            },
            "meta": {
                "resourceType": "User",
                "created": self.created_at.isoformat(),
                "lastModified": self.updated_at.isoformat(),
                "version": f"W/{self.version}"
            }
        }

    def __repr__(self) -> str:
        """String representation of Identity."""
        return f"<Identity(id={self.id}, provider='{self.provider}', email='{self.email}', status='{self.status}')>"
