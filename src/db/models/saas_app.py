"""
SaaS Application model for Cerby Identity Automation Platform.

This module defines the SaaSApplication model which represents various SaaS
applications that are integrated with the identity management system.
"""

from datetime import datetime
from typing import Optional, List, Dict, Any, TYPE_CHECKING
from enum import Enum
from sqlalchemy import Column, String, Boolean, DateTime, Text, JSON, Integer, Index, UniqueConstraint
from sqlalchemy.orm import relationship, Mapped

from src.db.database import BaseModel

if TYPE_CHECKING:
    from src.db.models.identity import Identity
    from src.db.models.policy import AccessPolicy


class SaaSAppType(str, Enum):
    """Types of SaaS applications."""
    IDENTITY_PROVIDER = "identity_provider"
    COLLABORATION = "collaboration"
    PRODUCTIVITY = "productivity"
    DEVELOPMENT = "development"
    CLOUD_STORAGE = "cloud_storage"
    CRM = "crm"
    PROJECT_MANAGEMENT = "project_management"
    COMMUNICATION = "communication"
    SECURITY = "security"
    CUSTOM = "custom"


class AuthType(str, Enum):
    """Authentication types supported by SaaS applications."""
    OAUTH2 = "oauth2"
    SAML = "saml"
    API_KEY = "api_key"
    BASIC_AUTH = "basic_auth"
    CUSTOM = "custom"


class SyncMethod(str, Enum):
    """Methods for syncing identity data."""
    SCIM = "scim"
    REST_API = "rest_api"
    WEBHOOK = "webhook"
    CSV_IMPORT = "csv_import"
    LDAP = "ldap"
    CUSTOM = "custom"


class SaaSApplication(BaseModel):
    """
    SaaSApplication model representing integrated SaaS applications.

    This model stores configuration and metadata for various SaaS applications
    that provide identity data or require identity management.
    """

    __tablename__ = "saas_applications"
    __table_args__ = (
        UniqueConstraint('name', 'tenant_id', name='uq_app_name_tenant'),
        Index('ix_saas_app_provider', 'provider'),
        Index('ix_saas_app_type', 'app_type'),
        Index('ix_saas_app_status', 'is_active'),
    )

    # Application identifiers
    name = Column(String(100), nullable=False)
    provider = Column(String(50), nullable=False)  # e.g., 'okta', 'azure_ad', 'slack'
    tenant_id = Column(String(255), nullable=True)  # For multi-tenant apps

    # Application metadata
    app_type = Column(String(50), default=SaaSAppType.CUSTOM.value, nullable=False)
    description = Column(Text, nullable=True)
    icon_url = Column(String(500), nullable=True)
    website_url = Column(String(500), nullable=True)

    # Integration configuration
    api_endpoint = Column(String(500), nullable=True)
    auth_type = Column(String(50), default=AuthType.API_KEY.value, nullable=False)
    sync_method = Column(String(50), default=SyncMethod.REST_API.value, nullable=False)

    # Authentication credentials (encrypted in production)
    client_id = Column(String(255), nullable=True)
    client_secret_encrypted = Column(Text, nullable=True)
    api_key_encrypted = Column(Text, nullable=True)
    auth_credentials = Column(JSON, default=dict, nullable=False)  # Additional auth data

    # API configuration
    api_version = Column(String(20), nullable=True)
    api_rate_limit = Column(Integer, default=1000, nullable=False)  # Requests per hour
    api_timeout_seconds = Column(Integer, default=30, nullable=False)
    custom_headers = Column(JSON, default=dict, nullable=False)

    # Sync configuration
    sync_enabled = Column(Boolean, default=True, nullable=False)
    sync_interval_minutes = Column(Integer, default=60, nullable=False)
    last_sync_at = Column(DateTime, nullable=True)
    last_successful_sync_at = Column(DateTime, nullable=True)
    sync_filters = Column(JSON, default=dict, nullable=False)  # e.g., {"department": "Engineering"}

    # SCIM configuration (if applicable)
    scim_endpoint = Column(String(500), nullable=True)
    scim_version = Column(String(10), default="2.0", nullable=True)
    scim_bearer_token_encrypted = Column(Text, nullable=True)

    # Webhook configuration (if applicable)
    webhook_url = Column(String(500), nullable=True)
    webhook_secret = Column(String(255), nullable=True)
    webhook_events = Column(JSON, default=list, nullable=False)  # List of subscribed events

    # Field mappings
    field_mappings = Column(JSON, default=dict, nullable=False)  # Map external fields to internal

    # Status and health
    is_active = Column(Boolean, default=True, nullable=False)
    health_status = Column(String(20), default="healthy", nullable=False)
    last_health_check_at = Column(DateTime, nullable=True)
    health_check_errors = Column(JSON, default=list, nullable=False)

    # Statistics
    total_identities_synced = Column(Integer, default=0, nullable=False)
    total_sync_errors = Column(Integer, default=0, nullable=False)
    average_sync_duration_seconds = Column(Integer, nullable=True)

    # Compliance and security
    compliance_certifications = Column(JSON, default=list, nullable=False)  # e.g., ["SOC2", "ISO27001"]
    data_residency = Column(String(50), nullable=True)  # e.g., "US", "EU"
    encryption_at_rest = Column(Boolean, default=True, nullable=False)
    encryption_in_transit = Column(Boolean, default=True, nullable=False)

    # Configuration for specific providers
    provider_config = Column(JSON, default=dict, nullable=False)

    # Relationships
    identities: Mapped[List["Identity"]] = relationship(
        "Identity",
        back_populates="saas_application",
        lazy="dynamic"
    )

    policies: Mapped[List["AccessPolicy"]] = relationship(
        "AccessPolicy",
        secondary="policy_saas_apps",
        back_populates="saas_applications",
        lazy="dynamic"
    )

    # Methods
    def is_identity_provider(self) -> bool:
        """Check if this is an identity provider application."""
        return self.app_type == SaaSAppType.IDENTITY_PROVIDER.value

    def requires_oauth(self) -> bool:
        """Check if application requires OAuth authentication."""
        return self.auth_type in [AuthType.OAUTH2.value, AuthType.SAML.value]

    def supports_scim(self) -> bool:
        """Check if application supports SCIM protocol."""
        return self.sync_method == SyncMethod.SCIM.value and self.scim_endpoint is not None

    def is_sync_due(self) -> bool:
        """Check if sync is due based on interval."""
        if not self.sync_enabled or not self.last_sync_at:
            return self.sync_enabled

        time_since_sync = datetime.utcnow() - self.last_sync_at
        return time_since_sync.total_seconds() >= (self.sync_interval_minutes * 60)

    def record_sync_success(self, identities_count: int, duration_seconds: int) -> None:
        """Record successful sync statistics."""
        self.last_sync_at = datetime.utcnow()
        self.last_successful_sync_at = datetime.utcnow()
        self.total_identities_synced += identities_count

        # Update average sync duration
        if self.average_sync_duration_seconds:
            # Weighted average with more weight on recent syncs
            self.average_sync_duration_seconds = int(
                (self.average_sync_duration_seconds * 0.7) + (duration_seconds * 0.3)
            )
        else:
            self.average_sync_duration_seconds = duration_seconds

        self.health_status = "healthy"
        self.health_check_errors = []

    def record_sync_error(self, error: str) -> None:
        """Record sync error."""
        self.total_sync_errors += 1
        self.health_status = "unhealthy"

        error_record = {
            'timestamp': datetime.utcnow().isoformat(),
            'error': error,
            'sync_attempt': self.total_sync_errors
        }
        self.health_check_errors = self.health_check_errors + [error_record]

        # Keep only last 10 errors
        if len(self.health_check_errors) > 10:
            self.health_check_errors = self.health_check_errors[-10:]

    def get_api_headers(self) -> Dict[str, str]:
        """Get combined API headers including custom headers."""
        headers = {
            'User-Agent': 'Cerby-Identity-Automation/1.0',
            'Accept': 'application/json',
            'Content-Type': 'application/json'
        }
        headers.update(self.custom_headers)
        return headers

    def get_field_mapping(self, external_field: str) -> Optional[str]:
        """Get internal field name for external field."""
        return self.field_mappings.get(external_field)

    def map_external_data(self, external_data: Dict[str, Any]) -> Dict[str, Any]:
        """Map external data to internal schema using field mappings."""
        mapped_data = {}

        for external_field, internal_field in self.field_mappings.items():
            if external_field in external_data:
                mapped_data[internal_field] = external_data[external_field]

        # Include unmapped fields in a separate key
        unmapped_fields = {
            k: v for k, v in external_data.items()
            if k not in self.field_mappings
        }
        if unmapped_fields:
            mapped_data['_unmapped'] = unmapped_fields

        return mapped_data

    def to_dict(self, include_sensitive: bool = False) -> Dict[str, Any]:
        """Convert SaaS application to dictionary representation."""
        data = {
            "id": self.id,
            "name": self.name,
            "provider": self.provider,
            "tenant_id": self.tenant_id,
            "app_type": self.app_type,
            "description": self.description,
            "icon_url": self.icon_url,
            "website_url": self.website_url,
            "auth_type": self.auth_type,
            "sync_method": self.sync_method,
            "sync_enabled": self.sync_enabled,
            "sync_interval_minutes": self.sync_interval_minutes,
            "is_active": self.is_active,
            "health_status": self.health_status,
            "last_sync_at": self.last_sync_at.isoformat() if self.last_sync_at else None,
            "last_successful_sync_at": self.last_successful_sync_at.isoformat() if self.last_successful_sync_at else None,
            "total_identities_synced": self.total_identities_synced,
            "total_sync_errors": self.total_sync_errors,
            "average_sync_duration_seconds": self.average_sync_duration_seconds,
            "compliance_certifications": self.compliance_certifications,
            "data_residency": self.data_residency,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
        }

        if include_sensitive:
            data.update({
                "api_endpoint": self.api_endpoint,
                "client_id": self.client_id,
                "api_version": self.api_version,
                "api_rate_limit": self.api_rate_limit,
                "sync_filters": self.sync_filters,
                "field_mappings": self.field_mappings,
                "provider_config": self.provider_config,
                "health_check_errors": self.health_check_errors,
            })

        return data

    def __repr__(self) -> str:
        """String representation of SaaSApplication."""
        return f"<SaaSApplication(id={self.id}, name='{self.name}', provider='{self.provider}', type='{self.app_type}')>"
