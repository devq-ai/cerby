"""
Base classes and interfaces for the ingestion system.

This module provides the foundation for all ingestion handlers including
abstract base classes, common data structures, and shared utilities.
"""

from abc import ABC, abstractmethod
from datetime import datetime
from typing import Dict, List, Any, Optional, Union, Callable
from enum import Enum
from dataclasses import dataclass, field
import asyncio
import uuid
from sqlalchemy.ext.asyncio import AsyncSession
import logfire

from src.db.models.identity import Identity, IdentityProvider
from src.db.models.saas_app import SaaSApplication
from src.db.models.audit import IdentityEvent, EventType
from src.core.config import settings


class IngestionStatus(str, Enum):
    """Status of an ingestion operation."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    PARTIAL = "partial"
    CANCELLED = "cancelled"


class IngestionError(Exception):
    """Base exception for ingestion errors."""

    def __init__(self, message: str, provider: Optional[str] = None,
                 error_code: Optional[str] = None, details: Optional[Dict[str, Any]] = None):
        super().__init__(message)
        self.provider = provider
        self.error_code = error_code
        self.details = details or {}
        self.timestamp = datetime.utcnow()


@dataclass
class IngestionResult:
    """Result of an ingestion operation."""

    status: IngestionStatus
    provider: str
    started_at: datetime
    completed_at: Optional[datetime] = None

    # Counts
    total_records: int = 0
    processed_records: int = 0
    created_records: int = 0
    updated_records: int = 0
    failed_records: int = 0
    skipped_records: int = 0

    # Errors and warnings
    errors: List[IngestionError] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    # Metadata
    batch_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    correlation_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def add_error(self, error: Union[str, IngestionError]) -> None:
        """Add an error to the result."""
        if isinstance(error, str):
            error = IngestionError(error, provider=self.provider)
        self.errors.append(error)
        self.failed_records += 1

    def add_warning(self, warning: str) -> None:
        """Add a warning to the result."""
        self.warnings.append(warning)

    def complete(self) -> None:
        """Mark the ingestion as completed."""
        self.completed_at = datetime.utcnow()
        if self.failed_records > 0 and self.processed_records > 0:
            self.status = IngestionStatus.PARTIAL
        elif self.failed_records > 0:
            self.status = IngestionStatus.FAILED
        else:
            self.status = IngestionStatus.COMPLETED

    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary."""
        return {
            "status": self.status.value,
            "provider": self.provider,
            "started_at": self.started_at.isoformat(),
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "batch_id": self.batch_id,
            "correlation_id": self.correlation_id,
            "statistics": {
                "total": self.total_records,
                "processed": self.processed_records,
                "created": self.created_records,
                "updated": self.updated_records,
                "failed": self.failed_records,
                "skipped": self.skipped_records,
                "success_rate": (self.processed_records - self.failed_records) / self.processed_records
                               if self.processed_records > 0 else 0
            },
            "errors": [
                {
                    "message": str(e),
                    "timestamp": e.timestamp.isoformat(),
                    "details": e.details
                } for e in self.errors[:10]  # Limit to first 10 errors
            ],
            "warnings": self.warnings[:10],  # Limit to first 10 warnings
            "metadata": self.metadata
        }


class BaseIngestionHandler(ABC):
    """
    Abstract base class for all ingestion handlers.

    This class provides common functionality for ingesting identity data
    from various sources including validation, transformation, and persistence.
    """

    def __init__(self, db_session: AsyncSession, saas_app: SaaSApplication):
        self.db_session = db_session
        self.saas_app = saas_app
        self.provider = saas_app.provider
        self.result = IngestionResult(
            status=IngestionStatus.PENDING,
            provider=self.provider,
            started_at=datetime.utcnow()
        )

        # Callbacks for extensibility
        self.pre_process_callbacks: List[Callable] = []
        self.post_process_callbacks: List[Callable] = []
        self.error_callbacks: List[Callable] = []

    @abstractmethod
    async def ingest(self, **kwargs) -> IngestionResult:
        """
        Main ingestion method to be implemented by subclasses.

        Returns:
            IngestionResult with statistics and any errors
        """
        pass

    @abstractmethod
    async def validate_data(self, data: Dict[str, Any]) -> bool:
        """
        Validate incoming data according to provider-specific rules.

        Args:
            data: Raw data from the provider

        Returns:
            True if data is valid, False otherwise
        """
        pass

    @abstractmethod
    async def transform_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Transform provider-specific data to internal schema.

        Args:
            data: Validated raw data

        Returns:
            Transformed data matching internal schema
        """
        pass

    async def process_identity(self, identity_data: Dict[str, Any]) -> Optional[Identity]:
        """
        Process a single identity record.

        Args:
            identity_data: Transformed identity data

        Returns:
            Created or updated Identity object, or None if skipped
        """
        with logfire.span("Process identity", provider=self.provider):
            try:
                # Run pre-process callbacks
                for callback in self.pre_process_callbacks:
                    identity_data = await callback(identity_data)

                # Check if identity already exists
                existing_identity = await self._find_existing_identity(
                    identity_data["external_id"]
                )

                if existing_identity:
                    # Update existing identity
                    identity = await self._update_identity(existing_identity, identity_data)
                    self.result.updated_records += 1
                else:
                    # Create new identity
                    identity = await self._create_identity(identity_data)
                    self.result.created_records += 1

                # Create identity event
                await self._create_identity_event(
                    identity,
                    EventType.USER_CREATED if not existing_identity else EventType.USER_UPDATED,
                    identity_data
                )

                # Run post-process callbacks
                for callback in self.post_process_callbacks:
                    await callback(identity)

                self.result.processed_records += 1
                return identity

            except Exception as e:
                logfire.error(
                    "Failed to process identity",
                    provider=self.provider,
                    external_id=identity_data.get("external_id"),
                    error=str(e)
                )

                # Run error callbacks
                for callback in self.error_callbacks:
                    await callback(identity_data, e)

                self.result.add_error(
                    IngestionError(
                        str(e),
                        provider=self.provider,
                        details={"identity_data": identity_data}
                    )
                )
                return None

    async def _find_existing_identity(self, external_id: str) -> Optional[Identity]:
        """Find existing identity by provider and external ID."""
        from sqlalchemy import select

        stmt = select(Identity).where(
            Identity.provider == self.provider,
            Identity.external_id == external_id
        )
        result = await self.db_session.execute(stmt)
        return result.scalar_one_or_none()

    async def _create_identity(self, data: Dict[str, Any]) -> Identity:
        """Create a new identity."""
        identity = Identity(
            provider=self.provider,
            external_id=data["external_id"],
            email=data["email"],
            username=data.get("username"),
            display_name=data.get("display_name"),
            first_name=data.get("first_name"),
            last_name=data.get("last_name"),
            department=data.get("department"),
            job_title=data.get("job_title"),
            manager_email=data.get("manager_email"),
            employee_id=data.get("employee_id"),
            location=data.get("location"),
            is_privileged=data.get("is_privileged", False),
            is_service_account=data.get("is_service_account", False),
            provider_attributes=data.get("provider_attributes", {}),
            saas_app_id=self.saas_app.id,
            provisioned_at=data.get("provisioned_at", datetime.utcnow()),
            last_sync_at=datetime.utcnow()
        )

        self.db_session.add(identity)
        await self.db_session.flush()
        return identity

    async def _update_identity(self, identity: Identity, data: Dict[str, Any]) -> Identity:
        """Update existing identity with new data."""
        # Track changes for versioning
        changes = {}

        # Update fields if changed
        fields_to_update = [
            "email", "username", "display_name", "first_name", "last_name",
            "department", "job_title", "manager_email", "employee_id",
            "location", "is_privileged", "is_service_account"
        ]

        for field in fields_to_update:
            if field in data and getattr(identity, field) != data[field]:
                changes[field] = {
                    "old": getattr(identity, field),
                    "new": data[field]
                }
                setattr(identity, field, data[field])

        # Update provider attributes
        if "provider_attributes" in data:
            identity.provider_attributes = data["provider_attributes"]

        # Update sync timestamp
        identity.last_sync_at = datetime.utcnow()
        identity.clear_sync_errors()

        # Increment version if there were changes
        if changes:
            identity.increment_version()

        await self.db_session.flush()
        return identity

    async def _create_identity_event(self, identity: Identity, event_type: EventType,
                                   data: Dict[str, Any]) -> IdentityEvent:
        """Create an identity event for audit trail."""
        event = IdentityEvent(
            event_type=event_type.value,
            identity_id=identity.id,
            external_id=identity.external_id,
            provider=self.provider,
            occurred_at=datetime.utcnow(),
            event_data=data,
            changed_fields=list(data.keys()),
            actor_type="system",
            correlation_id=self.result.correlation_id,
            is_processed=True,
            processed_at=datetime.utcnow()
        )

        self.db_session.add(event)
        await self.db_session.flush()
        return event

    async def process_batch(self, identities: List[Dict[str, Any]],
                          batch_size: int = 100) -> None:
        """
        Process a batch of identities efficiently.

        Args:
            identities: List of identity data to process
            batch_size: Number of identities to process in each batch
        """
        self.result.total_records = len(identities)

        with logfire.span("Process batch", provider=self.provider, total=len(identities)):
            for i in range(0, len(identities), batch_size):
                batch = identities[i:i + batch_size]

                # Process batch concurrently
                tasks = [self.process_identity(identity_data) for identity_data in batch]
                await asyncio.gather(*tasks)

                # Commit batch
                await self.db_session.commit()

                logfire.info(
                    "Batch processed",
                    provider=self.provider,
                    batch_number=i // batch_size + 1,
                    processed=min(i + batch_size, len(identities))
                )

    def add_pre_process_callback(self, callback: Callable) -> None:
        """Add a callback to run before processing each identity."""
        self.pre_process_callbacks.append(callback)

    def add_post_process_callback(self, callback: Callable) -> None:
        """Add a callback to run after processing each identity."""
        self.post_process_callbacks.append(callback)

    def add_error_callback(self, callback: Callable) -> None:
        """Add a callback to run when an error occurs."""
        self.error_callbacks.append(callback)
