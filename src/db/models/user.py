"""
User model for Cerby Identity Automation Platform.

This module defines the User model which represents internal users of the platform
who manage identities and policies.
"""

from datetime import datetime
from typing import Optional, List, TYPE_CHECKING
from sqlalchemy import Column, String, Boolean, DateTime, Text, JSON
from sqlalchemy.orm import relationship, Mapped
from passlib.context import CryptContext

from src.db.database import BaseModel

if TYPE_CHECKING:
    from src.db.models.identity import Identity
    from src.db.models.audit import AuditLog

# Password hashing context
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")


class User(BaseModel):
    """
    User model representing platform administrators and operators.

    This model stores information about users who can access and manage
    the identity automation platform, not the external identities being managed.
    """

    __tablename__ = "users"

    # Basic user information
    email = Column(String(255), unique=True, nullable=False, index=True)
    username = Column(String(100), unique=True, nullable=False, index=True)
    full_name = Column(String(255), nullable=True)
    hashed_password = Column(String(255), nullable=False)

    # User status and permissions
    is_active = Column(Boolean, default=True, nullable=False)
    is_superuser = Column(Boolean, default=False, nullable=False)
    is_verified = Column(Boolean, default=False, nullable=False)

    # Profile information
    department = Column(String(100), nullable=True)
    job_title = Column(String(100), nullable=True)
    phone_number = Column(String(50), nullable=True)
    timezone = Column(String(50), default="UTC", nullable=False)

    # Authentication tracking
    last_login_at = Column(DateTime, nullable=True)
    last_login_ip = Column(String(45), nullable=True)  # Support IPv6
    failed_login_attempts = Column(Integer, default=0, nullable=False)
    locked_until = Column(DateTime, nullable=True)

    # API access
    api_key_hash = Column(String(255), nullable=True, unique=True)
    api_key_created_at = Column(DateTime, nullable=True)
    api_key_last_used_at = Column(DateTime, nullable=True)

    # Preferences and settings
    preferences = Column(JSON, default=dict, nullable=False)
    notification_settings = Column(JSON, default=dict, nullable=False)

    # Password reset
    reset_token_hash = Column(String(255), nullable=True)
    reset_token_expires_at = Column(DateTime, nullable=True)

    # Email verification
    verification_token_hash = Column(String(255), nullable=True)
    verification_token_expires_at = Column(DateTime, nullable=True)

    # Soft delete
    deleted_at = Column(DateTime, nullable=True)
    deleted_by = Column(Integer, nullable=True)

    # Relationships
    managed_identities: Mapped[List["Identity"]] = relationship(
        "Identity",
        secondary="user_identities",
        back_populates="managing_users",
        lazy="dynamic"
    )

    audit_logs: Mapped[List["AuditLog"]] = relationship(
        "AuditLog",
        back_populates="user",
        lazy="dynamic",
        foreign_keys="AuditLog.user_id"
    )

    # Class methods for password management
    def set_password(self, password: str) -> None:
        """Hash and set user password."""
        self.hashed_password = pwd_context.hash(password)

    def verify_password(self, password: str) -> bool:
        """Verify password against hash."""
        return pwd_context.verify(password, self.hashed_password)

    def is_locked(self) -> bool:
        """Check if user account is locked."""
        if self.locked_until and self.locked_until > datetime.utcnow():
            return True
        return False

    def increment_failed_login(self) -> None:
        """Increment failed login attempts and lock if necessary."""
        self.failed_login_attempts += 1

        # Lock account after 5 failed attempts for 30 minutes
        if self.failed_login_attempts >= 5:
            self.locked_until = datetime.utcnow() + timedelta(minutes=30)

    def reset_failed_login(self) -> None:
        """Reset failed login attempts on successful login."""
        self.failed_login_attempts = 0
        self.locked_until = None
        self.last_login_at = datetime.utcnow()

    def generate_api_key(self) -> str:
        """Generate a new API key for the user."""
        import secrets
        api_key = f"cerby_{secrets.token_urlsafe(32)}"
        self.api_key_hash = pwd_context.hash(api_key)
        self.api_key_created_at = datetime.utcnow()
        return api_key

    def verify_api_key(self, api_key: str) -> bool:
        """Verify API key against hash."""
        if not self.api_key_hash:
            return False
        return pwd_context.verify(api_key, self.api_key_hash)

    def can_manage_identities(self) -> bool:
        """Check if user has permission to manage identities."""
        return self.is_active and (self.is_superuser or
                                   self.preferences.get("can_manage_identities", False))

    def can_manage_policies(self) -> bool:
        """Check if user has permission to manage policies."""
        return self.is_active and (self.is_superuser or
                                   self.preferences.get("can_manage_policies", False))

    def can_view_analytics(self) -> bool:
        """Check if user has permission to view analytics."""
        return self.is_active and (self.is_superuser or
                                   self.preferences.get("can_view_analytics", True))

    def to_dict(self, include_sensitive: bool = False) -> dict:
        """Convert user to dictionary representation."""
        data = {
            "id": self.id,
            "email": self.email,
            "username": self.username,
            "full_name": self.full_name,
            "is_active": self.is_active,
            "is_superuser": self.is_superuser,
            "is_verified": self.is_verified,
            "department": self.department,
            "job_title": self.job_title,
            "timezone": self.timezone,
            "last_login_at": self.last_login_at.isoformat() if self.last_login_at else None,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "preferences": self.preferences,
            "notification_settings": self.notification_settings,
        }

        if include_sensitive:
            data.update({
                "last_login_ip": self.last_login_ip,
                "failed_login_attempts": self.failed_login_attempts,
                "is_locked": self.is_locked(),
                "api_key_created_at": self.api_key_created_at.isoformat() if self.api_key_created_at else None,
                "api_key_last_used_at": self.api_key_last_used_at.isoformat() if self.api_key_last_used_at else None,
            })

        return data

    def __repr__(self) -> str:
        """String representation of User."""
        return f"<User(id={self.id}, email='{self.email}', username='{self.username}')>"


# Import at the bottom to avoid circular imports
from datetime import timedelta
from sqlalchemy import Integer
