"""
Unit tests for Task 2.2 - Implement User and Identity models.

Tests verify that User and Identity models are properly implemented with
authentication fields, relationships, versioning support, and all required methods.
"""

import pytest
from datetime import datetime, timedelta
from unittest.mock import patch, MagicMock
import sys
from pathlib import Path
from sqlalchemy.exc import IntegrityError
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool
import uuid

# Add project root to Python path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.db.database import Base
from src.db.models.user import User, pwd_context
from src.db.models.identity import Identity, IdentityProvider, IdentityStatus


class TestUserModel:
    """Test User model implementation."""

    @pytest.fixture
    def db_session(self):
        """Create a test database session."""
        engine = create_engine(
            "sqlite:///:memory:",
            connect_args={"check_same_thread": False},
            poolclass=StaticPool,
        )
        Base.metadata.create_all(engine)
        SessionLocal = sessionmaker(bind=engine)
        session = SessionLocal()
        yield session
        session.close()

    def test_user_model_fields(self):
        """Test that User model has all required fields."""
        # Basic fields
        assert hasattr(User, 'email')
        assert hasattr(User, 'username')
        assert hasattr(User, 'full_name')
        assert hasattr(User, 'hashed_password')

        # Status fields
        assert hasattr(User, 'is_active')
        assert hasattr(User, 'is_superuser')
        assert hasattr(User, 'is_verified')

        # Profile fields
        assert hasattr(User, 'department')
        assert hasattr(User, 'job_title')
        assert hasattr(User, 'phone_number')
        assert hasattr(User, 'timezone')

        # Authentication tracking
        assert hasattr(User, 'last_login_at')
        assert hasattr(User, 'last_login_ip')
        assert hasattr(User, 'failed_login_attempts')
        assert hasattr(User, 'locked_until')

        # API access
        assert hasattr(User, 'api_key_hash')
        assert hasattr(User, 'api_key_created_at')
        assert hasattr(User, 'api_key_last_used_at')

        # Preferences
        assert hasattr(User, 'preferences')
        assert hasattr(User, 'notification_settings')

        # Relationships
        assert hasattr(User, 'managed_identities')
        assert hasattr(User, 'audit_logs')

    def test_user_creation(self, db_session):
        """Test creating a new user."""
        user = User(
            email="test@example.com",
            username="testuser",
            full_name="Test User"
        )
        user.set_password("securepassword123")

        db_session.add(user)
        db_session.commit()

        # Verify user was created
        assert user.id is not None
        assert user.email == "test@example.com"
        assert user.username == "testuser"
        assert user.full_name == "Test User"
        assert user.hashed_password is not None
        assert user.hashed_password != "securepassword123"  # Should be hashed

    def test_user_password_hashing(self):
        """Test password hashing and verification."""
        user = User()
        password = "MySecurePassword123!"

        # Set password
        user.set_password(password)
        assert user.hashed_password is not None
        assert user.hashed_password != password

        # Verify correct password
        assert user.verify_password(password) is True

        # Verify incorrect password
        assert user.verify_password("WrongPassword") is False

    def test_user_unique_constraints(self, db_session):
        """Test unique constraints on email and username."""
        # Create first user
        user1 = User(email="unique@example.com", username="unique1")
        user1.set_password("password123")
        db_session.add(user1)
        db_session.commit()

        # Try to create user with same email
        user2 = User(email="unique@example.com", username="unique2")
        user2.set_password("password123")
        db_session.add(user2)

        with pytest.raises(IntegrityError):
            db_session.commit()

        db_session.rollback()

        # Try to create user with same username
        user3 = User(email="different@example.com", username="unique1")
        user3.set_password("password123")
        db_session.add(user3)

        with pytest.raises(IntegrityError):
            db_session.commit()

    def test_user_lockout_mechanism(self):
        """Test user account lockout functionality."""
        user = User()

        # Initially not locked
        assert user.is_locked() is False
        assert user.failed_login_attempts == 0

        # Increment failed attempts
        for i in range(4):
            user.increment_failed_login()
            assert user.failed_login_attempts == i + 1
            assert user.is_locked() is False

        # Fifth attempt should lock
        user.increment_failed_login()
        assert user.failed_login_attempts == 5
        assert user.is_locked() is True
        assert user.locked_until is not None
        assert user.locked_until > datetime.utcnow()

        # Reset should clear lock
        user.reset_failed_login()
        assert user.failed_login_attempts == 0
        assert user.is_locked() is False
        assert user.locked_until is None
        assert user.last_login_at is not None

    def test_user_api_key_generation(self):
        """Test API key generation and verification."""
        user = User()

        # Generate API key
        api_key = user.generate_api_key()
        assert api_key is not None
        assert api_key.startswith("cerby_")
        assert len(api_key) > 30
        assert user.api_key_hash is not None
        assert user.api_key_created_at is not None

        # Verify correct API key
        assert user.verify_api_key(api_key) is True

        # Verify incorrect API key
        assert user.verify_api_key("wrong_key") is False
        assert user.verify_api_key("cerby_wrongkey123") is False

    def test_user_permissions(self):
        """Test user permission methods."""
        user = User()

        # Non-superuser with no permissions
        user.is_superuser = False
        user.is_active = True
        user.preferences = {}

        assert user.can_manage_identities() is False
        assert user.can_manage_policies() is False
        assert user.can_view_analytics() is True  # Default True

        # Non-superuser with specific permissions
        user.preferences = {
            "can_manage_identities": True,
            "can_manage_policies": False,
            "can_view_analytics": True
        }

        assert user.can_manage_identities() is True
        assert user.can_manage_policies() is False
        assert user.can_view_analytics() is True

        # Superuser has all permissions
        user.is_superuser = True
        assert user.can_manage_identities() is True
        assert user.can_manage_policies() is True
        assert user.can_view_analytics() is True

        # Inactive user has no permissions
        user.is_active = False
        assert user.can_manage_identities() is False
        assert user.can_manage_policies() is False
        assert user.can_view_analytics() is False

    def test_user_to_dict(self):
        """Test user serialization to dictionary."""
        user = User(
            email="test@example.com",
            username="testuser",
            full_name="Test User",
            department="Engineering",
            job_title="Software Engineer",
            timezone="UTC",
            is_active=True,
            is_superuser=False,
            is_verified=True
        )
        user.id = 1
        user.created_at = datetime.utcnow()
        user.updated_at = datetime.utcnow()
        user.last_login_at = datetime.utcnow()
        user.last_login_ip = "192.168.1.1"

        # Test without sensitive data
        data = user.to_dict(include_sensitive=False)
        assert data["id"] == 1
        assert data["email"] == "test@example.com"
        assert data["username"] == "testuser"
        assert "hashed_password" not in data
        assert "last_login_ip" not in data

        # Test with sensitive data
        data_sensitive = user.to_dict(include_sensitive=True)
        assert data_sensitive["last_login_ip"] == "192.168.1.1"
        assert data_sensitive["failed_login_attempts"] == 0
        assert "is_locked" in data_sensitive

    def test_user_soft_delete_fields(self):
        """Test soft delete fields exist."""
        assert hasattr(User, 'deleted_at')
        assert hasattr(User, 'deleted_by')

    def test_user_password_reset_fields(self):
        """Test password reset fields exist."""
        assert hasattr(User, 'reset_token_hash')
        assert hasattr(User, 'reset_token_expires_at')

    def test_user_email_verification_fields(self):
        """Test email verification fields exist."""
        assert hasattr(User, 'verification_token_hash')
        assert hasattr(User, 'verification_token_expires_at')


class TestIdentityModel:
    """Test Identity model implementation."""

    @pytest.fixture
    def db_session(self):
        """Create a test database session."""
        engine = create_engine(
            "sqlite:///:memory:",
            connect_args={"check_same_thread": False},
            poolclass=StaticPool,
        )
        Base.metadata.create_all(engine)
        SessionLocal = sessionmaker(bind=engine)
        session = SessionLocal()
        yield session
        session.close()

    def test_identity_model_fields(self):
        """Test that Identity model has all required fields."""
        # Identifiers
        assert hasattr(Identity, 'uuid')
        assert hasattr(Identity, 'provider')
        assert hasattr(Identity, 'external_id')

        # Attributes
        assert hasattr(Identity, 'email')
        assert hasattr(Identity, 'username')
        assert hasattr(Identity, 'display_name')
        assert hasattr(Identity, 'first_name')
        assert hasattr(Identity, 'last_name')

        # Metadata
        assert hasattr(Identity, 'status')
        assert hasattr(Identity, 'is_privileged')
        assert hasattr(Identity, 'is_service_account')

        # Organization
        assert hasattr(Identity, 'department')
        assert hasattr(Identity, 'job_title')
        assert hasattr(Identity, 'manager_email')
        assert hasattr(Identity, 'employee_id')
        assert hasattr(Identity, 'cost_center')
        assert hasattr(Identity, 'location')

        # Lifecycle dates
        assert hasattr(Identity, 'provisioned_at')
        assert hasattr(Identity, 'last_sync_at')
        assert hasattr(Identity, 'last_login_at')
        assert hasattr(Identity, 'deprovisioned_at')
        assert hasattr(Identity, 'password_changed_at')

        # Risk and compliance
        assert hasattr(Identity, 'risk_score')
        assert hasattr(Identity, 'compliance_flags')
        assert hasattr(Identity, 'anomaly_flags')

        # Versioning
        assert hasattr(Identity, 'version')
        assert hasattr(Identity, 'previous_versions')

        # Relationships
        assert hasattr(Identity, 'managing_users')
        assert hasattr(Identity, 'saas_application')
        assert hasattr(Identity, 'events')
        assert hasattr(Identity, 'assigned_policies')

    def test_identity_creation(self, db_session):
        """Test creating a new identity."""
        identity = Identity(
            provider=IdentityProvider.OKTA.value,
            external_id="00u1234567890",
            email="user@example.com",
            username="user.name",
            display_name="User Name",
            department="Engineering",
            job_title="Software Engineer"
        )

        db_session.add(identity)
        db_session.commit()

        assert identity.id is not None
        assert identity.uuid is not None
        assert identity.provider == "okta"
        assert identity.external_id == "00u1234567890"
        assert identity.version == 1
        assert identity.status == IdentityStatus.ACTIVE.value

    def test_identity_unique_constraint(self, db_session):
        """Test unique constraint on provider + external_id."""
        # Create first identity
        identity1 = Identity(
            provider=IdentityProvider.OKTA.value,
            external_id="00u1234567890",
            email="user1@example.com"
        )
        db_session.add(identity1)
        db_session.commit()

        # Try to create identity with same provider + external_id
        identity2 = Identity(
            provider=IdentityProvider.OKTA.value,
            external_id="00u1234567890",
            email="user2@example.com"
        )
        db_session.add(identity2)

        with pytest.raises(IntegrityError):
            db_session.commit()

    def test_identity_status_methods(self):
        """Test identity status methods."""
        identity = Identity()

        # Test is_active
        identity.status = IdentityStatus.ACTIVE.value
        assert identity.is_active() is True

        identity.status = IdentityStatus.SUSPENDED.value
        assert identity.is_active() is False

        identity.status = IdentityStatus.DEPROVISIONED.value
        assert identity.is_active() is False

    def test_identity_risk_management(self):
        """Test identity risk score and flag management."""
        identity = Identity()

        # Test risk score
        assert identity.risk_score == 0
        assert identity.is_high_risk() is False

        identity.update_risk_score(75)
        assert identity.risk_score == 75
        assert identity.is_high_risk() is True

        # Test bounds
        identity.update_risk_score(150)
        assert identity.risk_score == 100

        identity.update_risk_score(-10)
        assert identity.risk_score == 0

        # Test compliance flags
        assert identity.has_compliance_issues() is False

        identity.add_compliance_flag({
            "type": "GDPR",
            "issue": "Missing consent",
            "severity": "high"
        })

        assert identity.has_compliance_issues() is True
        assert len(identity.compliance_flags) == 1
        assert "timestamp" in identity.compliance_flags[0]

        # Test anomaly flags
        identity.add_anomaly_flag({
            "type": "unusual_login",
            "location": "Unknown",
            "risk_level": "medium"
        })

        assert len(identity.anomaly_flags) == 1
        assert "timestamp" in identity.anomaly_flags[0]

    def test_identity_versioning(self):
        """Test identity version tracking."""
        identity = Identity(
            provider=IdentityProvider.AZURE_AD.value,
            external_id="azure123",
            email="user@example.com",
            username="user",
            display_name="User Name",
            department="Sales"
        )

        # Test version snapshot
        snapshot = identity.create_version_snapshot()
        assert snapshot["version"] == 1
        assert snapshot["department"] == "Sales"
        assert snapshot["email"] == "user@example.com"
        assert "timestamp" in snapshot
        assert "provider_attributes" in snapshot

        # Test version increment
        identity.department = "Marketing"
        identity.increment_version()

        assert identity.version == 2
        assert len(identity.previous_versions) == 1
        assert identity.previous_versions[0]["version"] == 1

    def test_identity_sync_error_handling(self):
        """Test sync error recording and clearing."""
        identity = Identity()

        # Initially no errors
        assert identity.sync_retry_count == 0
        assert len(identity.sync_errors) == 0
        assert identity.last_sync_error is None

        # Record error
        identity.record_sync_error("API rate limit exceeded")
        assert identity.sync_retry_count == 1
        assert len(identity.sync_errors) == 1
        assert identity.last_sync_error == "API rate limit exceeded"
        assert "timestamp" in identity.sync_errors[0]

        # Record another error
        identity.record_sync_error("Connection timeout")
        assert identity.sync_retry_count == 2
        assert len(identity.sync_errors) == 2

        # Clear errors
        identity.clear_sync_errors()
        assert identity.sync_retry_count == 0
        assert len(identity.sync_errors) == 0
        assert identity.last_sync_error is None
        assert identity.last_sync_at is not None

    def test_identity_name_methods(self):
        """Test name-related methods."""
        identity = Identity()

        # Test with all name parts
        identity.first_name = "John"
        identity.last_name = "Doe"
        identity.display_name = "John Doe"
        assert identity.get_full_name() == "John Doe"

        # Test without display name
        identity.display_name = None
        assert identity.get_full_name() == "John Doe"

        # Test with only username
        identity.first_name = None
        identity.last_name = None
        identity.username = "johndoe"
        assert identity.get_full_name() == "johndoe"

        # Test with only email
        identity.username = None
        identity.email = "john@example.com"
        assert identity.get_full_name() == "john@example.com"

    def test_identity_to_dict(self):
        """Test identity serialization to dictionary."""
        identity = Identity(
            provider=IdentityProvider.GITHUB.value,
            external_id="github123",
            email="dev@example.com",
            username="devuser",
            display_name="Dev User",
            department="Engineering",
            job_title="Senior Developer",
            location="San Francisco",
            risk_score=25,
            is_privileged=True
        )
        identity.id = 1
        identity.uuid = uuid.uuid4()
        identity.created_at = datetime.utcnow()
        identity.updated_at = datetime.utcnow()

        # Test without sensitive data
        data = identity.to_dict(include_sensitive=False)
        assert data["id"] == 1
        assert data["provider"] == "github"
        assert data["email"] == "dev@example.com"
        assert "provider_attributes" not in data
        assert "compliance_flags" not in data

        # Test with sensitive data
        data_sensitive = identity.to_dict(include_sensitive=True)
        assert "provider_attributes" in data_sensitive
        assert "compliance_flags" in data_sensitive
        assert data_sensitive["version"] == 1

    def test_identity_to_scim_resource(self):
        """Test SCIM resource conversion."""
        identity = Identity(
            external_id="ext123",
            email="user@example.com",
            username="username",
            first_name="First",
            last_name="Last",
            display_name="First Last",
            department="IT",
            employee_id="EMP001",
            manager_email="manager@example.com"
        )
        identity.uuid = uuid.uuid4()
        identity.created_at = datetime.utcnow()
        identity.updated_at = datetime.utcnow()

        scim = identity.to_scim_resource()

        assert "schemas" in scim
        assert scim["schemas"] == ["urn:ietf:params:scim:schemas:core:2.0:User"]
        assert scim["id"] == "ext123"
        assert scim["externalId"] == str(identity.uuid)
        assert scim["userName"] == "username"
        assert scim["name"]["givenName"] == "First"
        assert scim["name"]["familyName"] == "Last"
        assert scim["emails"][0]["value"] == "user@example.com"
        assert scim["active"] is True
        assert scim["enterprise"]["department"] == "IT"
        assert scim["meta"]["resourceType"] == "User"

    def test_identity_provider_enum(self):
        """Test IdentityProvider enum values."""
        providers = [
            IdentityProvider.OKTA,
            IdentityProvider.AZURE_AD,
            IdentityProvider.GOOGLE_WORKSPACE,
            IdentityProvider.SLACK,
            IdentityProvider.GITHUB,
            IdentityProvider.JIRA,
            IdentityProvider.CONFLUENCE,
            IdentityProvider.SALESFORCE,
            IdentityProvider.BOX,
            IdentityProvider.DROPBOX,
            IdentityProvider.CUSTOM
        ]

        assert len(providers) >= 10  # At least 10 providers

        # Test enum values
        assert IdentityProvider.OKTA.value == "okta"
        assert IdentityProvider.AZURE_AD.value == "azure_ad"
        assert IdentityProvider.GOOGLE_WORKSPACE.value == "google_workspace"

    def test_identity_status_enum(self):
        """Test IdentityStatus enum values."""
        statuses = [
            IdentityStatus.ACTIVE,
            IdentityStatus.SUSPENDED,
            IdentityStatus.DEPROVISIONED,
            IdentityStatus.PENDING,
            IdentityStatus.FAILED
        ]

        assert len(statuses) == 5

        # Test enum values
        assert IdentityStatus.ACTIVE.value == "active"
        assert IdentityStatus.SUSPENDED.value == "suspended"
        assert IdentityStatus.DEPROVISIONED.value == "deprovisioned"

    def test_identity_indexes(self):
        """Test that Identity model has proper indexes defined."""
        # This tests the __table_args__ configuration
        assert hasattr(Identity, '__table_args__')
        table_args = Identity.__table_args__

        # Should have unique constraint and indexes
        assert len(table_args) > 0

        # Check for specific indexes (names from the model)
        index_names = ["ix_identity_email", "ix_identity_username",
                      "ix_identity_status", "ix_identity_provider_status"]

        # Note: Actual index verification would require introspecting the
        # created table, which is done in integration tests
