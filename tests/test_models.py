"""
Test suite for database models in Cerby Identity Automation Platform.

This module tests all database models including User, Identity, SaaSApplication,
AccessPolicy, and audit models.
"""

import pytest
from datetime import datetime, timedelta
from sqlalchemy.exc import IntegrityError

from src.db.models.user import User
from src.db.models.identity import Identity, IdentityProvider, IdentityStatus
from src.db.models.saas_app import SaaSApplication, SaaSAppType, AuthType, SyncMethod
from src.db.models.policy import AccessPolicy, PolicyRule, PolicyVersion, PolicyType, PolicyEffect
from src.db.models.audit import AuditLog, IdentityEvent, AuditAction, EventType


class TestUserModel:
    """Test User model functionality."""

    def test_create_user(self, db_session):
        """Test creating a new user."""
        user = User(
            email="test@example.com",
            username="testuser",
            full_name="Test User"
        )
        user.set_password("securepassword123")

        db_session.add(user)
        db_session.commit()

        assert user.id is not None
        assert user.email == "test@example.com"
        assert user.username == "testuser"
        assert user.verify_password("securepassword123")
        assert not user.verify_password("wrongpassword")

    def test_user_unique_constraints(self, db_session):
        """Test unique constraints on email and username."""
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

    def test_user_permissions(self, db_session):
        """Test user permission methods."""
        user = User(
            email="perms@example.com",
            username="permsuser",
            is_superuser=False,
            preferences={
                "can_manage_identities": True,
                "can_manage_policies": False
            }
        )
        user.set_password("password123")
        db_session.add(user)
        db_session.commit()

        assert user.can_manage_identities()
        assert not user.can_manage_policies()
        assert user.can_view_analytics()  # Default True

        # Test superuser permissions
        user.is_superuser = True
        assert user.can_manage_policies()

    def test_user_lockout(self, db_session):
        """Test user lockout functionality."""
        user = User(email="lockout@example.com", username="lockoutuser")
        user.set_password("password123")
        db_session.add(user)
        db_session.commit()

        # Test failed login attempts
        assert not user.is_locked()

        for _ in range(5):
            user.increment_failed_login()

        assert user.is_locked()
        assert user.locked_until is not None

        # Test reset
        user.reset_failed_login()
        assert not user.is_locked()
        assert user.failed_login_attempts == 0

    def test_user_api_key(self, db_session):
        """Test API key generation and verification."""
        user = User(email="api@example.com", username="apiuser")
        user.set_password("password123")
        db_session.add(user)
        db_session.commit()

        # Generate API key
        api_key = user.generate_api_key()
        assert api_key.startswith("cerby_")
        assert user.api_key_hash is not None

        # Verify API key
        assert user.verify_api_key(api_key)
        assert not user.verify_api_key("wrong_key")


class TestIdentityModel:
    """Test Identity model functionality."""

    def test_create_identity(self, db_session):
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
        assert identity.is_active()
        assert identity.version == 1

    def test_identity_unique_constraint(self, db_session):
        """Test unique constraint on provider + external_id."""
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

    def test_identity_risk_management(self, db_session):
        """Test identity risk score and flags."""
        identity = Identity(
            provider=IdentityProvider.GITHUB.value,
            external_id="github123",
            email="dev@example.com"
        )
        db_session.add(identity)
        db_session.commit()

        # Test risk score
        assert not identity.is_high_risk()
        identity.update_risk_score(75)
        assert identity.risk_score == 75
        assert identity.is_high_risk()

        # Test compliance flags
        assert not identity.has_compliance_issues()
        identity.add_compliance_flag({
            "type": "GDPR",
            "issue": "Missing consent",
            "severity": "high"
        })
        assert identity.has_compliance_issues()
        assert len(identity.compliance_flags) == 1

    def test_identity_versioning(self, db_session):
        """Test identity version tracking."""
        identity = Identity(
            provider=IdentityProvider.AZURE_AD.value,
            external_id="azure123",
            email="user@example.com",
            department="Sales"
        )
        db_session.add(identity)
        db_session.commit()

        # Create version snapshot
        snapshot = identity.create_version_snapshot()
        assert snapshot["version"] == 1
        assert snapshot["department"] == "Sales"

        # Update and increment version
        identity.department = "Marketing"
        identity.increment_version()

        assert identity.version == 2
        assert len(identity.previous_versions) == 1

    def test_identity_sync_errors(self, db_session):
        """Test sync error handling."""
        identity = Identity(
            provider=IdentityProvider.SLACK.value,
            external_id="slack123",
            email="user@example.com"
        )
        db_session.add(identity)
        db_session.commit()

        # Record sync error
        identity.record_sync_error("API rate limit exceeded")
        assert identity.sync_retry_count == 1
        assert len(identity.sync_errors) == 1

        # Clear sync errors
        identity.clear_sync_errors()
        assert identity.sync_retry_count == 0
        assert len(identity.sync_errors) == 0


class TestSaaSApplicationModel:
    """Test SaaSApplication model functionality."""

    def test_create_saas_app(self, db_session):
        """Test creating a new SaaS application."""
        app = SaaSApplication(
            name="GitHub Enterprise",
            provider="github",
            app_type=SaaSAppType.DEVELOPMENT.value,
            auth_type=AuthType.OAUTH2.value,
            sync_method=SyncMethod.WEBHOOK.value,
            api_endpoint="https://api.github.com",
            tenant_id="my-org"
        )

        db_session.add(app)
        db_session.commit()

        assert app.id is not None
        assert app.requires_oauth()
        assert not app.supports_scim()
        assert app.is_sync_due()

    def test_saas_app_sync_tracking(self, db_session):
        """Test sync statistics tracking."""
        app = SaaSApplication(
            name="Okta",
            provider="okta",
            app_type=SaaSAppType.IDENTITY_PROVIDER.value,
            sync_method=SyncMethod.SCIM.value
        )
        db_session.add(app)
        db_session.commit()

        # Record successful sync
        app.record_sync_success(identities_count=100, duration_seconds=45)
        assert app.total_identities_synced == 100
        assert app.average_sync_duration_seconds == 45
        assert app.health_status == "healthy"

        # Record sync error
        app.record_sync_error("Connection timeout")
        assert app.total_sync_errors == 1
        assert app.health_status == "unhealthy"
        assert len(app.health_check_errors) == 1

    def test_saas_app_field_mapping(self, db_session):
        """Test field mapping functionality."""
        app = SaaSApplication(
            name="Salesforce",
            provider="salesforce",
            field_mappings={
                "Name": "display_name",
                "Email": "email",
                "Department__c": "department"
            }
        )
        db_session.add(app)
        db_session.commit()

        # Test mapping
        external_data = {
            "Name": "John Doe",
            "Email": "john@example.com",
            "Department__c": "Sales",
            "CustomField": "Value"
        }

        mapped_data = app.map_external_data(external_data)
        assert mapped_data["display_name"] == "John Doe"
        assert mapped_data["email"] == "john@example.com"
        assert mapped_data["department"] == "Sales"
        assert "_unmapped" in mapped_data
        assert mapped_data["_unmapped"]["CustomField"] == "Value"


class TestAccessPolicyModel:
    """Test AccessPolicy and PolicyRule models."""

    def test_create_policy(self, db_session):
        """Test creating a new access policy."""
        policy = AccessPolicy(
            name="Engineering Access Policy",
            description="Default policy for engineering team",
            policy_type=PolicyType.ROLE_BASED.value,
            priority=100
        )

        # Add rules
        rule1 = PolicyRule(
            resource="github:repo:*",
            action="read",
            effect=PolicyEffect.ALLOW.value,
            conditions={"user.department": "Engineering"},
            order=0
        )

        rule2 = PolicyRule(
            resource="jira:project:ENG",
            action="write",
            effect=PolicyEffect.ALLOW.value,
            conditions={"user.department": "Engineering"},
            order=1
        )

        policy.rules.append(rule1)
        policy.rules.append(rule2)

        db_session.add(policy)
        db_session.commit()

        assert policy.id is not None
        assert policy.uuid is not None
        assert len(policy.rules) == 2
        assert policy.is_effective()

    def test_policy_evaluation(self, db_session):
        """Test policy evaluation logic."""
        policy = AccessPolicy(
            name="Test Policy",
            policy_type=PolicyType.ATTRIBUTE_BASED.value
        )

        rule = PolicyRule(
            resource="app:resource:test",
            action="read",
            effect=PolicyEffect.ALLOW.value,
            conditions={"user.role": "admin"}
        )

        policy.rules.append(rule)
        db_session.add(policy)
        db_session.commit()

        # Test evaluation
        context1 = {
            "resource": "app:resource:test",
            "action": "read",
            "user": {"role": "admin"}
        }
        assert policy.evaluate(context1) == PolicyEffect.ALLOW

        context2 = {
            "resource": "app:resource:test",
            "action": "read",
            "user": {"role": "user"}
        }
        assert policy.evaluate(context2) == PolicyEffect.DENY

    def test_policy_fitness_tracking(self, db_session):
        """Test genetic algorithm fitness tracking."""
        policy = AccessPolicy(
            name="GA Optimized Policy",
            is_ga_optimized=True,
            generation_created=25
        )
        db_session.add(policy)
        db_session.commit()

        # Update fitness scores
        policy.update_fitness_scores(
            security=0.85,
            productivity=0.75,
            compliance=0.90
        )

        assert policy.security_score == 0.85
        assert policy.productivity_score == 0.75
        assert policy.compliance_score == 0.90
        assert policy.fitness_score == pytest.approx(0.83, 0.01)

    def test_policy_versioning(self, db_session):
        """Test policy version management."""
        policy = AccessPolicy(
            name="Versioned Policy",
            version=1
        )

        rule = PolicyRule(
            resource="*",
            action="*",
            effect=PolicyEffect.DENY.value
        )
        policy.rules.append(rule)

        db_session.add(policy)
        db_session.commit()

        # Create version
        version = policy.create_version()
        db_session.add(version)
        db_session.commit()

        assert version.version == 1
        assert version.policy_id == policy.id
        assert len(version.rules_snapshot) == 1
        assert policy.version == 2


class TestAuditModels:
    """Test AuditLog and IdentityEvent models."""

    def test_create_audit_log(self, db_session):
        """Test creating audit log entries."""
        user = User(email="auditor@example.com", username="auditor")
        user.set_password("password123")
        db_session.add(user)
        db_session.commit()

        audit = AuditLog(
            action=AuditAction.IDENTITY_CREATED.value,
            user_id=user.id,
            user_email=user.email,
            entity_type="identity",
            entity_id=1,
            compliance_relevant=True,
            compliance_frameworks=["SOX", "GDPR"]
        )

        db_session.add(audit)
        db_session.commit()

        assert audit.id is not None
        assert audit.uuid is not None
        assert audit.is_high_risk_action()
        assert audit.requires_retention()

    def test_audit_log_anonymization(self, db_session):
        """Test PII anonymization in audit logs."""
        audit = AuditLog(
            action=AuditAction.USER_DELETED.value,
            user_email="sensitive@example.com",
            user_ip="192.168.1.100",
            metadata={
                "email": "user@example.com",
                "name": "John Doe",
                "phone": "555-1234"
            }
        )

        db_session.add(audit)
        db_session.commit()

        # Anonymize PII
        audit.anonymize_pii()

        assert "@anonymized" in audit.user_email
        assert audit.user_ip == "0.0.0.0"
        assert audit.metadata["email"] == "[REDACTED]"
        assert audit.metadata["name"] == "[REDACTED]"
        assert audit.metadata["phone"] == "[REDACTED]"

    def test_create_identity_event(self, db_session):
        """Test creating identity events."""
        identity = Identity(
            provider=IdentityProvider.OKTA.value,
            external_id="okta123",
            email="user@example.com"
        )
        db_session.add(identity)
        db_session.commit()

        event = IdentityEvent(
            event_type=EventType.USER_CREATED.value,
            identity_id=identity.id,
            external_id=identity.external_id,
            provider=identity.provider,
            occurred_at=datetime.utcnow(),
            event_data={
                "source": "SCIM",
                "attributes": {"department": "Engineering"}
            }
        )

        db_session.add(event)
        db_session.commit()

        assert event.id is not None
        assert event.uuid is not None
        assert not event.is_processed

    def test_event_risk_scoring(self, db_session):
        """Test event risk score calculation."""
        identity = Identity(
            provider=IdentityProvider.GITHUB.value,
            external_id="github456",
            email="dev@example.com"
        )
        db_session.add(identity)
        db_session.commit()

        # High risk event
        event1 = IdentityEvent(
            event_type=EventType.MFA_DISABLED.value,
            identity_id=identity.id,
            external_id=identity.external_id,
            provider=identity.provider,
            occurred_at=datetime.utcnow().replace(hour=3),  # Outside business hours
            anomaly_detected=True
        )

        risk_score = event1.calculate_risk_score()
        assert risk_score >= 70
        assert event1.is_high_risk()
        assert event1.requires_immediate_action()

        # Low risk event
        event2 = IdentityEvent(
            event_type=EventType.USER_UPDATED.value,
            identity_id=identity.id,
            external_id=identity.external_id,
            provider=identity.provider,
            occurred_at=datetime.utcnow().replace(hour=14),  # Business hours
            actor_type="system"
        )

        risk_score2 = event2.calculate_risk_score()
        assert risk_score2 < 50
        assert not event2.is_high_risk()


@pytest.mark.integration
class TestModelRelationships:
    """Test relationships between models."""

    def test_user_identity_relationship(self, db_session):
        """Test many-to-many relationship between users and identities."""
        user = User(email="manager@example.com", username="manager")
        user.set_password("password123")

        identity1 = Identity(
            provider=IdentityProvider.OKTA.value,
            external_id="okta789",
            email="id1@example.com"
        )

        identity2 = Identity(
            provider=IdentityProvider.AZURE_AD.value,
            external_id="azure789",
            email="id2@example.com"
        )

        # Add identities to user
        user.managed_identities.append(identity1)
        user.managed_identities.append(identity2)

        db_session.add(user)
        db_session.commit()

        # Test relationship
        assert user.managed_identities.count() == 2
        assert identity1 in user.managed_identities
        assert user in identity1.managing_users

    def test_policy_identity_assignment(self, db_session):
        """Test policy assignment to identities."""
        policy = AccessPolicy(
            name="Test Assignment Policy",
            policy_type=PolicyType.ROLE_BASED.value
        )

        identity = Identity(
            provider=IdentityProvider.GOOGLE_WORKSPACE.value,
            external_id="google123",
            email="user@example.com"
        )

        # Assign policy to identity
        policy.assigned_identities.append(identity)

        db_session.add(policy)
        db_session.add(identity)
        db_session.commit()

        # Test relationship
        assert policy.assigned_identities.count() == 1
        assert identity.assigned_policies.count() == 1
        assert identity in policy.assigned_identities.all()
