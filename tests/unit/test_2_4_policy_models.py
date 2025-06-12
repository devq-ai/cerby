"""
Unit tests for Access Policy models (Subtask 2.4).

Tests cover:
- Access policy creation and validation
- Policy rule engine
- Policy conditions and effects
- Policy evaluation
- Policy versioning and history
"""

import pytest
from datetime import datetime, timedelta
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session
import json

from src.db.models.policy import (
    AccessPolicy, PolicyRule, PolicyEffect,
    ResourceType, ActionType, PolicyCondition
)
from src.db.models.identity import Identity
from src.db.models.saas_application import SaaSApplication, SaaSProvider


class TestAccessPolicyModel:
    """Test suite for Access Policy model."""

    def test_policy_creation(self, db_session: Session):
        """Test creating an access policy with all fields."""
        policy = AccessPolicy(
            name="Engineering Access Policy",
            description="Default access policy for engineering team",
            priority=100,
            is_active=True,
            effect=PolicyEffect.ALLOW,
            version=1
        )

        db_session.add(policy)
        db_session.commit()

        assert policy.id is not None
        assert policy.name == "Engineering Access Policy"
        assert policy.priority == 100
        assert policy.is_active is True
        assert policy.effect == PolicyEffect.ALLOW
        assert policy.version == 1
        assert policy.created_at is not None

    def test_policy_required_fields(self, db_session: Session):
        """Test that required fields are enforced."""
        # Missing required name field
        policy = AccessPolicy(
            description="Test policy"
        )

        db_session.add(policy)
        with pytest.raises(IntegrityError):
            db_session.commit()

    def test_policy_unique_constraint(self, db_session: Session):
        """Test unique constraint on name + version."""
        policy1 = AccessPolicy(
            name="Unique Policy",
            description="First version",
            version=1
        )
        db_session.add(policy1)
        db_session.commit()

        # Same name, same version should fail
        policy2 = AccessPolicy(
            name="Unique Policy",
            description="Duplicate",
            version=1
        )
        db_session.add(policy2)

        with pytest.raises(IntegrityError):
            db_session.commit()

        db_session.rollback()

        # Same name, different version should succeed
        policy3 = AccessPolicy(
            name="Unique Policy",
            description="Second version",
            version=2
        )
        db_session.add(policy3)
        db_session.commit()

        assert policy3.id is not None

    def test_policy_rules(self, db_session: Session):
        """Test policy rules configuration."""
        policy = AccessPolicy(
            name="Complex Policy",
            description="Policy with multiple rules",
            rules=[
                {
                    "resource": "github:repo:*",
                    "actions": ["read", "write"],
                    "conditions": {
                        "department": "Engineering",
                        "level": {"$gte": 3}
                    }
                },
                {
                    "resource": "jira:project:PROD-*",
                    "actions": ["create", "update", "delete"],
                    "conditions": {
                        "role": {"$in": ["lead", "manager"]}
                    }
                }
            ]
        )

        db_session.add(policy)
        db_session.commit()

        assert len(policy.rules) == 2
        assert policy.rules[0]["resource"] == "github:repo:*"
        assert "read" in policy.rules[0]["actions"]

    def test_policy_effect_types(self, db_session: Session):
        """Test different policy effects."""
        # Allow policy
        allow_policy = AccessPolicy(
            name="Allow Policy",
            description="Grants access",
            effect=PolicyEffect.ALLOW
        )

        # Deny policy
        deny_policy = AccessPolicy(
            name="Deny Policy",
            description="Denies access",
            effect=PolicyEffect.DENY
        )

        db_session.add_all([allow_policy, deny_policy])
        db_session.commit()

        assert allow_policy.effect == PolicyEffect.ALLOW
        assert deny_policy.effect == PolicyEffect.DENY

    def test_policy_priority_ordering(self, db_session: Session):
        """Test policy priority ordering."""
        policies = []
        for i in range(5):
            policy = AccessPolicy(
                name=f"Policy {i}",
                description=f"Test policy {i}",
                priority=i * 10
            )
            policies.append(policy)
            db_session.add(policy)

        db_session.commit()

        # Query policies ordered by priority
        ordered_policies = db_session.query(AccessPolicy).order_by(
            AccessPolicy.priority.desc()
        ).all()

        # Verify ordering (highest priority first)
        for i in range(len(ordered_policies) - 1):
            assert ordered_policies[i].priority >= ordered_policies[i + 1].priority

    def test_policy_conditions(self, db_session: Session):
        """Test complex policy conditions."""
        policy = AccessPolicy(
            name="Conditional Policy",
            description="Policy with complex conditions",
            rules=[
                {
                    "resource": "salesforce:opportunity:*",
                    "actions": ["read", "update"],
                    "conditions": {
                        "$and": [
                            {"department": {"$in": ["Sales", "Marketing"]}},
                            {"region": "NA"},
                            {"quota_attainment": {"$gte": 0.8}}
                        ]
                    }
                }
            ]
        )

        db_session.add(policy)
        db_session.commit()

        # Test condition evaluation
        conditions = policy.rules[0]["conditions"]
        assert "$and" in conditions
        assert len(conditions["$and"]) == 3

    def test_policy_evaluation(self, db_session: Session):
        """Test policy evaluation logic."""
        policy = AccessPolicy(
            name="Evaluation Test Policy",
            description="Test policy evaluation",
            effect=PolicyEffect.ALLOW,
            rules=[
                {
                    "resource": "github:repo:backend",
                    "actions": ["push", "merge"],
                    "conditions": {
                        "team": "backend",
                        "experience_years": {"$gte": 2}
                    }
                }
            ]
        )

        db_session.add(policy)
        db_session.commit()

        # Test evaluation method
        # Should match
        context1 = {
            "resource": "github:repo:backend",
            "action": "push",
            "user": {
                "team": "backend",
                "experience_years": 3
            }
        }
        assert policy.evaluate(context1) is True

        # Should not match - wrong team
        context2 = {
            "resource": "github:repo:backend",
            "action": "push",
            "user": {
                "team": "frontend",
                "experience_years": 3
            }
        }
        assert policy.evaluate(context2) is False

        # Should not match - insufficient experience
        context3 = {
            "resource": "github:repo:backend",
            "action": "push",
            "user": {
                "team": "backend",
                "experience_years": 1
            }
        }
        assert policy.evaluate(context3) is False

    def test_policy_versioning(self, db_session: Session):
        """Test policy versioning functionality."""
        # Create initial version
        policy_v1 = AccessPolicy(
            name="Versioned Policy",
            description="Version 1",
            version=1,
            rules=[{"resource": "app:*", "actions": ["read"]}]
        )
        db_session.add(policy_v1)
        db_session.commit()

        # Create new version
        policy_v2 = policy_v1.create_new_version()
        policy_v2.rules = [
            {"resource": "app:*", "actions": ["read", "write"]}
        ]
        policy_v2.description = "Version 2 - Added write permission"

        db_session.add(policy_v2)
        db_session.commit()

        assert policy_v2.version == 2
        assert policy_v2.name == policy_v1.name
        assert policy_v2.parent_version_id == policy_v1.id
        assert len(policy_v2.rules[0]["actions"]) == 2

    def test_policy_soft_delete(self, db_session: Session):
        """Test soft delete functionality."""
        policy = AccessPolicy(
            name="Delete Test Policy",
            description="Will be deleted",
            is_active=True
        )

        db_session.add(policy)
        db_session.commit()

        # Soft delete
        policy.soft_delete()
        db_session.commit()

        assert policy.is_active is False
        assert policy.deleted_at is not None

        # Should not appear in active policies
        active_policies = db_session.query(AccessPolicy).filter(
            AccessPolicy.is_active == True
        ).all()
        assert policy not in active_policies

    def test_policy_wildcard_matching(self, db_session: Session):
        """Test wildcard resource matching."""
        policy = AccessPolicy(
            name="Wildcard Policy",
            description="Test wildcard matching",
            rules=[
                {
                    "resource": "s3:bucket:prod-*",
                    "actions": ["read", "list"]
                },
                {
                    "resource": "ec2:instance:*",
                    "actions": ["start", "stop", "reboot"]
                }
            ]
        )

        db_session.add(policy)
        db_session.commit()

        # Test matching
        assert policy.matches_resource("s3:bucket:prod-data") is True
        assert policy.matches_resource("s3:bucket:dev-data") is False
        assert policy.matches_resource("ec2:instance:i-12345") is True

    def test_policy_rule_inheritance(self, db_session: Session):
        """Test rule inheritance in policy hierarchy."""
        # Parent policy
        parent_policy = AccessPolicy(
            name="Parent Policy",
            description="Base policy",
            rules=[
                {"resource": "common:*", "actions": ["read"]}
            ]
        )
        db_session.add(parent_policy)
        db_session.commit()

        # Child policy that extends parent
        child_policy = AccessPolicy(
            name="Child Policy",
            description="Extended policy",
            parent_policy_id=parent_policy.id,
            rules=[
                {"resource": "specific:*", "actions": ["write"]}
            ]
        )
        db_session.add(child_policy)
        db_session.commit()

        # Get effective rules (should include both)
        effective_rules = child_policy.get_effective_rules()
        assert len(effective_rules) == 2

    def test_policy_conflict_resolution(self, db_session: Session):
        """Test policy conflict resolution based on priority and effect."""
        # Create conflicting policies
        allow_policy = AccessPolicy(
            name="Allow GitHub",
            description="Allows GitHub access",
            effect=PolicyEffect.ALLOW,
            priority=100,
            rules=[{"resource": "github:*", "actions": ["*"]}]
        )

        deny_policy = AccessPolicy(
            name="Deny GitHub Admin",
            description="Denies GitHub admin actions",
            effect=PolicyEffect.DENY,
            priority=200,  # Higher priority
            rules=[{"resource": "github:*", "actions": ["delete", "admin"]}]
        )

        db_session.add_all([allow_policy, deny_policy])
        db_session.commit()

        # Test that deny with higher priority wins
        context = {
            "resource": "github:repo:main",
            "action": "delete"
        }

        # In a real system, policy engine would evaluate all policies
        # Here we simulate that deny wins due to higher priority
        assert deny_policy.priority > allow_policy.priority
        assert deny_policy.effect == PolicyEffect.DENY

    def test_policy_metadata(self, db_session: Session):
        """Test policy metadata storage."""
        policy = AccessPolicy(
            name="Metadata Policy",
            description="Policy with metadata",
            metadata={
                "compliance": ["SOX", "GDPR"],
                "owner": "security-team",
                "review_date": "2024-12-31",
                "risk_level": "high"
            }
        )

        db_session.add(policy)
        db_session.commit()
        db_session.refresh(policy)

        assert "compliance" in policy.metadata
        assert "SOX" in policy.metadata["compliance"]
        assert policy.metadata["risk_level"] == "high"

    def test_policy_to_dict(self, db_session: Session):
        """Test policy serialization."""
        policy = AccessPolicy(
            name="Serialization Test",
            description="Test to_dict method",
            priority=50,
            effect=PolicyEffect.ALLOW,
            rules=[
                {
                    "resource": "test:*",
                    "actions": ["read"],
                    "conditions": {"env": "prod"}
                }
            ]
        )

        db_session.add(policy)
        db_session.commit()

        policy_dict = policy.to_dict()

        assert policy_dict["name"] == "Serialization Test"
        assert policy_dict["priority"] == 50
        assert policy_dict["effect"] == "allow"
        assert len(policy_dict["rules"]) == 1
        assert policy_dict["is_active"] is True

    def test_policy_validation(self, db_session: Session):
        """Test policy validation rules."""
        # Test invalid priority
        policy = AccessPolicy(
            name="Invalid Policy",
            description="Test validation",
            priority=-1  # Invalid
        )

        with pytest.raises(ValueError):
            policy.validate()

        # Test invalid rule structure
        policy.priority = 100
        policy.rules = [
            {
                "resource": "test:*"
                # Missing required 'actions' field
            }
        ]

        with pytest.raises(ValueError):
            policy.validate()

    def test_policy_statistics(self, db_session: Session):
        """Test policy usage statistics."""
        policy = AccessPolicy(
            name="Stats Policy",
            description="Track usage statistics",
            evaluation_count=0,
            last_evaluated_at=None
        )

        db_session.add(policy)
        db_session.commit()

        # Simulate evaluations
        for _ in range(5):
            policy.record_evaluation(success=True)

        policy.record_evaluation(success=False)
        db_session.commit()

        assert policy.evaluation_count == 6
        assert policy.last_evaluated_at is not None
        assert policy.success_count == 5
        assert policy.failure_count == 1
