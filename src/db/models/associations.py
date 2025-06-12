"""
Association tables for many-to-many relationships in Cerby Identity Automation Platform.

This module defines the association tables that link entities in many-to-many
relationships, such as users to identities, policies to identities, etc.
"""

from datetime import datetime
from sqlalchemy import Table, Column, Integer, String, DateTime, Boolean, Text, ForeignKey, UniqueConstraint, Index
from src.db.database import Base


# User-Identity association table
user_identities = Table(
    "user_identities",
    Base.metadata,
    Column("user_id", Integer, ForeignKey("users.id", ondelete="CASCADE"), primary_key=True),
    Column("identity_id", Integer, ForeignKey("identities.id", ondelete="CASCADE"), primary_key=True),
    Column("assigned_at", DateTime, default=datetime.utcnow, nullable=False),
    Column("assigned_by", Integer, ForeignKey("users.id"), nullable=True),
    Column("role", String(50), default="viewer", nullable=False),  # "viewer", "editor", "admin"
    Column("notes", Text, nullable=True),
    UniqueConstraint("user_id", "identity_id", name="uq_user_identity"),
    Index("ix_user_identities_user", "user_id"),
    Index("ix_user_identities_identity", "identity_id"),
)


# Policy-Identity assignment table
policy_assignments = Table(
    "policy_assignments",
    Base.metadata,
    Column("policy_id", Integer, ForeignKey("access_policies.id", ondelete="CASCADE"), primary_key=True),
    Column("identity_id", Integer, ForeignKey("identities.id", ondelete="CASCADE"), primary_key=True),
    Column("assigned_at", DateTime, default=datetime.utcnow, nullable=False),
    Column("assigned_by", Integer, ForeignKey("users.id"), nullable=True),
    Column("effective_from", DateTime, default=datetime.utcnow, nullable=False),
    Column("effective_until", DateTime, nullable=True),
    Column("is_active", Boolean, default=True, nullable=False),
    Column("priority_override", Integer, nullable=True),  # Override policy priority for this assignment
    Column("assignment_reason", Text, nullable=True),
    Column("compliance_tags", String(255), nullable=True),  # Comma-separated compliance tags
    UniqueConstraint("policy_id", "identity_id", name="uq_policy_identity"),
    Index("ix_policy_assignments_policy", "policy_id"),
    Index("ix_policy_assignments_identity", "identity_id"),
    Index("ix_policy_assignments_active", "is_active"),
    Index("ix_policy_assignments_effective", "effective_from", "effective_until"),
)


# Policy-SaaSApplication association table
policy_saas_apps = Table(
    "policy_saas_apps",
    Base.metadata,
    Column("policy_id", Integer, ForeignKey("access_policies.id", ondelete="CASCADE"), primary_key=True),
    Column("saas_app_id", Integer, ForeignKey("saas_applications.id", ondelete="CASCADE"), primary_key=True),
    Column("linked_at", DateTime, default=datetime.utcnow, nullable=False),
    Column("linked_by", Integer, ForeignKey("users.id"), nullable=True),
    Column("scope", String(255), nullable=True),  # Specific scope within the app (e.g., "repos:read")
    Column("environment", String(50), nullable=True),  # "production", "staging", "development"
    Column("is_active", Boolean, default=True, nullable=False),
    UniqueConstraint("policy_id", "saas_app_id", name="uq_policy_saas_app"),
    Index("ix_policy_saas_apps_policy", "policy_id"),
    Index("ix_policy_saas_apps_app", "saas_app_id"),
    Index("ix_policy_saas_apps_active", "is_active"),
)


# Identity-Group association table (for future group management)
identity_groups = Table(
    "identity_groups",
    Base.metadata,
    Column("identity_id", Integer, ForeignKey("identities.id", ondelete="CASCADE"), primary_key=True),
    Column("group_id", Integer, ForeignKey("groups.id", ondelete="CASCADE"), primary_key=True),
    Column("joined_at", DateTime, default=datetime.utcnow, nullable=False),
    Column("joined_by", Integer, ForeignKey("users.id"), nullable=True),
    Column("is_primary", Boolean, default=False, nullable=False),  # Primary group for the identity
    Column("expiration_date", DateTime, nullable=True),  # For temporary group memberships
    UniqueConstraint("identity_id", "group_id", name="uq_identity_group"),
    Index("ix_identity_groups_identity", "identity_id"),
    Index("ix_identity_groups_group", "group_id"),
)


# Policy-Compliance association table
policy_compliance = Table(
    "policy_compliance",
    Base.metadata,
    Column("policy_id", Integer, ForeignKey("access_policies.id", ondelete="CASCADE"), primary_key=True),
    Column("compliance_id", Integer, ForeignKey("compliance_requirements.id", ondelete="CASCADE"), primary_key=True),
    Column("mapped_at", DateTime, default=datetime.utcnow, nullable=False),
    Column("mapped_by", Integer, ForeignKey("users.id"), nullable=True),
    Column("compliance_status", String(20), default="compliant", nullable=False),  # "compliant", "non_compliant", "partial"
    Column("last_validated_at", DateTime, nullable=True),
    Column("validation_notes", Text, nullable=True),
    UniqueConstraint("policy_id", "compliance_id", name="uq_policy_compliance"),
    Index("ix_policy_compliance_policy", "policy_id"),
    Index("ix_policy_compliance_requirement", "compliance_id"),
    Index("ix_policy_compliance_status", "compliance_status"),
)


# User-Role association table (for RBAC)
user_roles = Table(
    "user_roles",
    Base.metadata,
    Column("user_id", Integer, ForeignKey("users.id", ondelete="CASCADE"), primary_key=True),
    Column("role_id", Integer, ForeignKey("roles.id", ondelete="CASCADE"), primary_key=True),
    Column("assigned_at", DateTime, default=datetime.utcnow, nullable=False),
    Column("assigned_by", Integer, ForeignKey("users.id"), nullable=True),
    Column("valid_from", DateTime, default=datetime.utcnow, nullable=False),
    Column("valid_until", DateTime, nullable=True),
    Column("is_active", Boolean, default=True, nullable=False),
    UniqueConstraint("user_id", "role_id", name="uq_user_role"),
    Index("ix_user_roles_user", "user_id"),
    Index("ix_user_roles_role", "role_id"),
    Index("ix_user_roles_active", "is_active"),
)


# Identity-Identity relationship table (for manager relationships, service accounts, etc.)
identity_relationships = Table(
    "identity_relationships",
    Base.metadata,
    Column("parent_identity_id", Integer, ForeignKey("identities.id", ondelete="CASCADE"), primary_key=True),
    Column("child_identity_id", Integer, ForeignKey("identities.id", ondelete="CASCADE"), primary_key=True),
    Column("relationship_type", String(50), primary_key=True),  # "manages", "owns_service_account", "delegates_to"
    Column("established_at", DateTime, default=datetime.utcnow, nullable=False),
    Column("established_by", Integer, ForeignKey("users.id"), nullable=True),
    Column("valid_until", DateTime, nullable=True),
    Column("metadata", Text, nullable=True),  # JSON string for additional relationship data
    UniqueConstraint("parent_identity_id", "child_identity_id", "relationship_type", name="uq_identity_relationship"),
    Index("ix_identity_relationships_parent", "parent_identity_id"),
    Index("ix_identity_relationships_child", "child_identity_id"),
    Index("ix_identity_relationships_type", "relationship_type"),
)


# SaaS App dependencies table (for apps that depend on other apps)
saas_app_dependencies = Table(
    "saas_app_dependencies",
    Base.metadata,
    Column("app_id", Integer, ForeignKey("saas_applications.id", ondelete="CASCADE"), primary_key=True),
    Column("depends_on_id", Integer, ForeignKey("saas_applications.id", ondelete="CASCADE"), primary_key=True),
    Column("dependency_type", String(50), nullable=False),  # "authentication", "data_source", "integration"
    Column("is_required", Boolean, default=True, nullable=False),
    Column("created_at", DateTime, default=datetime.utcnow, nullable=False),
    Column("created_by", Integer, ForeignKey("users.id"), nullable=True),
    UniqueConstraint("app_id", "depends_on_id", name="uq_app_dependency"),
    Index("ix_saas_app_dependencies_app", "app_id"),
    Index("ix_saas_app_dependencies_depends", "depends_on_id"),
)


# Export association models for easier access
class UserIdentity:
    """Helper class for user-identity associations."""
    __table__ = user_identities


class PolicyAssignment:
    """Helper class for policy-identity assignments."""
    __table__ = policy_assignments


class PolicySaaSApp:
    """Helper class for policy-SaaS app associations."""
    __table__ = policy_saas_apps


class IdentityGroup:
    """Helper class for identity-group associations."""
    __table__ = identity_groups


class PolicyCompliance:
    """Helper class for policy-compliance associations."""
    __table__ = policy_compliance


class UserRole:
    """Helper class for user-role associations."""
    __table__ = user_roles


class IdentityRelationship:
    """Helper class for identity relationships."""
    __table__ = identity_relationships


class SaaSAppDependency:
    """Helper class for SaaS app dependencies."""
    __table__ = saas_app_dependencies


# Export all association tables
__all__ = [
    "user_identities",
    "policy_assignments",
    "policy_saas_apps",
    "identity_groups",
    "policy_compliance",
    "user_roles",
    "identity_relationships",
    "saas_app_dependencies",
    "UserIdentity",
    "PolicyAssignment",
    "PolicySaaSApp",
    "IdentityGroup",
    "PolicyCompliance",
    "UserRole",
    "IdentityRelationship",
    "SaaSAppDependency",
]
