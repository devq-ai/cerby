"""
Database models package for Cerby Identity Automation Platform.

This package contains all SQLAlchemy models for the identity management system,
including users, identities, policies, SaaS applications, and audit logs.
"""

from src.db.models.user import User
from src.db.models.identity import Identity, IdentityProvider
from src.db.models.saas_app import SaaSApplication, SaaSAppType
from src.db.models.policy import AccessPolicy, PolicyRule, PolicyVersion
from src.db.models.audit import AuditLog, IdentityEvent
from src.db.models.associations import UserIdentity, PolicyAssignment

__all__ = [
    # User models
    "User",

    # Identity models
    "Identity",
    "IdentityProvider",

    # SaaS Application models
    "SaaSApplication",
    "SaaSAppType",

    # Policy models
    "AccessPolicy",
    "PolicyRule",
    "PolicyVersion",

    # Audit models
    "AuditLog",
    "IdentityEvent",

    # Association tables
    "UserIdentity",
    "PolicyAssignment",
]
