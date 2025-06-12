"""
Access Policy model for Cerby Identity Automation Platform.

This module defines the AccessPolicy, PolicyRule, and PolicyVersion models
which represent access control policies that can be optimized using genetic algorithms.
"""

from datetime import datetime
from typing import Optional, List, Dict, Any, TYPE_CHECKING
from enum import Enum
from sqlalchemy import Column, String, Boolean, DateTime, Text, JSON, Integer, Float, ForeignKey, Index, UniqueConstraint
from sqlalchemy.orm import relationship, Mapped
from sqlalchemy.dialects.postgresql import UUID
import uuid

from src.db.database import BaseModel

if TYPE_CHECKING:
    from src.db.models.identity import Identity
    from src.db.models.saas_app import SaaSApplication
    from src.db.models.audit import AuditLog


class PolicyType(str, Enum):
    """Types of access policies."""
    ROLE_BASED = "role_based"
    ATTRIBUTE_BASED = "attribute_based"
    TIME_BASED = "time_based"
    LOCATION_BASED = "location_based"
    RISK_BASED = "risk_based"
    GENETIC_OPTIMIZED = "genetic_optimized"
    CUSTOM = "custom"


class PolicyEffect(str, Enum):
    """Policy decision effects."""
    ALLOW = "allow"
    DENY = "deny"
    CONDITIONAL = "conditional"


class PolicyPriority(str, Enum):
    """Policy evaluation priority."""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class AccessPolicy(BaseModel):
    """
    AccessPolicy model representing access control policies.

    These policies can be manually created or generated/optimized by the
    genetic algorithm engine. They define who can access what resources
    under which conditions.
    """

    __tablename__ = "access_policies"
    __table_args__ = (
        UniqueConstraint('name', 'version', name='uq_policy_name_version'),
        Index('ix_policy_type', 'policy_type'),
        Index('ix_policy_priority', 'priority'),
        Index('ix_policy_active', 'is_active'),
        Index('ix_policy_ga_optimized', 'is_ga_optimized'),
    )

    # Policy identifiers
    uuid = Column(UUID(as_uuid=True), default=uuid.uuid4, unique=True, nullable=False)
    name = Column(String(255), nullable=False)
    description = Column(Text, nullable=True)
    version = Column(Integer, default=1, nullable=False)

    # Policy metadata
    policy_type = Column(String(50), default=PolicyType.CUSTOM.value, nullable=False)
    priority = Column(Integer, default=100, nullable=False)  # Lower number = higher priority
    priority_level = Column(String(20), default=PolicyPriority.MEDIUM.value, nullable=False)
    is_active = Column(Boolean, default=True, nullable=False)
    is_ga_optimized = Column(Boolean, default=False, nullable=False)

    # Genetic Algorithm metadata
    generation_created = Column(Integer, nullable=True)  # GA generation when created
    fitness_score = Column(Float, nullable=True)  # Overall fitness score
    security_score = Column(Float, nullable=True)  # Security objective score
    productivity_score = Column(Float, nullable=True)  # Productivity objective score
    compliance_score = Column(Float, nullable=True)  # Compliance objective score

    # GA chromosome representation
    chromosome = Column(JSON, nullable=True)  # Encoded policy for GA
    parent_policies = Column(JSON, default=list, nullable=False)  # UUIDs of parent policies

    # Policy effectiveness metrics
    total_evaluations = Column(Integer, default=0, nullable=False)
    total_allows = Column(Integer, default=0, nullable=False)
    total_denies = Column(Integer, default=0, nullable=False)
    false_positive_count = Column(Integer, default=0, nullable=False)
    false_negative_count = Column(Integer, default=0, nullable=False)
    average_decision_time_ms = Column(Float, nullable=True)

    # Compliance tags
    compliance_frameworks = Column(JSON, default=list, nullable=False)  # e.g., ["SOX", "GDPR"]
    compliance_requirements = Column(JSON, default=list, nullable=False)  # Specific requirements

    # Lifecycle dates
    effective_from = Column(DateTime, default=datetime.utcnow, nullable=False)
    effective_until = Column(DateTime, nullable=True)
    last_evaluated_at = Column(DateTime, nullable=True)
    last_modified_by = Column(Integer, ForeignKey("users.id"), nullable=True)

    # Testing and validation
    is_tested = Column(Boolean, default=False, nullable=False)
    test_coverage_percent = Column(Float, default=0.0, nullable=False)
    test_scenarios = Column(JSON, default=list, nullable=False)

    # Relationships
    rules: Mapped[List["PolicyRule"]] = relationship(
        "PolicyRule",
        back_populates="policy",
        cascade="all, delete-orphan",
        order_by="PolicyRule.order"
    )

    assigned_identities: Mapped[List["Identity"]] = relationship(
        "Identity",
        secondary="policy_assignments",
        back_populates="assigned_policies",
        lazy="dynamic"
    )

    saas_applications: Mapped[List["SaaSApplication"]] = relationship(
        "SaaSApplication",
        secondary="policy_saas_apps",
        back_populates="policies",
        lazy="dynamic"
    )

    versions: Mapped[List["PolicyVersion"]] = relationship(
        "PolicyVersion",
        back_populates="policy",
        cascade="all, delete-orphan",
        order_by="desc(PolicyVersion.version)"
    )

    # Methods
    def is_effective(self) -> bool:
        """Check if policy is currently effective."""
        now = datetime.utcnow()
        if not self.is_active:
            return False
        if self.effective_from and self.effective_from > now:
            return False
        if self.effective_until and self.effective_until < now:
            return False
        return True

    def evaluate(self, context: Dict[str, Any]) -> PolicyEffect:
        """Evaluate policy against given context."""
        if not self.is_effective():
            return None

        # Track evaluation
        self.total_evaluations += 1
        self.last_evaluated_at = datetime.utcnow()

        # Evaluate all rules in order
        for rule in self.rules:
            if not rule.is_active:
                continue

            result = rule.evaluate(context)
            if result is not None:
                # Track decision metrics
                if result == PolicyEffect.ALLOW:
                    self.total_allows += 1
                elif result == PolicyEffect.DENY:
                    self.total_denies += 1

                return result

        # Default deny if no rules match
        self.total_denies += 1
        return PolicyEffect.DENY

    def update_fitness_scores(self, security: float, productivity: float, compliance: float) -> None:
        """Update fitness scores from genetic algorithm evaluation."""
        self.security_score = max(0.0, min(1.0, security))
        self.productivity_score = max(0.0, min(1.0, productivity))
        self.compliance_score = max(0.0, min(1.0, compliance))

        # Calculate overall fitness (can be weighted differently)
        self.fitness_score = (
            0.4 * self.security_score +
            0.3 * self.productivity_score +
            0.3 * self.compliance_score
        )

    def record_false_positive(self) -> None:
        """Record a false positive decision."""
        self.false_positive_count += 1

    def record_false_negative(self) -> None:
        """Record a false negative decision."""
        self.false_negative_count += 1

    def calculate_accuracy(self) -> float:
        """Calculate policy accuracy based on false positives/negatives."""
        if self.total_evaluations == 0:
            return 0.0

        errors = self.false_positive_count + self.false_negative_count
        return 1.0 - (errors / self.total_evaluations)

    def create_version(self) -> "PolicyVersion":
        """Create a new version snapshot of the policy."""
        version = PolicyVersion(
            policy_id=self.id,
            version=self.version,
            name=self.name,
            description=self.description,
            policy_type=self.policy_type,
            rules_snapshot=self.export_rules(),
            fitness_score=self.fitness_score,
            security_score=self.security_score,
            productivity_score=self.productivity_score,
            compliance_score=self.compliance_score,
            created_by=self.last_modified_by
        )
        self.version += 1
        return version

    def export_rules(self) -> List[Dict[str, Any]]:
        """Export all rules as a list of dictionaries."""
        return [rule.to_dict() for rule in self.rules]

    def import_rules(self, rules_data: List[Dict[str, Any]]) -> None:
        """Import rules from a list of dictionaries."""
        # Clear existing rules
        self.rules.clear()

        # Create new rules
        for i, rule_data in enumerate(rules_data):
            rule = PolicyRule(
                policy_id=self.id,
                order=i,
                **rule_data
            )
            self.rules.append(rule)

    def to_chromosome(self) -> List[Any]:
        """Convert policy to chromosome representation for GA."""
        if self.chromosome:
            return self.chromosome

        # Basic chromosome structure
        chromosome = []

        # Encode policy type and priority
        chromosome.append(self.policy_type)
        chromosome.append(self.priority)

        # Encode rules
        for rule in self.rules:
            chromosome.extend(rule.to_gene_sequence())

        return chromosome

    def from_chromosome(self, chromosome: List[Any]) -> None:
        """Update policy from chromosome representation."""
        self.chromosome = chromosome
        self.is_ga_optimized = True

        # Decode policy type and priority
        if len(chromosome) >= 2:
            self.policy_type = chromosome[0]
            self.priority = chromosome[1]

        # TODO: Decode rules from remaining chromosome

    def to_dict(self, include_rules: bool = True) -> Dict[str, Any]:
        """Convert policy to dictionary representation."""
        data = {
            "id": self.id,
            "uuid": str(self.uuid),
            "name": self.name,
            "description": self.description,
            "version": self.version,
            "policy_type": self.policy_type,
            "priority": self.priority,
            "priority_level": self.priority_level,
            "is_active": self.is_active,
            "is_ga_optimized": self.is_ga_optimized,
            "is_effective": self.is_effective(),
            "fitness_score": self.fitness_score,
            "security_score": self.security_score,
            "productivity_score": self.productivity_score,
            "compliance_score": self.compliance_score,
            "accuracy": self.calculate_accuracy(),
            "total_evaluations": self.total_evaluations,
            "total_allows": self.total_allows,
            "total_denies": self.total_denies,
            "compliance_frameworks": self.compliance_frameworks,
            "effective_from": self.effective_from.isoformat() if self.effective_from else None,
            "effective_until": self.effective_until.isoformat() if self.effective_until else None,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
        }

        if include_rules:
            data["rules"] = [rule.to_dict() for rule in self.rules]

        return data

    def __repr__(self) -> str:
        """String representation of AccessPolicy."""
        return f"<AccessPolicy(id={self.id}, name='{self.name}', type='{self.policy_type}', fitness={self.fitness_score})>"


class PolicyRule(BaseModel):
    """
    PolicyRule model representing individual rules within a policy.

    Rules define specific conditions and actions that make up a policy.
    """

    __tablename__ = "policy_rules"
    __table_args__ = (
        Index('ix_rule_policy_order', 'policy_id', 'order'),
        Index('ix_rule_resource', 'resource'),
        Index('ix_rule_action', 'action'),
    )

    # Rule identifiers
    policy_id = Column(Integer, ForeignKey("access_policies.id"), nullable=False)
    order = Column(Integer, default=0, nullable=False)  # Evaluation order
    name = Column(String(255), nullable=True)
    description = Column(Text, nullable=True)

    # Rule components
    resource = Column(String(255), nullable=False)  # e.g., "github:repo:*", "jira:project:ENG"
    action = Column(String(100), nullable=False)  # e.g., "read", "write", "delete", "*"
    effect = Column(String(20), default=PolicyEffect.ALLOW.value, nullable=False)

    # Conditions
    conditions = Column(JSON, default=dict, nullable=False)  # Complex condition tree
    condition_expression = Column(Text, nullable=True)  # Human-readable condition

    # Rule metadata
    is_active = Column(Boolean, default=True, nullable=False)
    risk_score_threshold = Column(Integer, nullable=True)  # Only apply if risk score below

    # Statistics
    evaluation_count = Column(Integer, default=0, nullable=False)
    match_count = Column(Integer, default=0, nullable=False)

    # Relationships
    policy: Mapped["AccessPolicy"] = relationship(
        "AccessPolicy",
        back_populates="rules"
    )

    # Methods
    def evaluate(self, context: Dict[str, Any]) -> Optional[PolicyEffect]:
        """Evaluate rule against context."""
        self.evaluation_count += 1

        # Check resource match
        if not self._matches_resource(context.get("resource")):
            return None

        # Check action match
        if not self._matches_action(context.get("action")):
            return None

        # Check conditions
        if not self._evaluate_conditions(context):
            return None

        # Check risk score if specified
        if self.risk_score_threshold is not None:
            user_risk = context.get("user", {}).get("risk_score", 0)
            if user_risk > self.risk_score_threshold:
                return None

        self.match_count += 1
        return PolicyEffect(self.effect)

    def _matches_resource(self, resource: str) -> bool:
        """Check if resource matches rule pattern."""
        if not resource:
            return False

        # Simple wildcard matching (can be enhanced)
        if self.resource == "*":
            return True

        if "*" in self.resource:
            pattern = self.resource.replace("*", ".*")
            import re
            return bool(re.match(pattern, resource))

        return self.resource == resource

    def _matches_action(self, action: str) -> bool:
        """Check if action matches rule."""
        if not action:
            return False

        if self.action == "*":
            return True

        return self.action == action

    def _evaluate_conditions(self, context: Dict[str, Any]) -> bool:
        """Evaluate complex conditions against context."""
        if not self.conditions:
            return True

        # Simple condition evaluation (can be enhanced with expression engine)
        for key, expected_value in self.conditions.items():
            actual_value = self._get_nested_value(context, key)

            if isinstance(expected_value, dict):
                # Handle operators like {"$gt": 5, "$lt": 10}
                if not self._evaluate_operators(actual_value, expected_value):
                    return False
            elif actual_value != expected_value:
                return False

        return True

    def _get_nested_value(self, data: Dict[str, Any], path: str) -> Any:
        """Get nested value from dictionary using dot notation."""
        keys = path.split(".")
        value = data

        for key in keys:
            if isinstance(value, dict):
                value = value.get(key)
            else:
                return None

        return value

    def _evaluate_operators(self, value: Any, operators: Dict[str, Any]) -> bool:
        """Evaluate conditional operators."""
        for op, expected in operators.items():
            if op == "$eq" and value != expected:
                return False
            elif op == "$ne" and value == expected:
                return False
            elif op == "$gt" and not (value > expected):
                return False
            elif op == "$gte" and not (value >= expected):
                return False
            elif op == "$lt" and not (value < expected):
                return False
            elif op == "$lte" and not (value <= expected):
                return False
            elif op == "$in" and value not in expected:
                return False
            elif op == "$nin" and value in expected:
                return False

        return True

    def to_gene_sequence(self) -> List[Any]:
        """Convert rule to gene sequence for GA."""
        return [
            self.resource,
            self.action,
            self.effect,
            self.conditions,
            self.risk_score_threshold
        ]

    def to_dict(self) -> Dict[str, Any]:
        """Convert rule to dictionary representation."""
        return {
            "id": self.id,
            "order": self.order,
            "name": self.name,
            "description": self.description,
            "resource": self.resource,
            "action": self.action,
            "effect": self.effect,
            "conditions": self.conditions,
            "condition_expression": self.condition_expression,
            "is_active": self.is_active,
            "risk_score_threshold": self.risk_score_threshold,
            "evaluation_count": self.evaluation_count,
            "match_count": self.match_count,
            "match_rate": self.match_count / self.evaluation_count if self.evaluation_count > 0 else 0
        }

    def __repr__(self) -> str:
        """String representation of PolicyRule."""
        return f"<PolicyRule(id={self.id}, resource='{self.resource}', action='{self.action}', effect='{self.effect}')>"


class PolicyVersion(BaseModel):
    """
    PolicyVersion model for tracking policy history.

    Stores snapshots of policies at different versions for audit trail
    and rollback capabilities.
    """

    __tablename__ = "policy_versions"
    __table_args__ = (
        Index('ix_policy_version', 'policy_id', 'version'),
    )

    # Version identifiers
    policy_id = Column(Integer, ForeignKey("access_policies.id"), nullable=False)
    version = Column(Integer, nullable=False)

    # Snapshot data
    name = Column(String(255), nullable=False)
    description = Column(Text, nullable=True)
    policy_type = Column(String(50), nullable=False)
    rules_snapshot = Column(JSON, nullable=False)  # Complete rules at this version

    # Fitness scores at time of versioning
    fitness_score = Column(Float, nullable=True)
    security_score = Column(Float, nullable=True)
    productivity_score = Column(Float, nullable=True)
    compliance_score = Column(Float, nullable=True)

    # Version metadata
    change_description = Column(Text, nullable=True)
    created_by = Column(Integer, ForeignKey("users.id"), nullable=True)

    # Relationships
    policy: Mapped["AccessPolicy"] = relationship(
        "AccessPolicy",
        back_populates="versions"
    )

    def to_dict(self) -> Dict[str, Any]:
        """Convert version to dictionary representation."""
        return {
            "id": self.id,
            "policy_id": self.policy_id,
            "version": self.version,
            "name": self.name,
            "description": self.description,
            "policy_type": self.policy_type,
            "rules_snapshot": self.rules_snapshot,
            "fitness_score": self.fitness_score,
            "security_score": self.security_score,
            "productivity_score": self.productivity_score,
            "compliance_score": self.compliance_score,
            "change_description": self.change_description,
            "created_by": self.created_by,
            "created_at": self.created_at.isoformat()
        }

    def __repr__(self) -> str:
        """String representation of PolicyVersion."""
        return f"<PolicyVersion(id={self.id}, policy_id={self.policy_id}, version={self.version})>"
