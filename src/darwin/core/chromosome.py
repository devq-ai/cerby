"""
Chromosome Representation for Policy Genetic Algorithm.

This module defines the chromosome structure for representing access policies
in the genetic algorithm, including genes, encoding/decoding, and validation.
"""

from typing import List, Dict, Any, Optional, Union, Tuple
from dataclasses import dataclass, field
from enum import Enum
import random
import json
import hashlib
from copy import deepcopy


class GeneType(Enum):
    """Types of genes in the policy chromosome."""
    RESOURCE = "resource"
    ACTION = "action"
    CONDITION = "condition"
    EFFECT = "effect"
    PRIORITY = "priority"


class ConditionOperator(Enum):
    """Operators for policy conditions."""
    EQUALS = "eq"
    NOT_EQUALS = "ne"
    GREATER_THAN = "gt"
    LESS_THAN = "lt"
    GREATER_EQUAL = "gte"
    LESS_EQUAL = "lte"
    IN = "in"
    NOT_IN = "not_in"
    CONTAINS = "contains"
    STARTS_WITH = "starts_with"
    ENDS_WITH = "ends_with"
    REGEX = "regex"


class PolicyEffect(Enum):
    """Effect of a policy rule."""
    ALLOW = "allow"
    DENY = "deny"


@dataclass
class Gene:
    """
    Base class for a gene in the policy chromosome.

    A gene represents a single component of a policy rule,
    such as a resource, action, or condition.
    """
    gene_type: GeneType
    value: Any
    mutable: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)

    def mutate(self, mutation_params: Dict[str, Any]) -> None:
        """Mutate this gene based on its type."""
        if not self.mutable:
            return

        if self.gene_type == GeneType.RESOURCE:
            self._mutate_resource(mutation_params)
        elif self.gene_type == GeneType.ACTION:
            self._mutate_action(mutation_params)
        elif self.gene_type == GeneType.CONDITION:
            self._mutate_condition(mutation_params)
        elif self.gene_type == GeneType.PRIORITY:
            self._mutate_priority(mutation_params)

    def _mutate_resource(self, params: Dict[str, Any]) -> None:
        """Mutate a resource gene."""
        resource_pool = params.get("resource_pool", [])
        if resource_pool and random.random() < params.get("change_probability", 0.3):
            self.value = random.choice(resource_pool)
        elif isinstance(self.value, str) and "*" in self.value:
            # Mutate wildcard patterns
            if random.random() < 0.5:
                # Make more specific
                self.value = self.value.replace("*", random.choice(["read", "write", "admin"]))
            else:
                # Change wildcard position
                parts = self.value.split(":")
                if len(parts) > 1:
                    idx = random.randint(0, len(parts) - 1)
                    parts[idx] = "*"
                    self.value = ":".join(parts)

    def _mutate_action(self, params: Dict[str, Any]) -> None:
        """Mutate an action gene."""
        action_pool = params.get("action_pool", ["read", "write", "delete", "create", "update"])
        if isinstance(self.value, list):
            if random.random() < 0.5 and len(self.value) > 1:
                # Remove an action
                self.value.remove(random.choice(self.value))
            else:
                # Add an action
                available = [a for a in action_pool if a not in self.value]
                if available:
                    self.value.append(random.choice(available))
        else:
            self.value = random.choice(action_pool)

    def _mutate_condition(self, params: Dict[str, Any]) -> None:
        """Mutate a condition gene."""
        if isinstance(self.value, dict):
            if "attribute" in self.value and random.random() < 0.3:
                # Change attribute
                attr_pool = params.get("attribute_pool", ["department", "role", "level", "location"])
                self.value["attribute"] = random.choice(attr_pool)

            if "operator" in self.value and random.random() < 0.2:
                # Change operator
                self.value["operator"] = random.choice(list(ConditionOperator)).value

            if "value" in self.value and random.random() < 0.4:
                # Mutate value based on type
                self._mutate_condition_value(params)

    def _mutate_condition_value(self, params: Dict[str, Any]) -> None:
        """Mutate the value in a condition."""
        current_value = self.value.get("value")
        attribute = self.value.get("attribute", "")

        if attribute == "department":
            departments = params.get("departments", ["Engineering", "Sales", "HR", "Finance"])
            self.value["value"] = random.choice(departments)
        elif attribute == "level" and isinstance(current_value, (int, float)):
            # Mutate numeric values
            delta = random.uniform(-2, 2)
            self.value["value"] = max(1, current_value + delta)
        elif isinstance(current_value, list):
            # Mutate list values
            if random.random() < 0.5 and len(current_value) > 1:
                current_value.remove(random.choice(current_value))
            else:
                # Add a random value
                if attribute in params.get("value_pools", {}):
                    pool = params["value_pools"][attribute]
                    available = [v for v in pool if v not in current_value]
                    if available:
                        current_value.append(random.choice(available))

    def _mutate_priority(self, params: Dict[str, Any]) -> None:
        """Mutate a priority gene."""
        if isinstance(self.value, (int, float)):
            delta = random.randint(-10, 10)
            self.value = max(0, min(1000, self.value + delta))

    def to_dict(self) -> Dict[str, Any]:
        """Convert gene to dictionary representation."""
        return {
            "type": self.gene_type.value,
            "value": self.value,
            "mutable": self.mutable,
            "metadata": self.metadata
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Gene":
        """Create gene from dictionary representation."""
        return cls(
            gene_type=GeneType(data["type"]),
            value=data["value"],
            mutable=data.get("mutable", True),
            metadata=data.get("metadata", {})
        )


@dataclass
class PolicyRule:
    """Represents a single policy rule composed of multiple genes."""

    resource: Gene
    actions: List[Gene]
    conditions: List[Gene]
    effect: Gene
    priority: Gene
    rule_id: Optional[str] = None

    def __post_init__(self):
        """Generate rule ID if not provided."""
        if not self.rule_id:
            self.rule_id = self._generate_id()

    def _generate_id(self) -> str:
        """Generate a unique ID for this rule."""
        content = f"{self.resource.value}:{self.actions}:{self.conditions}:{self.effect.value}"
        return hashlib.md5(content.encode()).hexdigest()[:8]

    def mutate(self, mutation_params: Dict[str, Any]) -> None:
        """Mutate this rule."""
        # Mutate individual genes
        if random.random() < mutation_params.get("resource_mutation_rate", 0.1):
            self.resource.mutate(mutation_params)

        # Mutate actions
        for action in self.actions:
            if random.random() < mutation_params.get("action_mutation_rate", 0.15):
                action.mutate(mutation_params)

        # Add/remove actions
        if random.random() < mutation_params.get("action_add_remove_rate", 0.1):
            if len(self.actions) > 1 and random.random() < 0.5:
                self.actions.pop(random.randint(0, len(self.actions) - 1))
            else:
                action_pool = mutation_params.get("action_pool", ["read", "write"])
                new_action = Gene(GeneType.ACTION, random.choice(action_pool))
                self.actions.append(new_action)

        # Mutate conditions
        for condition in self.conditions:
            if random.random() < mutation_params.get("condition_mutation_rate", 0.2):
                condition.mutate(mutation_params)

        # Add/remove conditions
        if random.random() < mutation_params.get("condition_add_remove_rate", 0.15):
            if len(self.conditions) > 0 and random.random() < 0.3:
                self.conditions.pop(random.randint(0, len(self.conditions) - 1))
            else:
                self._add_random_condition(mutation_params)

        # Rarely flip effect
        if random.random() < mutation_params.get("effect_flip_rate", 0.05):
            current_effect = PolicyEffect(self.effect.value)
            new_effect = PolicyEffect.DENY if current_effect == PolicyEffect.ALLOW else PolicyEffect.ALLOW
            self.effect.value = new_effect.value

        # Mutate priority
        if random.random() < mutation_params.get("priority_mutation_rate", 0.1):
            self.priority.mutate(mutation_params)

    def _add_random_condition(self, params: Dict[str, Any]) -> None:
        """Add a random condition to this rule."""
        attributes = params.get("attribute_pool", ["department", "role", "level"])
        attribute = random.choice(attributes)

        operator = random.choice(list(ConditionOperator)).value

        # Generate appropriate value based on attribute
        if attribute == "department":
            value = random.choice(params.get("departments", ["Engineering", "Sales"]))
        elif attribute == "level":
            value = random.randint(1, 5)
        elif attribute == "role":
            value = random.choice(params.get("roles", ["user", "admin", "manager"]))
        else:
            value = "default"

        condition = Gene(
            GeneType.CONDITION,
            {
                "attribute": attribute,
                "operator": operator,
                "value": value
            }
        )
        self.conditions.append(condition)

    def to_dict(self) -> Dict[str, Any]:
        """Convert rule to dictionary representation."""
        return {
            "rule_id": self.rule_id,
            "resource": self.resource.to_dict(),
            "actions": [a.to_dict() for a in self.actions],
            "conditions": [c.to_dict() for c in self.conditions],
            "effect": self.effect.to_dict(),
            "priority": self.priority.to_dict()
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PolicyRule":
        """Create rule from dictionary representation."""
        return cls(
            resource=Gene.from_dict(data["resource"]),
            actions=[Gene.from_dict(a) for a in data["actions"]],
            conditions=[Gene.from_dict(c) for c in data["conditions"]],
            effect=Gene.from_dict(data["effect"]),
            priority=Gene.from_dict(data["priority"]),
            rule_id=data.get("rule_id")
        )


class PolicyChromosome:
    """
    Chromosome representing a complete access policy.

    A chromosome consists of multiple policy rules that together
    form a complete access control policy.
    """

    def __init__(self, rules: Optional[List[PolicyRule]] = None):
        """Initialize chromosome with rules."""
        self.rules: List[PolicyRule] = rules or []
        self.fitness_scores: Dict[str, float] = {}
        self.generation: int = 0
        self.chromosome_id: str = self._generate_id()
        self.metadata: Dict[str, Any] = {}

    def _generate_id(self) -> str:
        """Generate a unique ID for this chromosome."""
        content = json.dumps([r.to_dict() for r in self.rules], sort_keys=True)
        return hashlib.sha256(content.encode()).hexdigest()[:16]

    def add_rule(self, rule: PolicyRule) -> None:
        """Add a rule to this chromosome."""
        self.rules.append(rule)
        self.chromosome_id = self._generate_id()  # Regenerate ID

    def remove_rule(self, rule_id: str) -> bool:
        """Remove a rule by ID."""
        initial_length = len(self.rules)
        self.rules = [r for r in self.rules if r.rule_id != rule_id]
        if len(self.rules) < initial_length:
            self.chromosome_id = self._generate_id()
            return True
        return False

    def mutate(self, mutation_params: Dict[str, Any]) -> None:
        """Mutate this chromosome."""
        # Mutate existing rules
        for rule in self.rules:
            if random.random() < mutation_params.get("rule_mutation_rate", 0.3):
                rule.mutate(mutation_params)

        # Add new rules
        if random.random() < mutation_params.get("rule_add_rate", 0.1):
            if len(self.rules) < mutation_params.get("max_rules", 50):
                self.rules.append(self._create_random_rule(mutation_params))

        # Remove rules
        if random.random() < mutation_params.get("rule_remove_rate", 0.05):
            if len(self.rules) > mutation_params.get("min_rules", 1):
                self.rules.pop(random.randint(0, len(self.rules) - 1))

        # Regenerate ID after mutations
        self.chromosome_id = self._generate_id()

    def _create_random_rule(self, params: Dict[str, Any]) -> PolicyRule:
        """Create a random policy rule."""
        # Random resource
        resources = params.get("resource_pool", ["app:*", "data:*", "api:*"])
        resource = Gene(GeneType.RESOURCE, random.choice(resources))

        # Random actions
        action_pool = params.get("action_pool", ["read", "write", "delete"])
        num_actions = random.randint(1, 3)
        actions = [
            Gene(GeneType.ACTION, action)
            for action in random.sample(action_pool, min(num_actions, len(action_pool)))
        ]

        # Random conditions
        num_conditions = random.randint(0, 3)
        conditions = []
        for _ in range(num_conditions):
            attribute = random.choice(params.get("attribute_pool", ["department", "role"]))
            operator = random.choice(list(ConditionOperator)).value

            if attribute == "department":
                value = random.choice(params.get("departments", ["Engineering", "Sales"]))
            elif attribute == "role":
                value = random.choice(params.get("roles", ["user", "admin"]))
            else:
                value = "default"

            condition = Gene(
                GeneType.CONDITION,
                {"attribute": attribute, "operator": operator, "value": value}
            )
            conditions.append(condition)

        # Random effect (bias towards ALLOW)
        effect = Gene(
            GeneType.EFFECT,
            PolicyEffect.ALLOW.value if random.random() < 0.8 else PolicyEffect.DENY.value
        )

        # Random priority
        priority = Gene(GeneType.PRIORITY, random.randint(1, 100))

        return PolicyRule(
            resource=resource,
            actions=actions,
            conditions=conditions,
            effect=effect,
            priority=priority
        )

    def crossover(self, other: "PolicyChromosome", crossover_params: Dict[str, Any]) -> Tuple["PolicyChromosome", "PolicyChromosome"]:
        """Perform crossover with another chromosome."""
        method = crossover_params.get("method", "uniform")

        if method == "uniform":
            return self._uniform_crossover(other, crossover_params)
        elif method == "single_point":
            return self._single_point_crossover(other, crossover_params)
        elif method == "two_point":
            return self._two_point_crossover(other, crossover_params)
        else:
            raise ValueError(f"Unknown crossover method: {method}")

    def _uniform_crossover(self, other: "PolicyChromosome", params: Dict[str, Any]) -> Tuple["PolicyChromosome", "PolicyChromosome"]:
        """Perform uniform crossover."""
        child1_rules = []
        child2_rules = []

        # Process rules from both parents
        max_rules = max(len(self.rules), len(other.rules))

        for i in range(max_rules):
            if i < len(self.rules) and i < len(other.rules):
                # Both parents have this rule index
                if random.random() < 0.5:
                    child1_rules.append(deepcopy(self.rules[i]))
                    child2_rules.append(deepcopy(other.rules[i]))
                else:
                    child1_rules.append(deepcopy(other.rules[i]))
                    child2_rules.append(deepcopy(self.rules[i]))
            elif i < len(self.rules):
                # Only parent 1 has this rule
                if random.random() < params.get("inherit_probability", 0.7):
                    child1_rules.append(deepcopy(self.rules[i]))
                if random.random() < params.get("inherit_probability", 0.7):
                    child2_rules.append(deepcopy(self.rules[i]))
            else:
                # Only parent 2 has this rule
                if random.random() < params.get("inherit_probability", 0.7):
                    child1_rules.append(deepcopy(other.rules[i]))
                if random.random() < params.get("inherit_probability", 0.7):
                    child2_rules.append(deepcopy(other.rules[i]))

        child1 = PolicyChromosome(child1_rules)
        child2 = PolicyChromosome(child2_rules)

        return child1, child2

    def _single_point_crossover(self, other: "PolicyChromosome", params: Dict[str, Any]) -> Tuple["PolicyChromosome", "PolicyChromosome"]:
        """Perform single-point crossover."""
        point = random.randint(1, min(len(self.rules), len(other.rules)) - 1)

        child1_rules = deepcopy(self.rules[:point]) + deepcopy(other.rules[point:])
        child2_rules = deepcopy(other.rules[:point]) + deepcopy(self.rules[point:])

        return PolicyChromosome(child1_rules), PolicyChromosome(child2_rules)

    def _two_point_crossover(self, other: "PolicyChromosome", params: Dict[str, Any]) -> Tuple["PolicyChromosome", "PolicyChromosome"]:
        """Perform two-point crossover."""
        size = min(len(self.rules), len(other.rules))
        if size < 3:
            return self._single_point_crossover(other, params)

        point1 = random.randint(1, size - 2)
        point2 = random.randint(point1 + 1, size - 1)

        child1_rules = (
            deepcopy(self.rules[:point1]) +
            deepcopy(other.rules[point1:point2]) +
            deepcopy(self.rules[point2:])
        )
        child2_rules = (
            deepcopy(other.rules[:point1]) +
            deepcopy(self.rules[point1:point2]) +
            deepcopy(other.rules[point2:])
        )

        return PolicyChromosome(child1_rules), PolicyChromosome(child2_rules)

    def validate(self, constraints: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """Validate this chromosome against constraints."""
        errors = []

        # Check rule count
        if len(self.rules) < constraints.get("min_rules", 1):
            errors.append(f"Too few rules: {len(self.rules)} < {constraints.get('min_rules', 1)}")
        if len(self.rules) > constraints.get("max_rules", 50):
            errors.append(f"Too many rules: {len(self.rules)} > {constraints.get('max_rules', 50)}")

        # Check for duplicate rules
        rule_signatures = set()
        for rule in self.rules:
            signature = f"{rule.resource.value}:{[a.value for a in rule.actions]}"
            if signature in rule_signatures:
                errors.append(f"Duplicate rule detected: {signature}")
            rule_signatures.add(signature)

        # Check for conflicting rules
        for i, rule1 in enumerate(self.rules):
            for rule2 in self.rules[i+1:]:
                if self._rules_conflict(rule1, rule2):
                    errors.append(f"Conflicting rules: {rule1.rule_id} and {rule2.rule_id}")

        # Check required resources
        if "required_resources" in constraints:
            covered_resources = {rule.resource.value for rule in self.rules}
            for required in constraints["required_resources"]:
                if not any(self._resource_matches(required, res) for res in covered_resources):
                    errors.append(f"Missing required resource coverage: {required}")

        return len(errors) == 0, errors

    def _rules_conflict(self, rule1: PolicyRule, rule2: PolicyRule) -> bool:
        """Check if two rules conflict."""
        # Rules conflict if they have the same resource and actions but different effects
        if rule1.resource.value != rule2.resource.value:
            return False

        actions1 = {a.value for a in rule1.actions}
        actions2 = {a.value for a in rule2.actions}

        if not actions1.intersection(actions2):
            return False

        return rule1.effect.value != rule2.effect.value

    def _resource_matches(self, pattern: str, resource: str) -> bool:
        """Check if a resource matches a pattern (with wildcard support)."""
        if pattern == resource:
            return True

        if "*" in pattern:
            import re
            regex_pattern = pattern.replace("*", ".*")
            return bool(re.match(f"^{regex_pattern}$", resource))

        return False

    def to_dict(self) -> Dict[str, Any]:
        """Convert chromosome to dictionary representation."""
        return {
            "chromosome_id": self.chromosome_id,
            "generation": self.generation,
            "rules": [r.to_dict() for r in self.rules],
            "fitness_scores": self.fitness_scores,
            "metadata": self.metadata
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PolicyChromosome":
        """Create chromosome from dictionary representation."""
        chromosome = cls(
            rules=[PolicyRule.from_dict(r) for r in data.get("rules", [])]
        )
        chromosome.generation = data.get("generation", 0)
        chromosome.fitness_scores = data.get("fitness_scores", {})
        chromosome.metadata = data.get("metadata", {})
        return chromosome

    def clone(self) -> "PolicyChromosome":
        """Create a deep copy of this chromosome."""
        return PolicyChromosome.from_dict(self.to_dict())

    def __repr__(self) -> str:
        """String representation of chromosome."""
        return f"PolicyChromosome(id={self.chromosome_id[:8]}, rules={len(self.rules)}, fitness={self.fitness_scores})"
