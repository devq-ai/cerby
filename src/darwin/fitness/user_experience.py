"""
User experience fitness evaluation for policy chromosomes.

This module implements fitness functions that evaluate how user-friendly
and understandable the access control policies are.
"""

from typing import Dict, Any, List, Optional
from dataclasses import dataclass
import re

from src.darwin.core.chromosome import (
    PolicyChromosome,
    PolicyRule,
    Gene,
    GeneType,
    PolicyEffect,
    ConditionOperator
)
from src.darwin.fitness.base import FitnessFunction, FitnessMetrics
from src.darwin.fitness.metrics import UXMetrics


class UserExperienceFitness(FitnessFunction):
    """
    Fitness function that evaluates user experience aspects of policies.

    This function considers:
    - Rule clarity and understandability
    - Policy simplicity
    - Consistency across rules
    - Discoverability of permissions
    - Error prevention
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize user experience fitness function.

        Args:
            config: Configuration with keys:
                - max_rules_per_user: Maximum rules a user should deal with
                - max_conditions_per_rule: Maximum conditions for clarity
                - clarity_weight: Weight for clarity scoring
                - simplicity_weight: Weight for simplicity scoring
                - preferred_operators: List of preferred condition operators
        """
        super().__init__(config)

        # Configuration
        self.max_rules_per_user = self.config.get('max_rules_per_user', 10)
        self.max_conditions_per_rule = self.config.get('max_conditions_per_rule', 3)

        # Weights
        self.weights = {
            'clarity': self.config.get('clarity_weight', 0.3),
            'simplicity': self.config.get('simplicity_weight', 0.3),
            'consistency': self.config.get('consistency_weight', 0.2),
            'discoverability': self.config.get('discoverability_weight', 0.1),
            'error_prevention': self.config.get('error_prevention_weight', 0.1)
        }

        # Preferred operators for clarity
        self.preferred_operators = self.config.get('preferred_operators', [
            ConditionOperator.EQUALS,
            ConditionOperator.IN,
            ConditionOperator.NOT_EQUALS
        ])

    def evaluate(self, chromosome: PolicyChromosome) -> float:
        """
        Evaluate the user experience fitness of a policy chromosome.

        Args:
            chromosome: The policy chromosome to evaluate

        Returns:
            UX fitness score between 0 and 1
        """
        metrics = self.calculate_metrics(chromosome)
        return metrics.overall_ux_score

    def calculate_metrics(self, chromosome: PolicyChromosome) -> UXMetrics:
        """
        Calculate detailed UX metrics for a policy chromosome.

        Args:
            chromosome: The policy chromosome to analyze

        Returns:
            Detailed UX metrics
        """
        # Calculate individual UX scores
        clarity = self._calculate_clarity_score(chromosome)
        simplicity = self._calculate_simplicity_score(chromosome)
        consistency = self._calculate_consistency_score(chromosome)
        discoverability = self._calculate_discoverability_score(chromosome)
        error_prevention = self._calculate_error_prevention_score(chromosome)

        # Calculate weighted overall score
        overall_score = (
            clarity * self.weights['clarity'] +
            simplicity * self.weights['simplicity'] +
            consistency * self.weights['consistency'] +
            discoverability * self.weights['discoverability'] +
            error_prevention * self.weights['error_prevention']
        )

        return UXMetrics(
            clarity_score=clarity,
            simplicity_score=simplicity,
            consistency_score=consistency,
            discoverability_score=discoverability,
            error_prevention_score=error_prevention,
            overall_ux_score=overall_score,
            details={
                'rule_count': len(chromosome.rules),
                'avg_conditions_per_rule': self._average_conditions_per_rule(chromosome),
                'complex_rules': self._count_complex_rules(chromosome),
                'naming_consistency': self._check_naming_consistency(chromosome)
            }
        )

    def _calculate_clarity_score(self, chromosome: PolicyChromosome) -> float:
        """Calculate clarity score based on rule understandability."""
        if not chromosome.rules:
            return 1.0

        clarity_scores = []

        for rule in chromosome.rules:
            # Resource clarity
            resource_clarity = self._score_resource_clarity(rule.resource.value)

            # Action clarity
            action_clarity = self._score_action_clarity(rule.actions)

            # Condition clarity
            condition_clarity = self._score_condition_clarity(rule.conditions)

            # Rule clarity is average of components
            rule_clarity = (resource_clarity + action_clarity + condition_clarity) / 3
            clarity_scores.append(rule_clarity)

        return sum(clarity_scores) / len(clarity_scores)

    def _score_resource_clarity(self, resource: str) -> float:
        """Score clarity of resource specification."""
        # Penalize wildcards (less clear)
        wildcard_count = resource.count('*')
        wildcard_penalty = wildcard_count * 0.2

        # Check for clear hierarchical structure
        parts = resource.split(':')
        if len(parts) == 3:  # service:type:identifier
            structure_bonus = 0.2
        else:
            structure_bonus = 0.0

        # Check for meaningful names (not UUIDs or hashes)
        if any(len(part) > 20 for part in parts):
            naming_penalty = 0.1
        else:
            naming_penalty = 0.0

        return max(0, min(1.0, 1.0 - wildcard_penalty + structure_bonus - naming_penalty))

    def _score_action_clarity(self, actions: List[Gene]) -> float:
        """Score clarity of actions."""
        if not actions:
            return 0.5

        # Standard, clear actions
        clear_actions = ['read', 'write', 'create', 'delete', 'update', 'list', 'view']

        clear_count = sum(1 for action in actions if action.value in clear_actions)
        clarity_ratio = clear_count / len(actions)

        # Penalty for too many actions (confusing)
        if len(actions) > 5:
            clarity_ratio *= 0.8

        return clarity_ratio

    def _score_condition_clarity(self, conditions: List[Gene]) -> float:
        """Score clarity of conditions."""
        if not conditions:
            return 1.0  # No conditions is clear

        # Too many conditions reduce clarity
        if len(conditions) > self.max_conditions_per_rule:
            base_score = 0.5
        else:
            base_score = 1.0 - (len(conditions) * 0.1)

        # Check operator clarity
        operator_clarity = 0.0
        for condition in conditions:
            if isinstance(condition.value, dict):
                operator = condition.value.get('operator')
                if operator in self.preferred_operators:
                    operator_clarity += 1.0
                elif operator == ConditionOperator.REGEX:
                    operator_clarity += 0.3  # Regex is hard to understand
                else:
                    operator_clarity += 0.7

        if conditions:
            operator_clarity /= len(conditions)

        return (base_score + operator_clarity) / 2

    def _calculate_simplicity_score(self, chromosome: PolicyChromosome) -> float:
        """Calculate simplicity score."""
        if not chromosome.rules:
            return 1.0

        # Rule count penalty
        if len(chromosome.rules) <= self.max_rules_per_user:
            rule_count_score = 1.0
        else:
            excess = len(chromosome.rules) - self.max_rules_per_user
            rule_count_score = max(0.3, 1.0 - (excess * 0.05))

        # Condition complexity
        avg_conditions = self._average_conditions_per_rule(chromosome)
        if avg_conditions <= 2:
            condition_score = 1.0
        elif avg_conditions <= self.max_conditions_per_rule:
            condition_score = 0.8
        else:
            condition_score = 0.5

        # Action simplicity
        avg_actions = sum(len(r.actions) for r in chromosome.rules) / len(chromosome.rules)
        if avg_actions <= 3:
            action_score = 1.0
        else:
            action_score = max(0.5, 1.0 - (avg_actions - 3) * 0.1)

        return (rule_count_score + condition_score + action_score) / 3

    def _calculate_consistency_score(self, chromosome: PolicyChromosome) -> float:
        """Calculate consistency score across rules."""
        if len(chromosome.rules) < 2:
            return 1.0

        consistency_checks = []

        # Naming consistency
        naming_score = self._check_naming_consistency(chromosome)
        consistency_checks.append(naming_score)

        # Pattern consistency
        pattern_score = self._check_pattern_consistency(chromosome)
        consistency_checks.append(pattern_score)

        # Priority consistency
        priority_score = self._check_priority_consistency(chromosome)
        consistency_checks.append(priority_score)

        return sum(consistency_checks) / len(consistency_checks)

    def _check_naming_consistency(self, chromosome: PolicyChromosome) -> float:
        """Check consistency in resource naming."""
        if not chromosome.rules:
            return 1.0

        # Extract service names (first part of resource)
        services = []
        for rule in chromosome.rules:
            parts = rule.resource.value.split(':')
            if parts:
                services.append(parts[0])

        # Check for consistent naming patterns
        unique_services = set(services)
        if len(unique_services) <= 5:
            # Small number of services is good
            return 1.0
        else:
            # Too many different services might be confusing
            return max(0.5, 1.0 - (len(unique_services) - 5) * 0.05)

    def _check_pattern_consistency(self, chromosome: PolicyChromosome) -> float:
        """Check for consistent patterns in rules."""
        if not chromosome.rules:
            return 1.0

        # Check if similar resources have similar rules
        resource_patterns = {}
        for rule in chromosome.rules:
            pattern = self._extract_pattern(rule.resource.value)
            if pattern not in resource_patterns:
                resource_patterns[pattern] = []
            resource_patterns[pattern].append(rule)

        # Rules with same pattern should have similar structure
        consistency_scores = []
        for pattern, rules in resource_patterns.items():
            if len(rules) > 1:
                # Compare action sets
                action_sets = [set(a.value for a in r.actions) for r in rules]
                if all(s == action_sets[0] for s in action_sets):
                    consistency_scores.append(1.0)
                else:
                    consistency_scores.append(0.5)

        return sum(consistency_scores) / len(consistency_scores) if consistency_scores else 1.0

    def _extract_pattern(self, resource: str) -> str:
        """Extract pattern from resource string."""
        # Replace specific identifiers with wildcards
        parts = resource.split(':')
        if len(parts) >= 2:
            return f"{parts[0]}:{parts[1]}:*"
        return resource

    def _check_priority_consistency(self, chromosome: PolicyChromosome) -> float:
        """Check if priorities are consistently applied."""
        if not chromosome.rules:
            return 1.0

        # Group rules by effect
        allow_priorities = [r.priority.value for r in chromosome.rules
                          if r.effect.value == PolicyEffect.ALLOW]
        deny_priorities = [r.priority.value for r in chromosome.rules
                         if r.effect.value == PolicyEffect.DENY]

        # Deny rules should generally have higher priority
        if allow_priorities and deny_priorities:
            avg_allow = sum(allow_priorities) / len(allow_priorities)
            avg_deny = sum(deny_priorities) / len(deny_priorities)
            if avg_deny > avg_allow:
                return 1.0
            else:
                return 0.7

        return 1.0

    def _calculate_discoverability_score(self, chromosome: PolicyChromosome) -> float:
        """Calculate how easily users can discover their permissions."""
        if not chromosome.rules:
            return 0.0

        # Clear resource naming helps discoverability
        clear_resources = sum(
            1 for rule in chromosome.rules
            if '*' not in rule.resource.value and len(rule.resource.value.split(':')) == 3
        )

        # Logical grouping helps discoverability
        service_groups = {}
        for rule in chromosome.rules:
            service = rule.resource.value.split(':')[0]
            if service not in service_groups:
                service_groups[service] = 0
            service_groups[service] += 1

        # Good if rules are grouped by service
        well_grouped = sum(1 for count in service_groups.values() if count >= 2)
        grouping_score = well_grouped / len(service_groups) if service_groups else 0

        resource_clarity = clear_resources / len(chromosome.rules)

        return (resource_clarity + grouping_score) / 2

    def _calculate_error_prevention_score(self, chromosome: PolicyChromosome) -> float:
        """Calculate error prevention capabilities."""
        if not chromosome.rules:
            return 0.5

        prevention_features = []

        # Explicit deny rules prevent errors
        deny_rules = sum(1 for r in chromosome.rules if r.effect.value == PolicyEffect.DENY)
        deny_ratio = deny_rules / len(chromosome.rules)
        prevention_features.append(min(1.0, deny_ratio * 3))  # 33% deny rules is good

        # Specific conditions prevent accidental access
        conditional_rules = sum(1 for r in chromosome.rules if len(r.conditions) > 0)
        condition_ratio = conditional_rules / len(chromosome.rules)
        prevention_features.append(condition_ratio)

        # No overly broad permissions
        broad_rules = sum(
            1 for r in chromosome.rules
            if r.resource.value == '*:*:*' or
            any(a.value == '*' for a in r.actions)
        )
        broad_ratio = 1.0 - (broad_rules / len(chromosome.rules))
        prevention_features.append(broad_ratio)

        return sum(prevention_features) / len(prevention_features)

    def _average_conditions_per_rule(self, chromosome: PolicyChromosome) -> float:
        """Calculate average number of conditions per rule."""
        if not chromosome.rules:
            return 0.0

        total_conditions = sum(len(rule.conditions) for rule in chromosome.rules)
        return total_conditions / len(chromosome.rules)

    def _count_complex_rules(self, chromosome: PolicyChromosome) -> int:
        """Count rules that are considered complex."""
        complex_count = 0

        for rule in chromosome.rules:
            is_complex = False

            # Many conditions
            if len(rule.conditions) > self.max_conditions_per_rule:
                is_complex = True

            # Complex operators
            for condition in rule.conditions:
                if isinstance(condition.value, dict):
                    operator = condition.value.get('operator')
                    if operator == ConditionOperator.REGEX:
                        is_complex = True
                        break

            # Many actions
            if len(rule.actions) > 5:
                is_complex = True

            if is_complex:
                complex_count += 1

        return complex_count
