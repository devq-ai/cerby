"""
Security fitness evaluation for policy chromosomes.

This module implements fitness functions that evaluate the security
characteristics of access control policies.
"""

from typing import Dict, Any, List, Optional, Set
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


@dataclass
class SecurityMetrics(FitnessMetrics):
    """Detailed security metrics for a policy chromosome."""
    privilege_score: float
    mfa_coverage: float
    sensitive_resource_protection: float
    condition_strength: float
    deny_rule_presence: float
    wildcard_usage: float


class SecurityFitness(FitnessFunction):
    """
    Fitness function that evaluates the security aspects of a policy.

    This function considers:
    - Principle of least privilege
    - MFA requirements for sensitive resources
    - Condition-based access controls
    - Appropriate use of deny rules
    - Minimal wildcard usage
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize security fitness function.

        Args:
            config: Configuration with keys:
                - min_password_length: Minimum required password length
                - require_mfa: Whether MFA should be required for sensitive resources
                - max_privilege_scope: Maximum acceptable scope of privileges (0-1)
                - principle_of_least_privilege_weight: Weight for least privilege scoring
                - require_mfa_weight: Weight for MFA requirement scoring
                - sensitive_resources: List of resource patterns considered sensitive
        """
        super().__init__(config)

        # Default configuration
        self.min_password_length = self.config.get('min_password_length', 12)
        self.require_mfa = self.config.get('require_mfa', True)
        self.max_privilege_scope = self.config.get('max_privilege_scope', 0.3)

        # Weights for different security aspects
        self.weights = {
            'least_privilege': self.config.get('principle_of_least_privilege_weight', 0.3),
            'mfa': self.config.get('require_mfa_weight', 0.2),
            'conditions': self.config.get('condition_weight', 0.2),
            'deny_rules': self.config.get('deny_rule_weight', 0.1),
            'wildcards': self.config.get('wildcard_penalty_weight', 0.2)
        }

        # Sensitive resource patterns
        self.sensitive_resources = self.config.get('sensitive_resources', [
            'hr:*', 'financial:*', 'admin:*', 'iam:*', 'security:*',
            '*:sensitive:*', '*:confidential:*', '*:pii:*'
        ])

    def evaluate(self, chromosome: PolicyChromosome) -> float:
        """
        Evaluate the security fitness of a policy chromosome.

        Args:
            chromosome: The policy chromosome to evaluate

        Returns:
            Security fitness score between 0 and 1
        """
        metrics = self.calculate_metrics(chromosome)
        return metrics.score

    def calculate_metrics(self, chromosome: PolicyChromosome) -> SecurityMetrics:
        """
        Calculate detailed security metrics for a policy chromosome.

        Args:
            chromosome: The policy chromosome to analyze

        Returns:
            Detailed security metrics
        """
        # Calculate individual security scores
        privilege_score = self._calculate_privilege_score(chromosome)
        mfa_coverage = self._calculate_mfa_coverage(chromosome)
        sensitive_protection = self._calculate_sensitive_resource_protection(chromosome)
        condition_strength = self._calculate_condition_strength(chromosome)
        deny_presence = self._calculate_deny_rule_presence(chromosome)
        wildcard_score = self._calculate_wildcard_usage_score(chromosome)

        # Calculate weighted overall score
        overall_score = (
            privilege_score * self.weights['least_privilege'] +
            mfa_coverage * self.weights['mfa'] +
            condition_strength * self.weights['conditions'] +
            deny_presence * self.weights['deny_rules'] +
            wildcard_score * self.weights['wildcards']
        )

        # Bonus for comprehensive sensitive resource protection
        if sensitive_protection > 0.8:
            overall_score = min(1.0, overall_score * 1.1)

        return SecurityMetrics(
            score=overall_score,
            details={
                'privilege_score': privilege_score,
                'mfa_coverage': mfa_coverage,
                'sensitive_protection': sensitive_protection,
                'condition_strength': condition_strength,
                'deny_presence': deny_presence,
                'wildcard_score': wildcard_score
            },
            privilege_score=privilege_score,
            mfa_coverage=mfa_coverage,
            sensitive_resource_protection=sensitive_protection,
            condition_strength=condition_strength,
            deny_rule_presence=deny_presence,
            wildcard_usage=wildcard_score
        )

    def _calculate_privilege_score(self, chromosome: PolicyChromosome) -> float:
        """Calculate principle of least privilege score."""
        if not chromosome.rules:
            return 0.5  # No rules = no excessive privileges, but also no access

        total_score = 0.0

        for rule in chromosome.rules:
            # Penalize wildcard resources
            resource_specificity = self._calculate_resource_specificity(rule.resource.value)

            # Penalize wildcard actions
            action_specificity = self._calculate_action_specificity(rule.actions)

            # Reward conditions
            condition_bonus = min(0.3, len(rule.conditions) * 0.1)

            # Calculate rule score
            rule_score = (resource_specificity + action_specificity) / 2 + condition_bonus

            # Weight by priority (higher priority rules have more impact)
            weight = rule.priority.value / 100.0
            total_score += rule_score * weight

        # Normalize by total weight
        total_weight = sum(rule.priority.value / 100.0 for rule in chromosome.rules)
        if total_weight > 0:
            return min(1.0, total_score / total_weight)

        return 0.5

    def _calculate_resource_specificity(self, resource: str) -> float:
        """Calculate how specific a resource pattern is."""
        # Count wildcards
        wildcard_count = resource.count('*')
        parts = resource.split(':')

        if wildcard_count == 0:
            return 1.0  # Fully specific
        elif wildcard_count == len(parts):
            return 0.0  # All wildcards
        else:
            # Partial wildcards
            return 1.0 - (wildcard_count / len(parts))

    def _calculate_action_specificity(self, actions: List[Gene]) -> float:
        """Calculate how specific the actions are."""
        if not actions:
            return 0.0

        # Check for wildcard action
        if any(action.value == '*' for action in actions):
            return 0.0

        # Penalize too many actions
        if len(actions) > 5:
            return 0.5
        elif len(actions) > 3:
            return 0.7
        else:
            return 1.0

    def _calculate_mfa_coverage(self, chromosome: PolicyChromosome) -> float:
        """Calculate MFA coverage for sensitive resources."""
        sensitive_rules = []

        for rule in chromosome.rules:
            if self._is_sensitive_resource(rule.resource.value):
                sensitive_rules.append(rule)

        if not sensitive_rules:
            return 1.0  # No sensitive resources = full coverage

        mfa_protected = 0
        for rule in sensitive_rules:
            if self._has_mfa_condition(rule):
                mfa_protected += 1

        return mfa_protected / len(sensitive_rules)

    def _is_sensitive_resource(self, resource: str) -> bool:
        """Check if a resource is considered sensitive."""
        for pattern in self.sensitive_resources:
            if self._matches_pattern(resource, pattern):
                return True
        return False

    def _matches_pattern(self, resource: str, pattern: str) -> bool:
        """Check if resource matches a pattern (supports wildcards)."""
        # Convert wildcard pattern to regex
        regex_pattern = pattern.replace('*', '.*')
        regex_pattern = f"^{regex_pattern}$"

        return bool(re.match(regex_pattern, resource))

    def _has_mfa_condition(self, rule: PolicyRule) -> bool:
        """Check if rule has MFA-related conditions."""
        mfa_fields = ['mfa_verified', 'mfa', 'multi_factor', 'two_factor', '2fa']

        for condition in rule.conditions:
            if isinstance(condition.value, dict):
                field = condition.value.get('field', '').lower()
                if any(mfa_field in field for mfa_field in mfa_fields):
                    # Check if it's requiring MFA (not negating it)
                    operator = condition.value.get('operator')
                    value = condition.value.get('value')
                    if operator == ConditionOperator.EQUALS and value is True:
                        return True

        return False

    def _calculate_sensitive_resource_protection(self, chromosome: PolicyChromosome) -> float:
        """Calculate how well sensitive resources are protected."""
        sensitive_resources = set()
        protected_resources = set()

        for rule in chromosome.rules:
            resource = rule.resource.value
            if self._is_sensitive_resource(resource):
                sensitive_resources.add(resource)

                # Check protection level
                has_conditions = len(rule.conditions) > 0
                has_mfa = self._has_mfa_condition(rule)
                is_deny_by_default = rule.effect.value == PolicyEffect.DENY

                if has_conditions and (has_mfa or is_deny_by_default):
                    protected_resources.add(resource)

        if not sensitive_resources:
            return 1.0

        return len(protected_resources) / len(sensitive_resources)

    def _calculate_condition_strength(self, chromosome: PolicyChromosome) -> float:
        """Calculate the strength of conditions across all rules."""
        if not chromosome.rules:
            return 0.0

        total_strength = 0.0

        for rule in chromosome.rules:
            # Base score for having conditions
            if not rule.conditions:
                rule_strength = 0.0
            else:
                rule_strength = 0.5  # Base score for having any conditions

                # Bonus for multiple conditions (defense in depth)
                rule_strength += min(0.3, len(rule.conditions) * 0.1)

                # Bonus for strong condition types
                strong_conditions = self._count_strong_conditions(rule.conditions)
                rule_strength += min(0.2, strong_conditions * 0.1)

            total_strength += rule_strength

        return total_strength / len(chromosome.rules)

    def _count_strong_conditions(self, conditions: List[Gene]) -> int:
        """Count conditions considered cryptographically strong."""
        strong_fields = [
            'mfa', 'certificate', 'signed', 'encrypted', 'token',
            'clearance_level', 'security_group', 'ip_whitelist'
        ]

        count = 0
        for condition in conditions:
            if isinstance(condition.value, dict):
                field = condition.value.get('field', '').lower()
                if any(strong in field for strong in strong_fields):
                    count += 1

        return count

    def _calculate_deny_rule_presence(self, chromosome: PolicyChromosome) -> float:
        """Calculate score based on appropriate use of deny rules."""
        if not chromosome.rules:
            return 0.0

        deny_rules = [r for r in chromosome.rules if r.effect.value == PolicyEffect.DENY]

        # Ideal ratio of deny rules is around 20-30%
        deny_ratio = len(deny_rules) / len(chromosome.rules)

        if deny_ratio < 0.1:
            return 0.3  # Too few deny rules
        elif deny_ratio > 0.5:
            return 0.5  # Too many deny rules (overly restrictive)
        elif 0.2 <= deny_ratio <= 0.3:
            return 1.0  # Ideal range
        else:
            return 0.7  # Acceptable range

    def _calculate_wildcard_usage_score(self, chromosome: PolicyChromosome) -> float:
        """Calculate score based on wildcard usage (lower is better for security)."""
        if not chromosome.rules:
            return 1.0

        total_wildcards = 0
        total_resources = 0

        for rule in chromosome.rules:
            resource = rule.resource.value
            total_resources += 1

            # Count wildcards in resource
            wildcards = resource.count('*')

            # Penalize based on position and number of wildcards
            if resource == '*:*:*':
                total_wildcards += 3  # Maximum penalty for complete wildcard
            elif resource.endswith(':*:*'):
                total_wildcards += 2  # High penalty for broad wildcards
            elif resource.endswith(':*'):
                total_wildcards += 1  # Moderate penalty
            else:
                total_wildcards += wildcards * 0.5  # Lower penalty for specific wildcards

        # Calculate score (inverse of wildcard usage)
        wildcard_ratio = total_wildcards / (total_resources * 3)  # Max 3 wildcards per resource
        return 1.0 - min(1.0, wildcard_ratio)
