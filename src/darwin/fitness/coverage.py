"""
Policy coverage fitness evaluation for policy chromosomes.

This module implements fitness functions that evaluate how well policies
cover required resources, actions, and conditions.
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
from src.darwin.fitness.metrics import CoverageMetrics


class PolicyCoverageFitness(FitnessFunction):
    """
    Fitness function that evaluates policy coverage.

    This function considers:
    - Resource coverage (are all required resources accessible?)
    - Action coverage (are all required actions allowed?)
    - Condition coverage (are security conditions properly applied?)
    - User coverage (what percentage of users have appropriate access?)
    """

    def __init__(self, required_resources: List[str],
                 required_actions: Optional[List[str]] = None,
                 security_conditions: Optional[List[str]] = None,
                 config: Optional[Dict[str, Any]] = None):
        """
        Initialize policy coverage fitness function.

        Args:
            required_resources: List of resource patterns that should be covered
            required_actions: List of actions that should be available
            security_conditions: List of security conditions that should be present
            config: Additional configuration parameters
        """
        super().__init__(config)

        self.required_resources = required_resources
        self.required_actions = required_actions or ['read', 'write', 'create', 'delete']
        self.security_conditions = security_conditions or []

        # Weights for different coverage aspects
        self.weights = {
            'resource': self.config.get('resource_weight', 0.4),
            'action': self.config.get('action_weight', 0.3),
            'condition': self.config.get('condition_weight', 0.2),
            'user': self.config.get('user_weight', 0.1)
        }

    def evaluate(self, chromosome: PolicyChromosome) -> float:
        """
        Evaluate the coverage fitness of a policy chromosome.

        Args:
            chromosome: The policy chromosome to evaluate

        Returns:
            Coverage fitness score between 0 and 1
        """
        metrics = self.calculate_metrics(chromosome)
        return metrics.overall_coverage

    def calculate_metrics(self, chromosome: PolicyChromosome) -> CoverageMetrics:
        """
        Calculate detailed coverage metrics for a policy chromosome.

        Args:
            chromosome: The policy chromosome to analyze

        Returns:
            Detailed coverage metrics
        """
        # Calculate individual coverage scores
        resource_coverage = self._calculate_resource_coverage(chromosome)
        action_coverage = self._calculate_action_coverage(chromosome)
        condition_coverage = self._calculate_condition_coverage(chromosome)
        user_coverage = self._calculate_user_coverage(chromosome)

        # Identify uncovered resources
        uncovered = self._identify_uncovered_resources(chromosome)

        # Calculate weighted overall score
        overall_coverage = (
            resource_coverage * self.weights['resource'] +
            action_coverage * self.weights['action'] +
            condition_coverage * self.weights['condition'] +
            user_coverage * self.weights['user']
        )

        return CoverageMetrics(
            resource_coverage=resource_coverage,
            action_coverage=action_coverage,
            condition_coverage=condition_coverage,
            user_coverage=user_coverage,
            overall_coverage=overall_coverage,
            uncovered_resources=uncovered,
            details={
                'total_required_resources': len(self.required_resources),
                'total_required_actions': len(self.required_actions),
                'total_security_conditions': len(self.security_conditions),
                'rules_analyzed': len(chromosome.rules)
            }
        )

    def _calculate_resource_coverage(self, chromosome: PolicyChromosome) -> float:
        """Calculate coverage of required resources."""
        if not self.required_resources:
            return 1.0

        covered_resources = set()

        for required in self.required_resources:
            for rule in chromosome.rules:
                if rule.effect.value == PolicyEffect.ALLOW:
                    rule_resource = rule.resource.value

                    # Check if rule covers the required resource
                    if self._resource_matches(rule_resource, required):
                        covered_resources.add(required)
                        break

        return len(covered_resources) / len(self.required_resources)

    def _resource_matches(self, rule_resource: str, required_resource: str) -> bool:
        """Check if a rule resource pattern matches a required resource."""
        # Direct match
        if rule_resource == required_resource:
            return True

        # Check if rule resource is more general (covers required)
        if '*' in rule_resource:
            pattern = rule_resource.replace('*', '.*')
            pattern = f"^{pattern}$"
            if re.match(pattern, required_resource):
                return True

        # Check if required resource is a pattern that rule matches
        if '*' in required_resource:
            pattern = required_resource.replace('*', '.*')
            pattern = f"^{pattern}$"
            if re.match(pattern, rule_resource):
                return True

        return False

    def _calculate_action_coverage(self, chromosome: PolicyChromosome) -> float:
        """Calculate coverage of required actions."""
        if not self.required_actions:
            return 1.0

        action_coverage_by_resource = {}

        # For each required resource, check which actions are covered
        for required_resource in self.required_resources:
            covered_actions = set()

            for rule in chromosome.rules:
                if rule.effect.value == PolicyEffect.ALLOW:
                    if self._resource_matches(rule.resource.value, required_resource):
                        # Add all actions from this rule
                        for action in rule.actions:
                            if action.value in self.required_actions:
                                covered_actions.add(action.value)

            # Calculate coverage for this resource
            if self.required_actions:
                coverage = len(covered_actions) / len(self.required_actions)
            else:
                coverage = 1.0

            action_coverage_by_resource[required_resource] = coverage

        # Return average coverage across all resources
        if action_coverage_by_resource:
            return sum(action_coverage_by_resource.values()) / len(action_coverage_by_resource)
        return 0.0

    def _calculate_condition_coverage(self, chromosome: PolicyChromosome) -> float:
        """Calculate coverage of security conditions."""
        if not self.security_conditions:
            return 1.0

        # Count rules that have security conditions
        rules_with_security_conditions = 0
        total_allow_rules = 0

        for rule in chromosome.rules:
            if rule.effect.value == PolicyEffect.ALLOW:
                total_allow_rules += 1

                # Check if rule has any of the required security conditions
                has_security_condition = False
                for condition in rule.conditions:
                    if isinstance(condition.value, dict):
                        field = condition.value.get('field', '').lower()
                        if any(sec_cond in field for sec_cond in self.security_conditions):
                            has_security_condition = True
                            break

                if has_security_condition:
                    rules_with_security_conditions += 1

        if total_allow_rules == 0:
            return 0.0

        # Base score on percentage of rules with security conditions
        base_score = rules_with_security_conditions / total_allow_rules

        # Bonus for covering all required conditions
        condition_types_found = self._count_condition_types(chromosome)
        if condition_types_found >= len(self.security_conditions):
            base_score = min(1.0, base_score * 1.2)

        return base_score

    def _count_condition_types(self, chromosome: PolicyChromosome) -> int:
        """Count unique types of security conditions present."""
        found_conditions = set()

        for rule in chromosome.rules:
            for condition in rule.conditions:
                if isinstance(condition.value, dict):
                    field = condition.value.get('field', '').lower()
                    for sec_cond in self.security_conditions:
                        if sec_cond in field:
                            found_conditions.add(sec_cond)

        return len(found_conditions)

    def _calculate_user_coverage(self, chromosome: PolicyChromosome) -> float:
        """Calculate estimated user coverage."""
        # This is a simplified calculation based on condition breadth
        if not chromosome.rules:
            return 0.0

        coverage_scores = []

        for rule in chromosome.rules:
            if rule.effect.value == PolicyEffect.ALLOW:
                # Rules with fewer conditions cover more users
                if len(rule.conditions) == 0:
                    coverage_scores.append(1.0)  # Covers all users
                else:
                    # Estimate coverage based on condition restrictiveness
                    restrictiveness = self._estimate_condition_restrictiveness(rule.conditions)
                    coverage_scores.append(1.0 - restrictiveness)

        if not coverage_scores:
            return 0.0

        return sum(coverage_scores) / len(coverage_scores)

    def _estimate_condition_restrictiveness(self, conditions: List[Gene]) -> float:
        """Estimate how restrictive a set of conditions is."""
        if not conditions:
            return 0.0

        restrictiveness = 0.0

        for condition in conditions:
            if isinstance(condition.value, dict):
                operator = condition.value.get('operator')
                value = condition.value.get('value')

                # Specific user/group conditions are very restrictive
                field = condition.value.get('field', '').lower()
                if any(specific in field for specific in ['user_id', 'user_name', 'specific_user']):
                    restrictiveness += 0.9
                elif any(group in field for group in ['group', 'team', 'department']):
                    restrictiveness += 0.5
                elif any(attr in field for attr in ['role', 'level', 'clearance']):
                    restrictiveness += 0.3
                else:
                    restrictiveness += 0.1

                # Operator type affects restrictiveness
                if operator == ConditionOperator.EQUALS:
                    restrictiveness += 0.1
                elif operator in [ConditionOperator.IN, ConditionOperator.NOT_IN]:
                    if isinstance(value, list):
                        # Larger lists are less restrictive
                        restrictiveness += 0.05 * (1.0 / max(1, len(value)))

        # Normalize by number of conditions
        return min(1.0, restrictiveness / len(conditions))

    def _identify_uncovered_resources(self, chromosome: PolicyChromosome) -> List[str]:
        """Identify which required resources are not covered."""
        uncovered = []

        for required in self.required_resources:
            covered = False

            for rule in chromosome.rules:
                if rule.effect.value == PolicyEffect.ALLOW:
                    if self._resource_matches(rule.resource.value, required):
                        covered = True
                        break

            if not covered:
                uncovered.append(required)

        return uncovered

    def get_coverage_gaps(self, chromosome: PolicyChromosome) -> Dict[str, Any]:
        """
        Get detailed analysis of coverage gaps.

        Args:
            chromosome: The policy chromosome to analyze

        Returns:
            Dictionary describing coverage gaps
        """
        gaps = {
            'uncovered_resources': self._identify_uncovered_resources(chromosome),
            'missing_actions_by_resource': {},
            'resources_without_conditions': [],
            'low_user_coverage_rules': []
        }

        # Analyze missing actions by resource
        for required_resource in self.required_resources:
            covered_actions = set()

            for rule in chromosome.rules:
                if rule.effect.value == PolicyEffect.ALLOW:
                    if self._resource_matches(rule.resource.value, required_resource):
                        for action in rule.actions:
                            covered_actions.add(action.value)

            missing_actions = set(self.required_actions) - covered_actions
            if missing_actions:
                gaps['missing_actions_by_resource'][required_resource] = list(missing_actions)

        # Find resources without security conditions
        for rule in chromosome.rules:
            if rule.effect.value == PolicyEffect.ALLOW and len(rule.conditions) == 0:
                gaps['resources_without_conditions'].append(rule.resource.value)

        # Identify rules with low user coverage
        for rule in chromosome.rules:
            if rule.effect.value == PolicyEffect.ALLOW:
                restrictiveness = self._estimate_condition_restrictiveness(rule.conditions)
                if restrictiveness > 0.7:  # Very restrictive
                    gaps['low_user_coverage_rules'].append({
                        'resource': rule.resource.value,
                        'restrictiveness': restrictiveness,
                        'conditions': len(rule.conditions)
                    })

        return gaps

    def suggest_improvements(self, chromosome: PolicyChromosome) -> List[str]:
        """
        Suggest improvements to increase coverage.

        Args:
            chromosome: The policy chromosome to analyze

        Returns:
            List of improvement suggestions
        """
        suggestions = []
        gaps = self.get_coverage_gaps(chromosome)

        # Suggest adding rules for uncovered resources
        for resource in gaps['uncovered_resources']:
            suggestions.append(f"Add rule to allow access to resource: {resource}")

        # Suggest adding missing actions
        for resource, missing_actions in gaps['missing_actions_by_resource'].items():
            suggestions.append(
                f"Add actions {missing_actions} for resource: {resource}"
            )

        # Suggest adding security conditions
        if gaps['resources_without_conditions']:
            suggestions.append(
                "Add security conditions to rules without any restrictions: " +
                ", ".join(gaps['resources_without_conditions'][:3])
            )

        # Suggest broadening overly restrictive rules
        if gaps['low_user_coverage_rules']:
            suggestions.append(
                "Consider broadening conditions for rules with high restrictiveness"
            )

        return suggestions
