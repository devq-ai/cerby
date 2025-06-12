"""
Productivity fitness evaluation for policy chromosomes.

This module implements fitness functions that evaluate how well policies
support user productivity and efficient workflows.
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
class ProductivityMetrics(FitnessMetrics):
    """Detailed productivity metrics for a policy chromosome."""
    access_coverage: float
    workflow_efficiency: float
    collaboration_score: float
    tool_availability: float
    rule_simplicity: float
    response_time_impact: float


class ProductivityFitness(FitnessFunction):
    """
    Fitness function that evaluates productivity aspects of a policy.

    This function considers:
    - Access coverage for common productivity tools
    - Workflow efficiency (minimal friction)
    - Collaboration enablement
    - Tool availability without excessive restrictions
    - Rule simplicity and understandability
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize productivity fitness function.

        Args:
            config: Configuration with keys:
                - access_request_reduction_weight: Weight for reducing access requests
                - workflow_efficiency_weight: Weight for efficient workflows
                - collaboration_enablement_weight: Weight for collaboration features
                - essential_tools: List of essential productivity tools
                - collaboration_resources: Resource patterns that enable collaboration
        """
        super().__init__(config)

        # Weights for different productivity aspects
        self.weights = {
            'access_coverage': self.config.get('access_coverage_weight', 0.25),
            'workflow_efficiency': self.config.get('workflow_efficiency_weight', 0.25),
            'collaboration': self.config.get('collaboration_enablement_weight', 0.20),
            'tool_availability': self.config.get('tool_availability_weight', 0.20),
            'simplicity': self.config.get('rule_simplicity_weight', 0.10)
        }

        # Essential productivity tools
        self.essential_tools = self.config.get('essential_tools', [
            'email', 'calendar', 'slack', 'teams', 'zoom',
            'github', 'gitlab', 'jira', 'confluence', 'notion',
            'drive', 'dropbox', 'box', 'sharepoint',
            'office365', 'gsuite', 'docs', 'sheets'
        ])

        # Collaboration resource patterns
        self.collaboration_resources = self.config.get('collaboration_resources', [
            'shared:*', 'team:*', 'project:*', 'workspace:*',
            '*:shared:*', '*:collaborative:*', '*:public:*'
        ])

        # Workflow patterns that indicate good productivity
        self.workflow_patterns = self.config.get('workflow_patterns', [
            {'resources': ['jira', 'github', 'slack'], 'workflow': 'development'},
            {'resources': ['salesforce', 'slack', 'calendar'], 'workflow': 'sales'},
            {'resources': ['hr', 'payroll', 'benefits'], 'workflow': 'hr_operations'}
        ])

    def evaluate(self, chromosome: PolicyChromosome) -> float:
        """
        Evaluate the productivity fitness of a policy chromosome.

        Args:
            chromosome: The policy chromosome to evaluate

        Returns:
            Productivity fitness score between 0 and 1
        """
        metrics = self.calculate_metrics(chromosome)
        return metrics.score

    def calculate_metrics(self, chromosome: PolicyChromosome) -> ProductivityMetrics:
        """
        Calculate detailed productivity metrics for a policy chromosome.

        Args:
            chromosome: The policy chromosome to analyze

        Returns:
            Detailed productivity metrics
        """
        # Calculate individual productivity scores
        access_coverage = self._calculate_access_coverage(chromosome)
        workflow_efficiency = self._calculate_workflow_efficiency(chromosome)
        collaboration_score = self._calculate_collaboration_score(chromosome)
        tool_availability = self._calculate_tool_availability(chromosome)
        rule_simplicity = self._calculate_rule_simplicity(chromosome)
        response_impact = self._calculate_response_time_impact(chromosome)

        # Calculate weighted overall score
        overall_score = (
            access_coverage * self.weights['access_coverage'] +
            workflow_efficiency * self.weights['workflow_efficiency'] +
            collaboration_score * self.weights['collaboration'] +
            tool_availability * self.weights['tool_availability'] +
            rule_simplicity * self.weights['simplicity']
        )

        # Penalty for poor response time impact
        if response_impact < 0.5:
            overall_score *= 0.9

        return ProductivityMetrics(
            score=overall_score,
            details={
                'access_coverage': access_coverage,
                'workflow_efficiency': workflow_efficiency,
                'collaboration_score': collaboration_score,
                'tool_availability': tool_availability,
                'rule_simplicity': rule_simplicity,
                'response_impact': response_impact
            },
            access_coverage=access_coverage,
            workflow_efficiency=workflow_efficiency,
            collaboration_score=collaboration_score,
            tool_availability=tool_availability,
            rule_simplicity=rule_simplicity,
            response_time_impact=response_impact
        )

    def _calculate_access_coverage(self, chromosome: PolicyChromosome) -> float:
        """Calculate coverage of essential productivity tools."""
        if not self.essential_tools:
            return 1.0

        covered_tools = set()

        for rule in chromosome.rules:
            if rule.effect.value == PolicyEffect.ALLOW:
                resource = rule.resource.value.lower()

                # Check which essential tools are covered
                for tool in self.essential_tools:
                    if tool in resource or self._matches_tool_pattern(resource, tool):
                        covered_tools.add(tool)

        coverage = len(covered_tools) / len(self.essential_tools)

        # Bonus for comprehensive coverage
        if coverage > 0.8:
            coverage = min(1.0, coverage * 1.1)

        return coverage

    def _matches_tool_pattern(self, resource: str, tool: str) -> bool:
        """Check if a resource pattern matches a tool."""
        # Direct match
        if tool in resource:
            return True

        # Wildcard matching
        if '*' in resource:
            pattern = resource.replace('*', '.*')
            if re.match(pattern, f"{tool}:workspace:data"):
                return True

        return False

    def _calculate_workflow_efficiency(self, chromosome: PolicyChromosome) -> float:
        """Calculate workflow efficiency score."""
        if not chromosome.rules:
            return 0.0

        efficiency_score = 0.0

        # Check for workflow pattern coverage
        for workflow in self.workflow_patterns:
            required_resources = workflow['resources']
            workflow_name = workflow['workflow']

            # Check if all resources in workflow are accessible
            covered = 0
            for resource in required_resources:
                if self._is_resource_accessible(chromosome, resource):
                    covered += 1

            workflow_coverage = covered / len(required_resources)
            efficiency_score += workflow_coverage

        # Normalize by number of workflows
        if self.workflow_patterns:
            efficiency_score /= len(self.workflow_patterns)

        # Bonus for consolidated rules (fewer rules = more efficient)
        rule_efficiency = self._calculate_rule_consolidation(chromosome)
        efficiency_score = (efficiency_score * 0.7) + (rule_efficiency * 0.3)

        return efficiency_score

    def _is_resource_accessible(self, chromosome: PolicyChromosome, resource: str) -> bool:
        """Check if a resource is accessible in the chromosome."""
        for rule in chromosome.rules:
            if rule.effect.value == PolicyEffect.ALLOW:
                rule_resource = rule.resource.value.lower()
                if resource in rule_resource or self._matches_pattern(rule_resource, resource):
                    # Check if there are blocking conditions
                    if not rule.conditions or self._are_conditions_reasonable(rule.conditions):
                        return True
        return False

    def _matches_pattern(self, pattern: str, resource: str) -> bool:
        """Check if a pattern matches a resource."""
        if '*' in pattern:
            regex_pattern = pattern.replace('*', '.*')
            regex_pattern = f"^{regex_pattern}$"
            return bool(re.match(regex_pattern, resource))
        return pattern == resource

    def _are_conditions_reasonable(self, conditions: List[Gene]) -> bool:
        """Check if conditions are reasonable for productivity."""
        # Conditions that don't significantly impact productivity
        reasonable_conditions = [
            'employment_status', 'team', 'department', 'project_member',
            'active_user', 'verified_email'
        ]

        unreasonable_count = 0
        for condition in conditions:
            if isinstance(condition.value, dict):
                field = condition.value.get('field', '').lower()
                if not any(reasonable in field for reasonable in reasonable_conditions):
                    unreasonable_count += 1

        # Allow up to 1 "unreasonable" condition
        return unreasonable_count <= 1

    def _calculate_rule_consolidation(self, chromosome: PolicyChromosome) -> float:
        """Calculate how well rules are consolidated."""
        if not chromosome.rules:
            return 0.0

        # Ideal number of rules (not too many, not too few)
        ideal_rule_count = 10
        rule_count = len(chromosome.rules)

        if rule_count <= ideal_rule_count:
            consolidation_score = 1.0
        else:
            # Penalty for too many rules
            excess = rule_count - ideal_rule_count
            consolidation_score = max(0.3, 1.0 - (excess * 0.05))

        # Check for rule overlap (penalize redundancy)
        overlap_penalty = self._calculate_rule_overlap(chromosome)
        consolidation_score *= (1.0 - overlap_penalty)

        return consolidation_score

    def _calculate_rule_overlap(self, chromosome: PolicyChromosome) -> float:
        """Calculate penalty for overlapping rules."""
        overlap_count = 0
        rules = chromosome.rules

        for i in range(len(rules)):
            for j in range(i + 1, len(rules)):
                if self._rules_overlap(rules[i], rules[j]):
                    overlap_count += 1

        if len(rules) <= 1:
            return 0.0

        # Maximum possible overlaps
        max_overlaps = (len(rules) * (len(rules) - 1)) / 2
        return min(0.5, overlap_count / max_overlaps)

    def _rules_overlap(self, rule1: PolicyRule, rule2: PolicyRule) -> bool:
        """Check if two rules overlap significantly."""
        # Same effect and similar resources
        if rule1.effect.value != rule2.effect.value:
            return False

        resource1 = rule1.resource.value
        resource2 = rule2.resource.value

        # Check for resource overlap
        if resource1 == resource2:
            return True

        # Check if one is a subset of the other
        if '*' in resource1 and self._matches_pattern(resource1, resource2):
            return True
        if '*' in resource2 and self._matches_pattern(resource2, resource1):
            return True

        return False

    def _calculate_collaboration_score(self, chromosome: PolicyChromosome) -> float:
        """Calculate collaboration enablement score."""
        collab_rules = 0
        total_rules = len(chromosome.rules)

        if total_rules == 0:
            return 0.0

        for rule in chromosome.rules:
            if rule.effect.value == PolicyEffect.ALLOW:
                resource = rule.resource.value.lower()

                # Check for collaboration patterns
                is_collaborative = any(
                    pattern in resource or self._matches_pattern(pattern, resource)
                    for pattern in self.collaboration_resources
                )

                # Check for collaborative actions
                has_collab_actions = any(
                    action.value in ['share', 'comment', 'collaborate', 'invite']
                    for action in rule.actions
                )

                if is_collaborative or has_collab_actions:
                    collab_rules += 1

        base_score = collab_rules / total_rules

        # Bonus for team-based conditions
        team_condition_bonus = self._calculate_team_condition_bonus(chromosome)

        return min(1.0, base_score + team_condition_bonus)

    def _calculate_team_condition_bonus(self, chromosome: PolicyChromosome) -> float:
        """Calculate bonus for team-based access conditions."""
        team_conditions = 0
        total_conditions = 0

        for rule in chromosome.rules:
            for condition in rule.conditions:
                total_conditions += 1
                if isinstance(condition.value, dict):
                    field = condition.value.get('field', '').lower()
                    if any(team_field in field for team_field in ['team', 'project', 'group']):
                        team_conditions += 1

        if total_conditions == 0:
            return 0.0

        return min(0.2, (team_conditions / total_conditions) * 0.3)

    def _calculate_tool_availability(self, chromosome: PolicyChromosome) -> float:
        """Calculate availability of productivity tools."""
        if not chromosome.rules:
            return 0.0

        # Count rules that grant access vs restrict
        allow_rules = sum(1 for r in chromosome.rules if r.effect.value == PolicyEffect.ALLOW)
        deny_rules = sum(1 for r in chromosome.rules if r.effect.value == PolicyEffect.DENY)

        # Good ratio is about 80/20 allow/deny
        total_rules = allow_rules + deny_rules
        if total_rules == 0:
            return 0.0

        allow_ratio = allow_rules / total_rules

        if allow_ratio >= 0.8:
            availability_score = 1.0
        elif allow_ratio >= 0.6:
            availability_score = 0.8
        elif allow_ratio >= 0.4:
            availability_score = 0.5
        else:
            availability_score = 0.2

        # Check for overly restrictive conditions
        restrictive_penalty = self._calculate_restrictive_conditions(chromosome)
        availability_score *= (1.0 - restrictive_penalty)

        return availability_score

    def _calculate_restrictive_conditions(self, chromosome: PolicyChromosome) -> float:
        """Calculate penalty for overly restrictive conditions."""
        restrictive_conditions = [
            'ip_address', 'mac_address', 'specific_device', 'exact_time',
            'narrow_window', 'single_use'
        ]

        restrictive_count = 0
        total_conditions = 0

        for rule in chromosome.rules:
            if rule.effect.value == PolicyEffect.ALLOW:
                for condition in rule.conditions:
                    total_conditions += 1
                    if isinstance(condition.value, dict):
                        field = condition.value.get('field', '').lower()
                        if any(restrictive in field for restrictive in restrictive_conditions):
                            restrictive_count += 1

        if total_conditions == 0:
            return 0.0

        return min(0.5, restrictive_count / total_conditions)

    def _calculate_rule_simplicity(self, chromosome: PolicyChromosome) -> float:
        """Calculate rule simplicity and understandability."""
        if not chromosome.rules:
            return 1.0  # No rules = simple

        simplicity_scores = []

        for rule in chromosome.rules:
            # Factors that make a rule simple
            resource_simplicity = self._calculate_resource_simplicity(rule.resource.value)
            action_simplicity = self._calculate_action_simplicity(rule.actions)
            condition_simplicity = self._calculate_condition_simplicity(rule.conditions)

            rule_simplicity = (
                resource_simplicity * 0.4 +
                action_simplicity * 0.3 +
                condition_simplicity * 0.3
            )

            simplicity_scores.append(rule_simplicity)

        return sum(simplicity_scores) / len(simplicity_scores)

    def _calculate_resource_simplicity(self, resource: str) -> float:
        """Calculate simplicity of a resource specification."""
        parts = resource.split(':')

        # Ideal is 2-3 parts with clear naming
        if len(parts) == 1:
            return 0.7  # Too simple, might be unclear
        elif 2 <= len(parts) <= 3:
            return 1.0  # Ideal
        else:
            return max(0.3, 1.0 - (len(parts) - 3) * 0.2)

    def _calculate_action_simplicity(self, actions: List[Gene]) -> float:
        """Calculate simplicity of action specifications."""
        if not actions:
            return 0.5

        # Common, understandable actions
        simple_actions = ['read', 'write', 'create', 'delete', 'update', 'list']

        simple_count = sum(1 for a in actions if a.value in simple_actions)
        return simple_count / len(actions)

    def _calculate_condition_simplicity(self, conditions: List[Gene]) -> float:
        """Calculate simplicity of conditions."""
        if not conditions:
            return 1.0  # No conditions = simple

        if len(conditions) > 5:
            return 0.3  # Too many conditions
        elif len(conditions) > 3:
            return 0.6
        else:
            # Check condition types
            simple_operators = [ConditionOperator.EQUALS, ConditionOperator.IN, ConditionOperator.NOT_EQUALS]
            simple_count = 0

            for condition in conditions:
                if isinstance(condition.value, dict):
                    operator = condition.value.get('operator')
                    if operator in simple_operators:
                        simple_count += 1

            return 0.5 + (simple_count / len(conditions)) * 0.5

    def _calculate_response_time_impact(self, chromosome: PolicyChromosome) -> float:
        """Estimate impact on response time from policy evaluation."""
        if not chromosome.rules:
            return 1.0  # No rules = fast

        # Factors that slow down evaluation
        total_complexity = 0

        for rule in chromosome.rules:
            # Wildcard matching is slower
            wildcard_count = rule.resource.value.count('*')

            # Complex conditions are slower
            condition_complexity = sum(
                self._get_condition_complexity(c) for c in rule.conditions
            )

            # More actions to check = slower
            action_count = len(rule.actions)

            rule_complexity = (
                wildcard_count * 2 +
                condition_complexity +
                action_count * 0.5
            )

            total_complexity += rule_complexity

        # Normalize to 0-1 (inverse, as lower complexity = better)
        max_acceptable_complexity = 50
        normalized = min(1.0, total_complexity / max_acceptable_complexity)

        return 1.0 - normalized

    def _get_condition_complexity(self, condition: Gene) -> float:
        """Get complexity score for a condition."""
        if not isinstance(condition.value, dict):
            return 1.0

        operator = condition.value.get('operator')
        value = condition.value.get('value')

        # Complex operators
        if operator == ConditionOperator.REGEX:
            return 5.0
        elif operator in [ConditionOperator.CONTAINS, ConditionOperator.STARTS_WITH, ConditionOperator.ENDS_WITH]:
            return 3.0
        elif operator in [ConditionOperator.IN, ConditionOperator.NOT_IN]:
            if isinstance(value, list) and len(value) > 10:
                return 4.0
            return 2.0
        else:
            return 1.0
