"""
Performance fitness evaluation for policy chromosomes.

This module implements fitness functions that evaluate the performance
impact and efficiency of access control policies.
"""

from typing import Dict, Any, List, Optional
from dataclasses import dataclass
import asyncio
import time

from src.darwin.core.chromosome import (
    PolicyChromosome,
    PolicyRule,
    Gene,
    GeneType,
    PolicyEffect,
    ConditionOperator
)
from src.darwin.fitness.base import FitnessFunction, FitnessMetrics, AsyncFitnessFunction
from src.darwin.fitness.metrics import PerformanceMetrics


class PerformanceFitness(AsyncFitnessFunction):
    """
    Fitness function that evaluates performance characteristics of policies.

    This function considers:
    - Evaluation complexity (how fast can policies be evaluated)
    - Caching effectiveness (can results be cached)
    - Rule efficiency (minimal redundancy)
    - Response time estimates
    - Scalability under load
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize performance fitness function.

        Args:
            config: Configuration with keys:
                - max_evaluation_time_ms: Maximum acceptable evaluation time
                - max_rules_to_evaluate: Maximum rules before performance degrades
                - cache_effectiveness_weight: Weight for caching potential
                - complexity_penalty_factor: How much to penalize complexity
        """
        super().__init__(config)

        # Configuration
        self.max_evaluation_time_ms = self.config.get('max_evaluation_time_ms', 100)
        self.max_rules_to_evaluate = self.config.get('max_rules_to_evaluate', 50)

        # Weights
        self.weights = {
            'complexity': self.config.get('complexity_weight', 0.3),
            'caching': self.config.get('cache_effectiveness_weight', 0.2),
            'efficiency': self.config.get('rule_efficiency_weight', 0.2),
            'response_time': self.config.get('response_time_weight', 0.2),
            'scalability': self.config.get('scalability_weight', 0.1)
        }

        # Complexity factors
        self.complexity_factors = {
            'wildcard': 2.0,
            'regex': 5.0,
            'in_operator': 1.5,
            'complex_condition': 3.0,
            'many_conditions': 2.0
        }

    def evaluate(self, chromosome: PolicyChromosome) -> float:
        """
        Evaluate the performance fitness of a policy chromosome.

        Args:
            chromosome: The policy chromosome to evaluate

        Returns:
            Performance fitness score between 0 and 1
        """
        metrics = self.calculate_metrics(chromosome)
        return metrics.overall_performance

    async def evaluate_async(self, chromosome: PolicyChromosome) -> float:
        """
        Asynchronously evaluate performance with actual timing.

        Args:
            chromosome: The policy chromosome to evaluate

        Returns:
            Performance fitness score between 0 and 1
        """
        start_time = time.perf_counter()

        # Simulate policy evaluation
        await self._simulate_evaluation(chromosome)

        actual_time_ms = (time.perf_counter() - start_time) * 1000

        # Calculate time-based penalty
        if actual_time_ms <= self.max_evaluation_time_ms:
            time_score = 1.0
        else:
            excess_ratio = actual_time_ms / self.max_evaluation_time_ms
            time_score = max(0.3, 1.0 / excess_ratio)

        # Get other metrics
        metrics = self.calculate_metrics(chromosome)

        # Adjust overall score with actual timing
        adjusted_score = metrics.overall_performance * 0.7 + time_score * 0.3

        return adjusted_score

    async def _simulate_evaluation(self, chromosome: PolicyChromosome):
        """Simulate policy evaluation delay based on complexity."""
        complexity = self._calculate_evaluation_complexity(chromosome)

        # Simulate processing time (ms)
        simulated_time = complexity * 0.1
        await asyncio.sleep(simulated_time / 1000)

    def calculate_metrics(self, chromosome: PolicyChromosome) -> PerformanceMetrics:
        """
        Calculate detailed performance metrics for a policy chromosome.

        Args:
            chromosome: The policy chromosome to analyze

        Returns:
            Detailed performance metrics
        """
        # Calculate individual performance scores
        complexity_score = 1.0 - self._calculate_complexity_score(chromosome)
        caching_score = self._calculate_caching_effectiveness(chromosome)
        efficiency_score = self._calculate_rule_efficiency(chromosome)
        response_time_score = self._calculate_response_time_score(chromosome)
        scalability_score = self._calculate_scalability_score(chromosome)

        # Calculate weighted overall score
        overall_performance = (
            complexity_score * self.weights['complexity'] +
            caching_score * self.weights['caching'] +
            efficiency_score * self.weights['efficiency'] +
            response_time_score * self.weights['response_time'] +
            scalability_score * self.weights['scalability']
        )

        return PerformanceMetrics(
            evaluation_complexity=1.0 - complexity_score,
            caching_effectiveness=caching_score,
            rule_efficiency=efficiency_score,
            response_time_estimate=self._estimate_response_time(chromosome),
            scalability_score=scalability_score,
            overall_performance=overall_performance,
            details={
                'total_rules': len(chromosome.rules),
                'complex_rules': self._count_complex_rules(chromosome),
                'cacheable_rules': self._count_cacheable_rules(chromosome),
                'estimated_ops': self._estimate_operations(chromosome)
            }
        )

    def _calculate_complexity_score(self, chromosome: PolicyChromosome) -> float:
        """Calculate normalized complexity score (0-1, higher = more complex)."""
        if not chromosome.rules:
            return 0.0

        total_complexity = 0.0

        for rule in chromosome.rules:
            rule_complexity = self._calculate_rule_complexity(rule)
            total_complexity += rule_complexity

        # Normalize by expected maximum complexity
        max_expected = len(chromosome.rules) * 20  # 20 is max complexity per rule
        normalized = min(1.0, total_complexity / max_expected)

        return normalized

    def _calculate_rule_complexity(self, rule: PolicyRule) -> float:
        """Calculate complexity score for a single rule."""
        complexity = 0.0

        # Resource complexity
        resource = rule.resource.value
        wildcard_count = resource.count('*')
        complexity += wildcard_count * self.complexity_factors['wildcard']

        # Action complexity
        action_count = len(rule.actions)
        if action_count > 3:
            complexity += (action_count - 3) * 0.5

        # Condition complexity
        for condition in rule.conditions:
            if isinstance(condition.value, dict):
                operator = condition.value.get('operator')
                value = condition.value.get('value')

                if operator == ConditionOperator.REGEX:
                    complexity += self.complexity_factors['regex']
                elif operator in [ConditionOperator.IN, ConditionOperator.NOT_IN]:
                    if isinstance(value, list) and len(value) > 5:
                        complexity += self.complexity_factors['in_operator'] * (len(value) / 5)
                else:
                    complexity += 0.5

        # Many conditions multiplier
        if len(rule.conditions) > 3:
            complexity *= self.complexity_factors['many_conditions']

        return complexity

    def _calculate_caching_effectiveness(self, chromosome: PolicyChromosome) -> float:
        """Calculate how effectively rules can be cached."""
        if not chromosome.rules:
            return 1.0

        cacheable_rules = self._count_cacheable_rules(chromosome)
        return cacheable_rules / len(chromosome.rules)

    def _count_cacheable_rules(self, chromosome: PolicyChromosome) -> int:
        """Count rules that can be effectively cached."""
        cacheable = 0

        for rule in chromosome.rules:
            if self._is_cacheable(rule):
                cacheable += 1

        return cacheable

    def _is_cacheable(self, rule: PolicyRule) -> bool:
        """Check if a rule can be cached effectively."""
        # Rules with dynamic conditions cannot be cached
        dynamic_fields = ['current_time', 'random', 'session', 'request_id',
                         'timestamp', 'now', 'today']

        for condition in rule.conditions:
            if isinstance(condition.value, dict):
                field = condition.value.get('field', '').lower()
                if any(dynamic in field for dynamic in dynamic_fields):
                    return False

        # Rules with too many conditions are less cache-effective
        if len(rule.conditions) > 5:
            return False

        # Wildcard resources are less cacheable
        if rule.resource.value.count('*') > 1:
            return False

        return True

    def _calculate_rule_efficiency(self, chromosome: PolicyChromosome) -> float:
        """Calculate rule efficiency (minimal redundancy)."""
        if not chromosome.rules:
            return 1.0

        efficiency_factors = []

        # Check for redundant rules
        redundancy_score = 1.0 - self._calculate_redundancy_ratio(chromosome)
        efficiency_factors.append(redundancy_score)

        # Check for rule consolidation opportunities
        consolidation_score = self._calculate_consolidation_potential(chromosome)
        efficiency_factors.append(consolidation_score)

        # Check for optimal ordering
        ordering_score = self._calculate_ordering_efficiency(chromosome)
        efficiency_factors.append(ordering_score)

        return sum(efficiency_factors) / len(efficiency_factors)

    def _calculate_redundancy_ratio(self, chromosome: PolicyChromosome) -> float:
        """Calculate ratio of redundant rules."""
        if len(chromosome.rules) < 2:
            return 0.0

        redundant_count = 0

        for i in range(len(chromosome.rules)):
            for j in range(i + 1, len(chromosome.rules)):
                if self._rules_redundant(chromosome.rules[i], chromosome.rules[j]):
                    redundant_count += 1

        max_redundant = (len(chromosome.rules) * (len(chromosome.rules) - 1)) / 2
        return redundant_count / max_redundant

    def _rules_redundant(self, rule1: PolicyRule, rule2: PolicyRule) -> bool:
        """Check if two rules are redundant."""
        # Same effect and resource
        if (rule1.effect.value == rule2.effect.value and
            rule1.resource.value == rule2.resource.value):

            # Check if actions overlap significantly
            actions1 = set(a.value for a in rule1.actions)
            actions2 = set(a.value for a in rule2.actions)

            if actions1 == actions2:
                return True

        return False

    def _calculate_consolidation_potential(self, chromosome: PolicyChromosome) -> float:
        """Calculate potential for rule consolidation."""
        if not chromosome.rules:
            return 1.0

        # Group rules by similar patterns
        groups = {}
        for rule in chromosome.rules:
            pattern = self._get_rule_pattern(rule)
            if pattern not in groups:
                groups[pattern] = []
            groups[pattern].append(rule)

        # Rules in same group could potentially be consolidated
        consolidatable = sum(len(group) - 1 for group in groups.values() if len(group) > 1)

        if len(chromosome.rules) > 1:
            return 1.0 - (consolidatable / len(chromosome.rules))
        return 1.0

    def _get_rule_pattern(self, rule: PolicyRule) -> str:
        """Get pattern signature for grouping similar rules."""
        resource_base = rule.resource.value.split(':')[0] if ':' in rule.resource.value else rule.resource.value
        effect = rule.effect.value
        has_conditions = len(rule.conditions) > 0
        return f"{resource_base}:{effect}:{has_conditions}"

    def _calculate_ordering_efficiency(self, chromosome: PolicyChromosome) -> float:
        """Calculate efficiency of rule ordering."""
        if not chromosome.rules:
            return 1.0

        # Higher priority rules should be more specific (evaluated first)
        efficiency_score = 0.0

        for i, rule in enumerate(chromosome.rules):
            priority = rule.priority.value
            specificity = self._calculate_rule_specificity(rule)

            # Higher priority should correlate with higher specificity
            if priority >= 80 and specificity >= 0.7:
                efficiency_score += 1.0
            elif priority <= 30 and specificity <= 0.3:
                efficiency_score += 1.0
            elif 30 < priority < 80 and 0.3 < specificity < 0.7:
                efficiency_score += 1.0
            else:
                efficiency_score += 0.5

        return efficiency_score / len(chromosome.rules)

    def _calculate_rule_specificity(self, rule: PolicyRule) -> float:
        """Calculate how specific a rule is (0-1, higher = more specific)."""
        specificity = 1.0

        # Wildcards reduce specificity
        wildcard_count = rule.resource.value.count('*')
        specificity -= wildcard_count * 0.3

        # Conditions increase specificity
        specificity += min(0.3, len(rule.conditions) * 0.1)

        # Specific actions increase specificity
        if not any(a.value == '*' for a in rule.actions):
            specificity += 0.1

        return max(0, min(1.0, specificity))

    def _calculate_response_time_score(self, chromosome: PolicyChromosome) -> float:
        """Calculate response time score based on estimated evaluation time."""
        estimated_ms = self._estimate_response_time(chromosome)

        if estimated_ms <= self.max_evaluation_time_ms:
            return 1.0
        elif estimated_ms <= self.max_evaluation_time_ms * 2:
            return 0.7
        elif estimated_ms <= self.max_evaluation_time_ms * 3:
            return 0.5
        else:
            return 0.3

    def _estimate_response_time(self, chromosome: PolicyChromosome) -> float:
        """Estimate response time in milliseconds."""
        if not chromosome.rules:
            return 0.0

        base_time = 1.0  # Base overhead

        for rule in chromosome.rules:
            # Each rule evaluation has overhead
            rule_time = 0.5

            # Add complexity-based time
            complexity = self._calculate_rule_complexity(rule)
            rule_time += complexity * 0.2

            base_time += rule_time

        return base_time

    def _calculate_scalability_score(self, chromosome: PolicyChromosome) -> float:
        """Calculate how well the policy scales with increased load."""
        factors = []

        # Rule count scalability
        if len(chromosome.rules) <= self.max_rules_to_evaluate:
            rule_scale = 1.0
        else:
            excess = len(chromosome.rules) - self.max_rules_to_evaluate
            rule_scale = max(0.3, 1.0 - (excess / self.max_rules_to_evaluate))
        factors.append(rule_scale)

        # Complexity scalability
        avg_complexity = sum(self._calculate_rule_complexity(r) for r in chromosome.rules) / max(1, len(chromosome.rules))
        if avg_complexity <= 5:
            complexity_scale = 1.0
        elif avg_complexity <= 10:
            complexity_scale = 0.7
        else:
            complexity_scale = 0.4
        factors.append(complexity_scale)

        # Caching benefit
        cache_ratio = self._count_cacheable_rules(chromosome) / max(1, len(chromosome.rules))
        factors.append(cache_ratio)

        return sum(factors) / len(factors)

    def _count_complex_rules(self, chromosome: PolicyChromosome) -> int:
        """Count rules considered complex from a performance perspective."""
        complex_count = 0

        for rule in chromosome.rules:
            complexity = self._calculate_rule_complexity(rule)
            if complexity > 10:  # Threshold for "complex"
                complex_count += 1

        return complex_count

    def _estimate_operations(self, chromosome: PolicyChromosome) -> int:
        """Estimate number of operations needed to evaluate all rules."""
        total_ops = 0

        for rule in chromosome.rules:
            # Base operation for rule check
            ops = 1

            # Resource matching operations
            wildcard_count = rule.resource.value.count('*')
            ops += wildcard_count * 2

            # Action checks
            ops += len(rule.actions)

            # Condition evaluations
            for condition in rule.conditions:
                if isinstance(condition.value, dict):
                    operator = condition.value.get('operator')
                    if operator == ConditionOperator.REGEX:
                        ops += 10
                    elif operator in [ConditionOperator.IN, ConditionOperator.NOT_IN]:
                        value = condition.value.get('value')
                        if isinstance(value, list):
                            ops += len(value)
                    else:
                        ops += 1

            total_ops += ops

        return total_ops

    async def evaluate_batch_async(self, chromosomes: List[PolicyChromosome]) -> List[float]:
        """
        Evaluate multiple chromosomes in parallel for performance testing.

        Args:
            chromosomes: List of chromosomes to evaluate

        Returns:
            List of performance scores
        """
        tasks = [self.evaluate_async(c) for c in chromosomes]
        return await asyncio.gather(*tasks)
