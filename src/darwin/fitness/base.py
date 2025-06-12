"""
Base classes for fitness evaluation in the Darwin genetic algorithm framework.

This module provides abstract base classes and common functionality for
implementing fitness functions that evaluate policy chromosomes.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
import asyncio

from src.darwin.core.chromosome import PolicyChromosome


@dataclass
class FitnessMetrics:
    """Base class for fitness metrics."""
    score: float  # Overall fitness score [0, 1]
    details: Dict[str, Any]  # Detailed breakdown of the score


class FitnessFunction(ABC):
    """
    Abstract base class for fitness functions.

    All fitness functions should inherit from this class and implement
    the evaluate method to score policy chromosomes.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize fitness function with optional configuration.

        Args:
            config: Configuration parameters for the fitness function
        """
        self.config = config or {}

    @abstractmethod
    def evaluate(self, chromosome: PolicyChromosome) -> float:
        """
        Evaluate a policy chromosome and return a fitness score.

        Args:
            chromosome: The policy chromosome to evaluate

        Returns:
            Fitness score between 0 and 1, where 1 is optimal
        """
        pass

    @abstractmethod
    def calculate_metrics(self, chromosome: PolicyChromosome) -> FitnessMetrics:
        """
        Calculate detailed metrics for a policy chromosome.

        Args:
            chromosome: The policy chromosome to analyze

        Returns:
            Detailed fitness metrics including score and breakdown
        """
        pass

    def normalize_score(self, value: float, min_val: float = 0, max_val: float = 1) -> float:
        """
        Normalize a value to [0, 1] range.

        Args:
            value: Value to normalize
            min_val: Minimum expected value
            max_val: Maximum expected value

        Returns:
            Normalized value between 0 and 1
        """
        if max_val == min_val:
            return 0.5

        normalized = (value - min_val) / (max_val - min_val)
        return max(0, min(1, normalized))


class AsyncFitnessFunction(FitnessFunction):
    """
    Base class for fitness functions that support asynchronous evaluation.

    This is useful for fitness functions that need to query external
    services or perform expensive computations.
    """

    async def evaluate_async(self, chromosome: PolicyChromosome) -> float:
        """
        Asynchronously evaluate a policy chromosome.

        Args:
            chromosome: The policy chromosome to evaluate

        Returns:
            Fitness score between 0 and 1
        """
        # Default implementation delegates to sync version
        return self.evaluate(chromosome)

    async def evaluate_batch_async(self, chromosomes: List[PolicyChromosome]) -> List[float]:
        """
        Asynchronously evaluate a batch of chromosomes.

        Args:
            chromosomes: List of chromosomes to evaluate

        Returns:
            List of fitness scores
        """
        tasks = [self.evaluate_async(c) for c in chromosomes]
        return await asyncio.gather(*tasks)


class CompositeFitnessFunction(FitnessFunction):
    """
    Base class for composite fitness functions that combine multiple objectives.
    """

    def __init__(self, objectives: Dict[str, Dict[str, Any]]):
        """
        Initialize composite fitness function.

        Args:
            objectives: Dictionary mapping objective names to their configurations
                       Each entry should have 'function' and 'weight' keys
        """
        super().__init__()
        self.objectives = objectives

        # Normalize weights
        total_weight = sum(obj.get('weight', 1.0) for obj in objectives.values())
        for obj in objectives.values():
            obj['weight'] = obj.get('weight', 1.0) / total_weight

    def evaluate(self, chromosome: PolicyChromosome) -> float:
        """
        Evaluate chromosome using weighted combination of objectives.

        Args:
            chromosome: The policy chromosome to evaluate

        Returns:
            Weighted average fitness score
        """
        total_score = 0.0

        for name, objective in self.objectives.items():
            function = objective['function']
            weight = objective['weight']
            score = function.evaluate(chromosome)
            total_score += score * weight

        return total_score

    def evaluate_with_breakdown(self, chromosome: PolicyChromosome) -> Tuple[float, Dict[str, float]]:
        """
        Evaluate chromosome and return breakdown by objective.

        Args:
            chromosome: The policy chromosome to evaluate

        Returns:
            Tuple of (overall_score, objective_scores)
        """
        breakdown = {}
        total_score = 0.0

        for name, objective in self.objectives.items():
            function = objective['function']
            weight = objective['weight']
            score = function.evaluate(chromosome)
            breakdown[name] = score
            total_score += score * weight

        return total_score, breakdown

    def calculate_metrics(self, chromosome: PolicyChromosome) -> FitnessMetrics:
        """
        Calculate detailed metrics for all objectives.

        Args:
            chromosome: The policy chromosome to analyze

        Returns:
            Composite fitness metrics
        """
        overall_score, breakdown = self.evaluate_with_breakdown(chromosome)

        detailed_metrics = {}
        for name, objective in self.objectives.items():
            function = objective['function']
            metrics = function.calculate_metrics(chromosome)
            detailed_metrics[name] = metrics

        return FitnessMetrics(
            score=overall_score,
            details={
                'breakdown': breakdown,
                'objective_metrics': detailed_metrics
            }
        )


class CachedFitnessFunction(FitnessFunction):
    """
    Decorator class that adds caching to fitness functions.
    """

    def __init__(self, fitness_function: FitnessFunction, cache_size: int = 1000):
        """
        Initialize cached fitness function.

        Args:
            fitness_function: The fitness function to wrap
            cache_size: Maximum number of evaluations to cache
        """
        super().__init__(fitness_function.config)
        self.fitness_function = fitness_function
        self.cache_size = cache_size
        self.cache: Dict[str, float] = {}
        self.access_order: List[str] = []

    def _get_cache_key(self, chromosome: PolicyChromosome) -> str:
        """Generate cache key for chromosome."""
        return chromosome.chromosome_id

    def evaluate(self, chromosome: PolicyChromosome) -> float:
        """
        Evaluate with caching.

        Args:
            chromosome: The policy chromosome to evaluate

        Returns:
            Cached or computed fitness score
        """
        cache_key = self._get_cache_key(chromosome)

        if cache_key in self.cache:
            # Move to end (LRU)
            self.access_order.remove(cache_key)
            self.access_order.append(cache_key)
            return self.cache[cache_key]

        # Compute score
        score = self.fitness_function.evaluate(chromosome)

        # Add to cache
        self.cache[cache_key] = score
        self.access_order.append(cache_key)

        # Evict oldest if cache full
        if len(self.cache) > self.cache_size:
            oldest = self.access_order.pop(0)
            del self.cache[oldest]

        return score

    def calculate_metrics(self, chromosome: PolicyChromosome) -> FitnessMetrics:
        """Pass through to wrapped function."""
        return self.fitness_function.calculate_metrics(chromosome)

    def clear_cache(self):
        """Clear the evaluation cache."""
        self.cache.clear()
        self.access_order.clear()
