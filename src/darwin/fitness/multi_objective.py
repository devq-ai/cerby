"""
Multi-objective fitness evaluation for policy chromosomes.

This module implements fitness functions that combine multiple objectives
and handle Pareto-optimal solutions.
"""

from typing import Dict, Any, List, Optional, Tuple, Set
from dataclasses import dataclass
import numpy as np

from src.darwin.core.chromosome import PolicyChromosome
from src.darwin.fitness.base import FitnessFunction, FitnessMetrics, CompositeFitnessFunction
from src.darwin.fitness.metrics import MultiObjectiveMetrics


@dataclass
class ObjectiveWeight:
    """Configuration for a single objective in multi-objective optimization."""
    function: FitnessFunction
    weight: float
    name: str
    minimize: bool = False  # If True, lower values are better


@dataclass
class ParetoFront:
    """Represents a Pareto front in multi-objective optimization."""
    solutions: List[PolicyChromosome]
    objective_values: List[Dict[str, float]]

    def add_solution(self, chromosome: PolicyChromosome, objectives: Dict[str, float]):
        """Add a solution to the Pareto front."""
        self.solutions.append(chromosome)
        self.objective_values.append(objectives)

    def get_size(self) -> int:
        """Get the number of solutions in the Pareto front."""
        return len(self.solutions)


class MultiObjectiveFitness(CompositeFitnessFunction):
    """
    Multi-objective fitness function that combines multiple objectives.

    Supports:
    - Weighted sum aggregation
    - Pareto dominance checking
    - Objective conflict detection
    - Trade-off analysis
    """

    def __init__(self, objectives: Dict[str, Dict[str, Any]]):
        """
        Initialize multi-objective fitness function.

        Args:
            objectives: Dictionary mapping objective names to configurations.
                       Each should have 'function' and 'weight' keys.
        """
        super().__init__(objectives)
        self.pareto_front = ParetoFront([], [])

    def evaluate_objectives(self, chromosome: PolicyChromosome) -> Dict[str, float]:
        """
        Evaluate all objectives independently.

        Args:
            chromosome: The policy chromosome to evaluate

        Returns:
            Dictionary mapping objective names to scores
        """
        objective_scores = {}

        for name, config in self.objectives.items():
            function = config['function']
            score = function.evaluate(chromosome)
            objective_scores[name] = score

        return objective_scores

    def dominates(self, chromosome1: PolicyChromosome, chromosome2: PolicyChromosome) -> bool:
        """
        Check if chromosome1 Pareto-dominates chromosome2.

        Args:
            chromosome1: First chromosome
            chromosome2: Second chromosome

        Returns:
            True if chromosome1 dominates chromosome2
        """
        objectives1 = self.evaluate_objectives(chromosome1)
        objectives2 = self.evaluate_objectives(chromosome2)

        at_least_one_better = False

        for name in objectives1:
            if objectives1[name] < objectives2[name]:
                return False  # chromosome2 is better in this objective
            elif objectives1[name] > objectives2[name]:
                at_least_one_better = True

        return at_least_one_better

    def is_non_dominated(self, chromosome: PolicyChromosome,
                        population: List[PolicyChromosome]) -> bool:
        """
        Check if a chromosome is non-dominated in a population.

        Args:
            chromosome: The chromosome to check
            population: The population to check against

        Returns:
            True if the chromosome is non-dominated
        """
        for other in population:
            if other.chromosome_id != chromosome.chromosome_id:
                if self.dominates(other, chromosome):
                    return False
        return True

    def update_pareto_front(self, population: List[PolicyChromosome]):
        """
        Update the Pareto front with non-dominated solutions.

        Args:
            population: Current population of chromosomes
        """
        non_dominated = []
        objective_values = []

        for chromosome in population:
            if self.is_non_dominated(chromosome, population):
                non_dominated.append(chromosome)
                objectives = self.evaluate_objectives(chromosome)
                objective_values.append(objectives)

        self.pareto_front = ParetoFront(non_dominated, objective_values)

    def calculate_crowding_distance(self, population: List[PolicyChromosome]) -> Dict[str, float]:
        """
        Calculate crowding distance for diversity preservation.

        Args:
            population: Population of chromosomes

        Returns:
            Dictionary mapping chromosome IDs to crowding distances
        """
        distances = {c.chromosome_id: 0.0 for c in population}

        if len(population) <= 2:
            # Boundary solutions get infinite distance
            for c in population:
                distances[c.chromosome_id] = float('inf')
            return distances

        # Calculate for each objective
        for obj_name in self.objectives:
            # Sort by objective value
            sorted_pop = sorted(
                population,
                key=lambda c: self.evaluate_objectives(c)[obj_name]
            )

            # Boundary solutions
            distances[sorted_pop[0].chromosome_id] = float('inf')
            distances[sorted_pop[-1].chromosome_id] = float('inf')

            # Calculate distances for intermediate solutions
            obj_range = (
                self.evaluate_objectives(sorted_pop[-1])[obj_name] -
                self.evaluate_objectives(sorted_pop[0])[obj_name]
            )

            if obj_range > 0:
                for i in range(1, len(sorted_pop) - 1):
                    prev_val = self.evaluate_objectives(sorted_pop[i-1])[obj_name]
                    next_val = self.evaluate_objectives(sorted_pop[i+1])[obj_name]
                    distances[sorted_pop[i].chromosome_id] += (next_val - prev_val) / obj_range

        return distances

    def detect_objective_conflicts(self, chromosome: PolicyChromosome,
                                 threshold: float = 0.3) -> List[Tuple[str, str]]:
        """
        Detect conflicting objectives.

        Args:
            chromosome: Chromosome to analyze
            threshold: Threshold for considering objectives conflicting

        Returns:
            List of conflicting objective pairs
        """
        objectives = self.evaluate_objectives(chromosome)
        conflicts = []

        objective_names = list(objectives.keys())
        for i in range(len(objective_names)):
            for j in range(i + 1, len(objective_names)):
                obj1, obj2 = objective_names[i], objective_names[j]

                # Check if improving one degrades the other significantly
                # This is a simplified check - in practice, you'd analyze the gradient
                if abs(objectives[obj1] - objectives[obj2]) > threshold:
                    if (objectives[obj1] > 0.7 and objectives[obj2] < 0.3) or \
                       (objectives[obj1] < 0.3 and objectives[obj2] > 0.7):
                        conflicts.append((obj1, obj2))

        return conflicts

    def calculate_metrics(self, chromosome: PolicyChromosome) -> MultiObjectiveMetrics:
        """
        Calculate detailed multi-objective metrics.

        Args:
            chromosome: The policy chromosome to analyze

        Returns:
            Multi-objective metrics
        """
        # Get individual objective scores
        objective_scores = self.evaluate_objectives(chromosome)

        # Calculate weighted total
        weighted_total = sum(
            score * self.objectives[name]['weight']
            for name, score in objective_scores.items()
        )

        # Calculate Pareto rank (simplified - in practice use NSGA-II ranking)
        pareto_rank = self._calculate_pareto_rank(chromosome, [chromosome])

        # Count dominance relationships
        dominance_count = 0
        dominated_by_count = 0

        # Detect conflicts
        conflicts = self.detect_objective_conflicts(chromosome)

        return MultiObjectiveMetrics(
            objective_scores=objective_scores,
            weighted_total=weighted_total,
            pareto_rank=pareto_rank,
            dominance_count=dominance_count,
            dominated_by_count=dominated_by_count,
            details={
                'conflicts': conflicts,
                'pareto_front_size': self.pareto_front.get_size(),
                'objective_weights': {name: obj['weight']
                                    for name, obj in self.objectives.items()}
            }
        )

    def _calculate_pareto_rank(self, chromosome: PolicyChromosome,
                              population: List[PolicyChromosome]) -> int:
        """
        Calculate Pareto rank of a chromosome.

        Args:
            chromosome: Chromosome to rank
            population: Population to compare against

        Returns:
            Pareto rank (0 = non-dominated)
        """
        dominated_by = 0

        for other in population:
            if other.chromosome_id != chromosome.chromosome_id:
                if self.dominates(other, chromosome):
                    dominated_by += 1

        return dominated_by

    def get_compromise_solution(self, pareto_front: Optional[ParetoFront] = None) -> Optional[PolicyChromosome]:
        """
        Find a compromise solution from the Pareto front.

        Uses the solution closest to the ideal point.

        Args:
            pareto_front: Pareto front to search (uses internal if None)

        Returns:
            Best compromise solution or None if front is empty
        """
        front = pareto_front or self.pareto_front

        if not front.solutions:
            return None

        # Find ideal point (best value for each objective)
        ideal_point = {}
        for obj_name in self.objectives:
            ideal_point[obj_name] = max(
                obj_vals[obj_name] for obj_vals in front.objective_values
            )

        # Find solution closest to ideal point (normalized Euclidean distance)
        best_solution = None
        best_distance = float('inf')

        for i, solution in enumerate(front.solutions):
            distance = 0.0
            obj_values = front.objective_values[i]

            for obj_name, ideal_val in ideal_point.items():
                # Normalize by ideal value to handle different scales
                if ideal_val > 0:
                    normalized_diff = (ideal_val - obj_values[obj_name]) / ideal_val
                    distance += normalized_diff ** 2

            distance = np.sqrt(distance)

            if distance < best_distance:
                best_distance = distance
                best_solution = solution

        return best_solution

    def visualize_trade_offs(self, population: List[PolicyChromosome]) -> Dict[str, Any]:
        """
        Prepare data for trade-off visualization.

        Args:
            population: Population to analyze

        Returns:
            Visualization data
        """
        objective_data = []

        for chromosome in population:
            objectives = self.evaluate_objectives(chromosome)
            objectives['id'] = chromosome.chromosome_id
            objectives['is_pareto'] = self.is_non_dominated(chromosome, population)
            objective_data.append(objectives)

        return {
            'data': objective_data,
            'objectives': list(self.objectives.keys()),
            'pareto_front_size': sum(1 for d in objective_data if d['is_pareto'])
        }
