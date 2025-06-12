"""
Population Management for Darwin Genetic Algorithm.

This module manages populations of individuals (chromosomes) throughout
the evolution process, including initialization, selection, and diversity tracking.
"""

from typing import List, Optional, Dict, Any, Tuple, Callable
from dataclasses import dataclass, field
import random
import statistics
import numpy as np
from collections import defaultdict
import json
import pickle
from datetime import datetime

from src.darwin.core.chromosome import PolicyChromosome, PolicyRule
from src.darwin.core.config import DarwinConfig


@dataclass
class Individual:
    """
    Represents an individual in the population.

    An individual wraps a chromosome and tracks additional metadata
    such as fitness scores, age, and lineage.
    """

    chromosome: PolicyChromosome
    fitness: Optional[float] = None
    multi_objective_fitness: Dict[str, float] = field(default_factory=dict)
    age: int = 0
    parent_ids: List[str] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)
    evaluated: bool = False

    @property
    def id(self) -> str:
        """Get the individual's unique identifier."""
        return self.chromosome.chromosome_id

    def update_fitness(self, fitness: float, objectives: Optional[Dict[str, float]] = None) -> None:
        """Update the individual's fitness scores."""
        self.fitness = fitness
        if objectives:
            self.multi_objective_fitness = objectives
        self.evaluated = True
        self.chromosome.fitness_scores = objectives or {}

    def increment_age(self) -> None:
        """Increment the individual's age by one generation."""
        self.age += 1

    def to_dict(self) -> Dict[str, Any]:
        """Convert individual to dictionary representation."""
        return {
            "chromosome": self.chromosome.to_dict(),
            "fitness": self.fitness,
            "multi_objective_fitness": self.multi_objective_fitness,
            "age": self.age,
            "parent_ids": self.parent_ids,
            "created_at": self.created_at.isoformat(),
            "evaluated": self.evaluated
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Individual":
        """Create individual from dictionary representation."""
        individual = cls(
            chromosome=PolicyChromosome.from_dict(data["chromosome"]),
            fitness=data.get("fitness"),
            multi_objective_fitness=data.get("multi_objective_fitness", {}),
            age=data.get("age", 0),
            parent_ids=data.get("parent_ids", []),
            evaluated=data.get("evaluated", False)
        )
        if "created_at" in data:
            individual.created_at = datetime.fromisoformat(data["created_at"])
        return individual

    def __lt__(self, other: "Individual") -> bool:
        """Compare individuals by fitness (for sorting)."""
        if self.fitness is None:
            return True
        if other.fitness is None:
            return False
        return self.fitness < other.fitness


class Population:
    """
    Manages a population of individuals in the genetic algorithm.

    Handles population initialization, selection, diversity metrics,
    and generation management.
    """

    def __init__(self, config: DarwinConfig, generation: int = 0):
        """Initialize population with configuration."""
        self.config = config
        self.individuals: List[Individual] = []
        self.generation = generation
        self.best_individual: Optional[Individual] = None
        self.diversity_metrics: Dict[str, float] = {}
        self.statistics: Dict[str, Any] = {}
        self.history: List[Dict[str, Any]] = []

    def initialize_random(self, initialization_params: Dict[str, Any]) -> None:
        """Initialize population with random individuals."""
        for _ in range(self.config.evolution.population_size):
            chromosome = self._create_random_chromosome(initialization_params)
            individual = Individual(chromosome=chromosome)
            self.individuals.append(individual)

        # Set generation for all chromosomes
        for individual in self.individuals:
            individual.chromosome.generation = self.generation

    def initialize_from_seed(self, seed_chromosomes: List[PolicyChromosome]) -> None:
        """Initialize population from seed chromosomes."""
        # Use seed chromosomes directly
        for chromosome in seed_chromosomes[:self.config.evolution.population_size]:
            individual = Individual(chromosome=chromosome.clone())
            individual.chromosome.generation = self.generation
            self.individuals.append(individual)

        # Fill remaining slots with mutations of seed chromosomes
        while len(self.individuals) < self.config.evolution.population_size:
            # Select random seed and mutate it
            seed = random.choice(seed_chromosomes)
            mutated = seed.clone()
            mutated.mutate(self._get_mutation_params())
            individual = Individual(chromosome=mutated)
            individual.chromosome.generation = self.generation
            self.individuals.append(individual)

    def _create_random_chromosome(self, params: Dict[str, Any]) -> PolicyChromosome:
        """Create a random chromosome based on initialization parameters."""
        chromosome = PolicyChromosome()

        # Determine number of rules
        min_rules = params.get("min_rules", 5)
        max_rules = params.get("max_rules", 20)
        num_rules = random.randint(min_rules, max_rules)

        # Create random rules
        for _ in range(num_rules):
            rule = chromosome._create_random_rule(params)
            chromosome.add_rule(rule)

        return chromosome

    def _get_mutation_params(self) -> Dict[str, Any]:
        """Get mutation parameters from config."""
        return {
            "rule_mutation_rate": self.config.evolution.mutation_rate,
            "resource_mutation_rate": 0.1,
            "action_mutation_rate": 0.15,
            "condition_mutation_rate": 0.2,
            "effect_flip_rate": 0.05,
            "priority_mutation_rate": 0.1,
            "rule_add_rate": 0.1,
            "rule_remove_rate": 0.05,
            "max_rules": self.config.constraints.max_rules_per_policy,
            "min_rules": self.config.constraints.min_rules_per_policy,
            "resource_pool": ["app:*", "data:*", "api:*", "admin:*"],
            "action_pool": ["read", "write", "create", "update", "delete", "execute"],
            "attribute_pool": ["department", "role", "level", "location", "team"],
            "departments": ["Engineering", "Sales", "Marketing", "HR", "Finance", "Operations"],
            "roles": ["user", "admin", "manager", "developer", "analyst"]
        }

    def select_parents(self, num_parents: int) -> List[Individual]:
        """Select parents for reproduction using configured selection method."""
        method = self.config.evolution.selection_method

        if method == "tournament":
            return self._tournament_selection(num_parents)
        elif method == "roulette":
            return self._roulette_selection(num_parents)
        elif method == "rank":
            return self._rank_selection(num_parents)
        else:
            raise ValueError(f"Unknown selection method: {method}")

    def _tournament_selection(self, num_parents: int) -> List[Individual]:
        """Select parents using tournament selection."""
        parents = []
        tournament_size = self.config.evolution.tournament_size

        for _ in range(num_parents):
            # Random tournament
            tournament = random.sample(self.individuals, min(tournament_size, len(self.individuals)))
            # Select best from tournament
            winner = max(tournament, key=lambda x: x.fitness or float('-inf'))
            parents.append(winner)

        return parents

    def _roulette_selection(self, num_parents: int) -> List[Individual]:
        """Select parents using roulette wheel selection."""
        # Calculate fitness proportions
        total_fitness = sum(ind.fitness or 0 for ind in self.individuals)
        if total_fitness == 0:
            # Fallback to random selection
            return random.sample(self.individuals, num_parents)

        # Create roulette wheel
        wheel = []
        cumulative = 0
        for ind in self.individuals:
            cumulative += (ind.fitness or 0) / total_fitness
            wheel.append((cumulative, ind))

        # Select parents
        parents = []
        for _ in range(num_parents):
            r = random.random()
            for threshold, ind in wheel:
                if r <= threshold:
                    parents.append(ind)
                    break

        return parents

    def _rank_selection(self, num_parents: int) -> List[Individual]:
        """Select parents using rank-based selection."""
        # Sort by fitness
        sorted_individuals = sorted(
            self.individuals,
            key=lambda x: x.fitness or float('-inf'),
            reverse=True
        )

        # Assign rank-based probabilities
        n = len(sorted_individuals)
        probabilities = [(2 - i / n) / n for i in range(n)]

        # Select based on probabilities
        parents = random.choices(
            sorted_individuals,
            weights=probabilities,
            k=num_parents
        )

        return parents

    def get_elite(self) -> List[Individual]:
        """Get the elite individuals to preserve."""
        sorted_individuals = sorted(
            self.individuals,
            key=lambda x: x.fitness or float('-inf'),
            reverse=True
        )
        return sorted_individuals[:self.config.evolution.elite_size]

    def replace_population(self, new_individuals: List[Individual]) -> None:
        """Replace current population with new individuals."""
        self.individuals = new_individuals

        # Update generation
        self.generation += 1
        for ind in self.individuals:
            ind.chromosome.generation = self.generation
            ind.increment_age()

    def add_individuals(self, individuals: List[Individual]) -> None:
        """Add new individuals to the population."""
        for ind in individuals:
            ind.chromosome.generation = self.generation
        self.individuals.extend(individuals)

    def remove_worst(self, num_to_remove: int) -> None:
        """Remove the worst performing individuals."""
        if num_to_remove >= len(self.individuals):
            return

        sorted_individuals = sorted(
            self.individuals,
            key=lambda x: x.fitness or float('-inf'),
            reverse=True
        )
        self.individuals = sorted_individuals[:-num_to_remove]

    def calculate_diversity(self) -> Dict[str, float]:
        """Calculate population diversity metrics."""
        if not self.individuals:
            return {}

        # Rule count diversity
        rule_counts = [len(ind.chromosome.rules) for ind in self.individuals]
        rule_count_std = statistics.stdev(rule_counts) if len(rule_counts) > 1 else 0

        # Fitness diversity
        fitnesses = [ind.fitness for ind in self.individuals if ind.fitness is not None]
        fitness_std = statistics.stdev(fitnesses) if len(fitnesses) > 1 else 0

        # Unique chromosomes
        unique_ids = len(set(ind.id for ind in self.individuals))
        uniqueness_ratio = unique_ids / len(self.individuals)

        # Hamming distance between chromosomes (sample)
        if len(self.individuals) > 1:
            sample_size = min(50, len(self.individuals))
            sample = random.sample(self.individuals, sample_size)
            distances = []

            for i in range(len(sample)):
                for j in range(i + 1, len(sample)):
                    dist = self._chromosome_distance(
                        sample[i].chromosome,
                        sample[j].chromosome
                    )
                    distances.append(dist)

            avg_distance = statistics.mean(distances) if distances else 0
        else:
            avg_distance = 0

        self.diversity_metrics = {
            "rule_count_std": rule_count_std,
            "fitness_std": fitness_std,
            "uniqueness_ratio": uniqueness_ratio,
            "avg_chromosome_distance": avg_distance,
            "unique_chromosomes": unique_ids
        }

        return self.diversity_metrics

    def _chromosome_distance(self, chr1: PolicyChromosome, chr2: PolicyChromosome) -> float:
        """Calculate distance between two chromosomes."""
        # Simple distance based on rule differences
        rules1 = {(r.resource.value, tuple(a.value for a in r.actions)) for r in chr1.rules}
        rules2 = {(r.resource.value, tuple(a.value for a in r.actions)) for r in chr2.rules}

        symmetric_diff = len(rules1.symmetric_difference(rules2))
        total_rules = len(rules1) + len(rules2)

        return symmetric_diff / total_rules if total_rules > 0 else 0

    def calculate_statistics(self) -> Dict[str, Any]:
        """Calculate population statistics."""
        fitnesses = [ind.fitness for ind in self.individuals if ind.fitness is not None]

        if not fitnesses:
            return {}

        # Basic statistics
        stats = {
            "generation": self.generation,
            "population_size": len(self.individuals),
            "evaluated_count": len(fitnesses),
            "best_fitness": max(fitnesses),
            "worst_fitness": min(fitnesses),
            "avg_fitness": statistics.mean(fitnesses),
            "median_fitness": statistics.median(fitnesses),
            "fitness_std": statistics.stdev(fitnesses) if len(fitnesses) > 1 else 0
        }

        # Multi-objective statistics
        if self.individuals[0].multi_objective_fitness:
            objectives = list(self.individuals[0].multi_objective_fitness.keys())
            for obj in objectives:
                obj_values = [
                    ind.multi_objective_fitness.get(obj, 0)
                    for ind in self.individuals
                    if ind.multi_objective_fitness
                ]
                if obj_values:
                    stats[f"{obj}_avg"] = statistics.mean(obj_values)
                    stats[f"{obj}_best"] = max(obj_values)

        # Age statistics
        ages = [ind.age for ind in self.individuals]
        stats["avg_age"] = statistics.mean(ages)
        stats["max_age"] = max(ages)

        # Rule statistics
        rule_counts = [len(ind.chromosome.rules) for ind in self.individuals]
        stats["avg_rules"] = statistics.mean(rule_counts)
        stats["max_rules"] = max(rule_counts)
        stats["min_rules"] = min(rule_counts)

        self.statistics = stats
        return stats

    def update_best_individual(self) -> None:
        """Update the best individual in the population."""
        if not self.individuals:
            return

        evaluated = [ind for ind in self.individuals if ind.evaluated]
        if not evaluated:
            return

        best = max(evaluated, key=lambda x: x.fitness or float('-inf'))

        if self.best_individual is None or (best.fitness or 0) > (self.best_individual.fitness or 0):
            self.best_individual = best

    def save_snapshot(self, filepath: str) -> None:
        """Save population snapshot to file."""
        snapshot = {
            "generation": self.generation,
            "individuals": [ind.to_dict() for ind in self.individuals],
            "best_individual": self.best_individual.to_dict() if self.best_individual else None,
            "diversity_metrics": self.diversity_metrics,
            "statistics": self.statistics,
            "config": self.config.to_dict()
        }

        with open(filepath, 'w') as f:
            json.dump(snapshot, f, indent=2)

    @classmethod
    def load_snapshot(cls, filepath: str) -> "Population":
        """Load population from snapshot file."""
        with open(filepath, 'r') as f:
            snapshot = json.load(f)

        config = DarwinConfig(**snapshot["config"])
        population = cls(config, generation=snapshot["generation"])

        # Load individuals
        population.individuals = [
            Individual.from_dict(ind_data)
            for ind_data in snapshot["individuals"]
        ]

        # Load best individual
        if snapshot["best_individual"]:
            population.best_individual = Individual.from_dict(snapshot["best_individual"])

        # Load metrics
        population.diversity_metrics = snapshot.get("diversity_metrics", {})
        population.statistics = snapshot.get("statistics", {})

        return population

    def get_pareto_front(self) -> List[Individual]:
        """Get Pareto-optimal individuals for multi-objective optimization."""
        if not self.individuals or not self.individuals[0].multi_objective_fitness:
            return []

        pareto_front = []

        for ind in self.individuals:
            if not ind.multi_objective_fitness:
                continue

            dominated = False
            for other in self.individuals:
                if other == ind or not other.multi_objective_fitness:
                    continue

                if self._dominates(other, ind):
                    dominated = True
                    break

            if not dominated:
                pareto_front.append(ind)

        return pareto_front

    def _dominates(self, ind1: Individual, ind2: Individual) -> bool:
        """Check if ind1 dominates ind2 in multi-objective space."""
        objectives1 = ind1.multi_objective_fitness
        objectives2 = ind2.multi_objective_fitness

        if not objectives1 or not objectives2:
            return False

        # Check if ind1 is at least as good in all objectives
        at_least_as_good = all(
            objectives1.get(obj, 0) >= objectives2.get(obj, 0)
            for obj in objectives1.keys()
        )

        # Check if ind1 is strictly better in at least one objective
        strictly_better = any(
            objectives1.get(obj, 0) > objectives2.get(obj, 0)
            for obj in objectives1.keys()
        )

        return at_least_as_good and strictly_better

    def apply_migration(self, migrants: List[Individual]) -> None:
        """Apply migration by replacing some individuals with migrants."""
        if not migrants:
            return

        num_to_replace = int(len(self.individuals) * self.config.evolution.migration_rate)
        num_to_replace = min(num_to_replace, len(migrants))

        if num_to_replace == 0:
            return

        # Replace worst individuals with migrants
        self.remove_worst(num_to_replace)
        self.add_individuals(migrants[:num_to_replace])

    def detect_stagnation(self, lookback: int = 5) -> bool:
        """Detect if population has stagnated."""
        if len(self.history) < lookback:
            return False

        recent_best = [h.get("best_fitness", 0) for h in self.history[-lookback:]]

        # Check if best fitness hasn't improved
        if len(set(recent_best)) == 1:
            return True

        # Check if improvement is minimal
        if max(recent_best) - min(recent_best) < 0.001:
            return True

        return False

    def record_history(self) -> None:
        """Record current population state in history."""
        stats = self.calculate_statistics()
        diversity = self.calculate_diversity()

        history_entry = {
            **stats,
            **diversity,
            "timestamp": datetime.now().isoformat()
        }

        self.history.append(history_entry)

        # Limit history size
        max_history = 100
        if len(self.history) > max_history:
            self.history = self.history[-max_history:]
