"""
Genetic Algorithm Engine for Darwin Framework.

This module implements the main genetic algorithm engine that orchestrates
the evolution process, fitness evaluation, and optimization workflow.
"""

import os
import time
import random
import asyncio
import multiprocessing
from typing import List, Optional, Dict, Any, Callable, Tuple, Union
from datetime import datetime, timedelta
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
import logging
from pathlib import Path
import json

import logfire
import numpy as np

from src.darwin.core.config import DarwinConfig
from src.darwin.core.population import Population, Individual
from src.darwin.core.chromosome import PolicyChromosome


class GeneticAlgorithmEngine:
    """
    Main engine for running genetic algorithm optimization.

    Orchestrates the evolution process including initialization,
    fitness evaluation, selection, crossover, mutation, and termination.
    """

    def __init__(
        self,
        config: DarwinConfig,
        fitness_function: Callable[[PolicyChromosome], Dict[str, float]],
        logger: Optional[logging.Logger] = None
    ):
        """
        Initialize the genetic algorithm engine.

        Args:
            config: Darwin configuration
            fitness_function: Function to evaluate chromosome fitness
            logger: Optional logger instance
        """
        self.config = config
        self.fitness_function = fitness_function
        self.logger = logger or self._setup_logger()

        # Set random seed if specified
        if config.random_seed is not None:
            random.seed(config.random_seed)
            np.random.seed(config.random_seed)

        # State tracking
        self.current_population: Optional[Population] = None
        self.start_time: Optional[datetime] = None
        self.total_evaluations = 0
        self.stagnation_counter = 0

        # Setup directories
        self.checkpoint_dir = Path(config.checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Setup parallel execution if enabled
        self.executor: Optional[Union[ProcessPoolExecutor, ThreadPoolExecutor]] = None
        if config.parallelization.enable_parallel:
            num_workers = config.parallelization.num_workers or multiprocessing.cpu_count()
            # Use ThreadPoolExecutor for better compatibility with async code
            self.executor = ThreadPoolExecutor(max_workers=num_workers)

        # Island model setup
        self.islands: List[Population] = []
        if config.parallelization.island_model:
            self._initialize_islands()

    def _setup_logger(self) -> logging.Logger:
        """Setup default logger."""
        logger = logging.getLogger("darwin.engine")
        logger.setLevel(getattr(logging, self.config.logging.log_level))

        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)

        return logger

    def _initialize_islands(self) -> None:
        """Initialize island populations for island model."""
        num_islands = self.config.parallelization.num_islands
        island_size = self.config.evolution.population_size // num_islands

        # Adjust config for each island
        island_config = self.config.model_copy(deep=True)
        island_config.evolution.population_size = island_size
        island_config.parallelization.island_model = False  # Prevent recursive islands

        for i in range(num_islands):
            island = Population(island_config, generation=0)
            self.islands.append(island)

    async def evolve(
        self,
        initial_population: Optional[Population] = None,
        initialization_params: Optional[Dict[str, Any]] = None
    ) -> Population:
        """
        Run the genetic algorithm evolution process.

        Args:
            initial_population: Optional pre-initialized population
            initialization_params: Parameters for random initialization

        Returns:
            Final evolved population
        """
        with logfire.span("GA Evolution",
                         population_size=self.config.evolution.population_size,
                         generations=self.config.evolution.generations):

            self.start_time = datetime.now()
            self.logger.info(f"Starting evolution with population size {self.config.evolution.population_size}")

            # Initialize population
            if initial_population:
                self.current_population = initial_population
            else:
                self.current_population = await self._initialize_population(initialization_params)

            # Main evolution loop
            for generation in range(self.config.evolution.generations):
                with logfire.span("Generation", generation=generation):
                    # Check termination conditions
                    if self._should_terminate():
                        self.logger.info(f"Early termination at generation {generation}")
                        break

                    # Evaluate fitness
                    await self._evaluate_population()

                    # Update statistics and best individual
                    self.current_population.update_best_individual()
                    self.current_population.calculate_statistics()
                    self.current_population.calculate_diversity()

                    # Log progress
                    if generation % self.config.logging.log_interval == 0:
                        self._log_progress(generation)

                    # Save snapshot
                    if self.config.logging.save_snapshots and \
                       generation % self.config.logging.snapshot_interval == 0:
                        self._save_snapshot(generation)

                    # Check for stagnation
                    if self._detect_stagnation():
                        await self._handle_stagnation()

                    # Record history
                    self.current_population.record_history()

                    # Create next generation (unless last generation)
                    if generation < self.config.evolution.generations - 1:
                        if self.config.parallelization.island_model:
                            await self._evolve_islands()
                        else:
                            await self._create_next_generation()

            # Final evaluation
            await self._evaluate_population()
            self.current_population.update_best_individual()

            # Save final results
            self._save_final_results()

            # Cleanup
            if self.executor:
                self.executor.shutdown(wait=True)

            elapsed_time = datetime.now() - self.start_time
            self.logger.info(f"Evolution completed in {elapsed_time}")

            return self.current_population

    async def _initialize_population(self, params: Optional[Dict[str, Any]]) -> Population:
        """Initialize the population."""
        with logfire.span("Initialize Population"):
            if params is None:
                params = self._get_default_initialization_params()

            population = Population(self.config, generation=0)

            if self.config.parallelization.island_model:
                # Initialize each island
                for island in self.islands:
                    island.initialize_random(params)

                # Combine islands into main population
                all_individuals = []
                for island in self.islands:
                    all_individuals.extend(island.individuals)
                population.individuals = all_individuals
            else:
                population.initialize_random(params)

            self.logger.info(f"Initialized population with {len(population.individuals)} individuals")
            return population

    def _get_default_initialization_params(self) -> Dict[str, Any]:
        """Get default initialization parameters."""
        return {
            "min_rules": self.config.constraints.min_rules_per_policy,
            "max_rules": self.config.constraints.max_rules_per_policy,
            "resource_pool": ["app:*", "data:*", "api:*", "admin:*", "service:*"],
            "action_pool": ["read", "write", "create", "update", "delete", "execute", "manage"],
            "attribute_pool": ["department", "role", "level", "location", "team", "project"],
            "departments": ["Engineering", "Sales", "Marketing", "HR", "Finance", "Operations", "Legal"],
            "roles": ["user", "admin", "manager", "developer", "analyst", "executive"],
            "value_pools": {
                "department": ["Engineering", "Sales", "Marketing", "HR", "Finance"],
                "role": ["user", "admin", "manager", "developer"],
                "level": list(range(1, 6)),
                "location": ["US", "EU", "APAC", "LATAM"]
            }
        }

    async def _evaluate_population(self) -> None:
        """Evaluate fitness for all individuals in the population."""
        with logfire.span("Evaluate Population", size=len(self.current_population.individuals)):
            unevaluated = [ind for ind in self.current_population.individuals if not ind.evaluated]

            if not unevaluated:
                return

            if self.config.parallelization.enable_parallel and self.executor:
                await self._parallel_evaluation(unevaluated)
            else:
                await self._sequential_evaluation(unevaluated)

            self.total_evaluations += len(unevaluated)
            logfire.info(f"Evaluated {len(unevaluated)} individuals",
                        total_evaluations=self.total_evaluations)

    async def _sequential_evaluation(self, individuals: List[Individual]) -> None:
        """Evaluate individuals sequentially."""
        for individual in individuals:
            objectives = self.fitness_function(individual.chromosome)
            fitness = self._calculate_combined_fitness(objectives)
            individual.update_fitness(fitness, objectives)

    async def _parallel_evaluation(self, individuals: List[Individual]) -> None:
        """Evaluate individuals in parallel."""
        chunk_size = self.config.parallelization.chunk_size
        chunks = [individuals[i:i + chunk_size] for i in range(0, len(individuals), chunk_size)]

        futures = []
        for chunk in chunks:
            future = self.executor.submit(self._evaluate_chunk, chunk)
            futures.append(future)

        # Wait for all evaluations to complete
        for future in as_completed(futures):
            evaluated_chunk = future.result()
            for ind, (fitness, objectives) in evaluated_chunk:
                ind.update_fitness(fitness, objectives)

    def _evaluate_chunk(self, individuals: List[Individual]) -> List[Tuple[Individual, Tuple[float, Dict[str, float]]]]:
        """Evaluate a chunk of individuals (for parallel processing)."""
        results = []
        for individual in individuals:
            objectives = self.fitness_function(individual.chromosome)
            fitness = self._calculate_combined_fitness(objectives)
            results.append((individual, (fitness, objectives)))
        return results

    def _calculate_combined_fitness(self, objectives: Dict[str, float]) -> float:
        """Calculate combined fitness from multiple objectives."""
        if not objectives:
            return 0.0

        # Normalize objectives if configured
        normalized = self._normalize_objectives(objectives)

        # Apply weights and combine
        weighted_sum = 0.0
        total_weight = 0.0

        for obj_config in self.config.fitness.objectives:
            if obj_config.name in normalized:
                value = normalized[obj_config.name]

                # Invert if minimizing
                if obj_config.minimize:
                    value = 1.0 - value

                weighted_sum += value * obj_config.weight
                total_weight += obj_config.weight

        # Normalize by total weight
        if total_weight > 0:
            combined = weighted_sum / total_weight
        else:
            combined = sum(normalized.values()) / len(normalized)

        return combined

    def _normalize_objectives(self, objectives: Dict[str, float]) -> Dict[str, float]:
        """Normalize objective values based on configuration."""
        if self.config.fitness.normalization_method == "none":
            return objectives

        normalized = {}

        if self.config.fitness.normalization_method == "minmax":
            # Get min/max from population history
            for name, value in objectives.items():
                history_values = []
                for ind in self.current_population.individuals:
                    if ind.multi_objective_fitness and name in ind.multi_objective_fitness:
                        history_values.append(ind.multi_objective_fitness[name])

                if history_values:
                    min_val = min(history_values)
                    max_val = max(history_values)
                    if max_val > min_val:
                        normalized[name] = (value - min_val) / (max_val - min_val)
                    else:
                        normalized[name] = 0.5
                else:
                    normalized[name] = value

        elif self.config.fitness.normalization_method == "zscore":
            # Z-score normalization
            for name, value in objectives.items():
                history_values = []
                for ind in self.current_population.individuals:
                    if ind.multi_objective_fitness and name in ind.multi_objective_fitness:
                        history_values.append(ind.multi_objective_fitness[name])

                if len(history_values) > 1:
                    mean = np.mean(history_values)
                    std = np.std(history_values)
                    if std > 0:
                        z_score = (value - mean) / std
                        # Convert to 0-1 range using sigmoid
                        normalized[name] = 1 / (1 + np.exp(-z_score))
                    else:
                        normalized[name] = 0.5
                else:
                    normalized[name] = value

        return normalized

    async def _create_next_generation(self) -> None:
        """Create the next generation of individuals."""
        with logfire.span("Create Next Generation"):
            new_individuals = []

            # Preserve elite
            elite = self.current_population.get_elite()
            new_individuals.extend([ind for ind in elite])

            # Generate offspring
            while len(new_individuals) < self.config.evolution.population_size:
                # Select parents
                parents = self.current_population.select_parents(2)

                # Crossover
                if random.random() < self.config.evolution.crossover_rate:
                    offspring = self._crossover(parents[0], parents[1])
                else:
                    # Clone parents without crossover
                    offspring = [
                        Individual(chromosome=parents[0].chromosome.clone()),
                        Individual(chromosome=parents[1].chromosome.clone())
                    ]

                # Mutation
                for child in offspring:
                    if random.random() < self.config.evolution.mutation_rate:
                        self._mutate(child)

                    # Set parent IDs
                    child.parent_ids = [p.id for p in parents]

                new_individuals.extend(offspring)

            # Trim to exact population size
            new_individuals = new_individuals[:self.config.evolution.population_size]

            # Replace population
            self.current_population.replace_population(new_individuals)

    def _crossover(self, parent1: Individual, parent2: Individual) -> List[Individual]:
        """Perform crossover between two parents."""
        crossover_params = {
            "method": "uniform",
            "inherit_probability": 0.7
        }

        child1_chr, child2_chr = parent1.chromosome.crossover(
            parent2.chromosome,
            crossover_params
        )

        return [
            Individual(chromosome=child1_chr),
            Individual(chromosome=child2_chr)
        ]

    def _mutate(self, individual: Individual) -> None:
        """Mutate an individual."""
        mutation_params = self._get_mutation_params()

        # Adaptive mutation
        if self.config.evolution.adaptive_mutation:
            # Increase mutation rate if diversity is low
            diversity = self.current_population.diversity_metrics.get("uniqueness_ratio", 1.0)
            if diversity < 0.5:
                for key in mutation_params:
                    if "rate" in key:
                        mutation_params[key] *= 1.5

        individual.chromosome.mutate(mutation_params)

    def _get_mutation_params(self) -> Dict[str, Any]:
        """Get mutation parameters."""
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
            **self._get_default_initialization_params()
        }

    async def _evolve_islands(self) -> None:
        """Evolve islands independently and perform migration."""
        # Evolve each island
        island_futures = []
        for i, island in enumerate(self.islands):
            future = self._evolve_single_island(island, i)
            island_futures.append(future)

        # Wait for all islands to complete
        await asyncio.gather(*island_futures)

        # Perform migration
        if self.current_population.generation % 5 == 0:  # Migrate every 5 generations
            self._perform_migration()

    async def _evolve_single_island(self, island: Population, island_id: int) -> None:
        """Evolve a single island."""
        # Similar to _create_next_generation but for an island
        # This would be implemented based on island-specific evolution
        pass

    def _perform_migration(self) -> None:
        """Perform migration between islands."""
        migration_size = max(1, int(self.islands[0].config.evolution.population_size *
                                   self.config.evolution.migration_rate))

        # Ring topology migration
        for i in range(len(self.islands)):
            source = self.islands[i]
            target = self.islands[(i + 1) % len(self.islands)]

            # Select best individuals from source
            migrants = source.get_elite()[:migration_size]

            # Apply migration to target
            target.apply_migration(migrants)

    def _should_terminate(self) -> bool:
        """Check if evolution should terminate early."""
        # Check runtime limit
        if self.config.max_runtime:
            elapsed = datetime.now() - self.start_time
            if elapsed > self.config.max_runtime:
                self.logger.info("Terminating due to runtime limit")
                return True

        # Check fitness convergence
        if self.current_population.best_individual:
            best_fitness = self.current_population.best_individual.fitness or 0

            # Check if fitness target reached
            for obj in self.config.fitness.objectives:
                if obj.target_value is not None:
                    current_value = self.current_population.best_individual.multi_objective_fitness.get(obj.name, 0)
                    if obj.minimize and current_value <= obj.target_value:
                        self.logger.info(f"Target reached for {obj.name}: {current_value} <= {obj.target_value}")
                        return True
                    elif not obj.minimize and current_value >= obj.target_value:
                        self.logger.info(f"Target reached for {obj.name}: {current_value} >= {obj.target_value}")
                        return True

        return False

    def _detect_stagnation(self) -> bool:
        """Detect if evolution has stagnated."""
        if self.current_population.detect_stagnation(self.config.evolution.stagnation_generations):
            self.stagnation_counter += 1
            return True
        else:
            self.stagnation_counter = 0
            return False

    async def _handle_stagnation(self) -> None:
        """Handle stagnation in evolution."""
        self.logger.warning(f"Stagnation detected (count: {self.stagnation_counter})")

        with logfire.span("Handle Stagnation"):
            # Increase mutation rate temporarily
            original_rate = self.config.evolution.mutation_rate
            self.config.evolution.mutation_rate = min(0.5, original_rate * 2)

            # Introduce random individuals
            num_random = int(self.config.evolution.population_size * 0.2)
            random_individuals = []

            params = self._get_default_initialization_params()
            for _ in range(num_random):
                chromosome = self.current_population._create_random_chromosome(params)
                random_individuals.append(Individual(chromosome=chromosome))

            # Replace worst individuals
            self.current_population.remove_worst(num_random)
            self.current_population.add_individuals(random_individuals)

            # Restore mutation rate
            self.config.evolution.mutation_rate = original_rate

    def _log_progress(self, generation: int) -> None:
        """Log evolution progress."""
        stats = self.current_population.statistics
        diversity = self.current_population.diversity_metrics

        self.logger.info(
            f"Generation {generation}: "
            f"Best: {stats.get('best_fitness', 0):.4f}, "
            f"Avg: {stats.get('avg_fitness', 0):.4f}, "
            f"Diversity: {diversity.get('uniqueness_ratio', 0):.2f}"
        )

        if self.config.logging.metrics_export:
            # Merge all metrics, avoiding duplicate keys
            metrics = {
                "evolution_generation": generation,
                **{k: v for k, v in stats.items() if k != "generation"},
                **diversity
            }
            logfire.info("Evolution Progress", **metrics)

    def _save_snapshot(self, generation: int) -> None:
        """Save population snapshot."""
        filename = f"population_gen_{generation:04d}.json"
        filepath = self.checkpoint_dir / filename

        try:
            self.current_population.save_snapshot(str(filepath))
            self.logger.debug(f"Saved snapshot to {filepath}")
        except Exception as e:
            self.logger.error(f"Failed to save snapshot: {e}")

    def _save_final_results(self) -> None:
        """Save final evolution results."""
        results = {
            "config": self.config.to_dict(),
            "best_individual": self.current_population.best_individual.to_dict()
                              if self.current_population.best_individual else None,
            "final_statistics": self.current_population.statistics,
            "evolution_history": self.current_population.history,
            "total_evaluations": self.total_evaluations,
            "runtime": str(datetime.now() - self.start_time),
            "final_generation": self.current_population.generation
        }

        # Save results
        results_file = self.checkpoint_dir / "final_results.json"
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)

        # Save best chromosome
        if self.current_population.best_individual:
            best_file = self.checkpoint_dir / "best_chromosome.json"
            with open(best_file, 'w') as f:
                json.dump(
                    self.current_population.best_individual.chromosome.to_dict(),
                    f,
                    indent=2
                )

        self.logger.info(f"Saved final results to {self.checkpoint_dir}")

    def load_checkpoint(self, checkpoint_path: str) -> None:
        """Load evolution state from checkpoint."""
        self.current_population = Population.load_snapshot(checkpoint_path)
        self.logger.info(f"Loaded checkpoint from {checkpoint_path}")

    async def evaluate_single(self, chromosome: PolicyChromosome) -> Dict[str, float]:
        """Evaluate a single chromosome (useful for testing)."""
        objectives = self.fitness_function(chromosome)
        fitness = self._calculate_combined_fitness(objectives)
        return {
            "combined_fitness": fitness,
            **objectives
        }
