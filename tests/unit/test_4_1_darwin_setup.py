"""
Unit tests for Darwin Framework Integration (Subtask 4.1).

Tests cover:
- Darwin configuration setup
- Basic GA engine initialization
- Population management
- Evolution parameter validation
- Chromosome representation basics
"""

import pytest
import asyncio
from pathlib import Path
import json
import tempfile
import random
from datetime import timedelta
from typing import Dict, Any

from src.darwin import (
    DarwinConfig,
    EvolutionParameters,
    GeneticAlgorithmEngine,
    Population,
    Individual,
    PolicyChromosome,
    Gene,
    GeneType,
    PolicyRule,
    PolicyEffect
)
from src.darwin.core.config import (
    ObjectiveConfig,
    FitnessConfig,
    ConstraintConfig,
    create_default_config,
    create_test_config,
    create_production_config
)


class TestDarwinConfiguration:
    """Test suite for Darwin configuration."""

    def test_default_config_creation(self):
        """Test creating default Darwin configuration."""
        config = DarwinConfig()

        assert config.evolution.population_size == 100
        assert config.evolution.generations == 50
        assert config.evolution.mutation_rate == 0.1
        assert config.evolution.crossover_rate == 0.8
        assert config.evolution.elite_size == 10

    def test_config_from_environment(self, monkeypatch):
        """Test loading configuration from environment variables."""
        monkeypatch.setenv("DARWIN_POPULATION_SIZE", "200")
        monkeypatch.setenv("DARWIN_GENERATIONS", "100")
        monkeypatch.setenv("DARWIN_MUTATION_RATE", "0.2")
        monkeypatch.setenv("DARWIN_RANDOM_SEED", "42")

        config = DarwinConfig.from_env()

        assert config.evolution.population_size == 200
        assert config.evolution.generations == 100
        assert config.evolution.mutation_rate == 0.2
        assert config.random_seed == 42

    def test_evolution_parameters_validation(self):
        """Test validation of evolution parameters."""
        # Valid parameters
        params = EvolutionParameters(
            population_size=100,
            generations=50,
            mutation_rate=0.1,
            crossover_rate=0.8,
            elite_size=10
        )
        assert params.population_size == 100

        # Invalid mutation rate
        with pytest.raises(ValueError):
            EvolutionParameters(mutation_rate=1.5)

        # Invalid population size
        with pytest.raises(ValueError):
            EvolutionParameters(population_size=5)

    def test_config_consistency_validation(self):
        """Test configuration consistency checks."""
        config = DarwinConfig()

        # Elite size larger than population should fail
        config.evolution.population_size = 50
        config.evolution.elite_size = 60

        with pytest.raises(ValueError):
            config.validate_consistency()

    def test_fitness_objectives_configuration(self):
        """Test fitness objectives configuration."""
        config = DarwinConfig()

        # Default objectives
        assert len(config.fitness.objectives) == 3
        assert any(obj.name == "security" for obj in config.fitness.objectives)
        assert any(obj.name == "productivity" for obj in config.fitness.objectives)
        assert any(obj.name == "compliance" for obj in config.fitness.objectives)

        # Custom objectives
        custom_objectives = [
            ObjectiveConfig(name="performance", weight=0.5),
            ObjectiveConfig(name="cost", weight=0.3, minimize=True),
            ObjectiveConfig(name="reliability", weight=0.2)
        ]
        config.fitness.objectives = custom_objectives

        assert len(config.fitness.objectives) == 3
        assert config.fitness.objectives[1].minimize is True

    def test_constraint_configuration(self):
        """Test constraint configuration."""
        config = DarwinConfig()

        assert config.constraints.max_rules_per_policy == 50
        assert config.constraints.min_rules_per_policy == 1
        assert config.constraints.max_conditions_per_rule == 10

        # Add required attributes
        config.constraints.required_attributes = ["department", "role"]
        assert "department" in config.constraints.required_attributes

    def test_parallelization_configuration(self):
        """Test parallelization configuration."""
        config = DarwinConfig()

        assert config.parallelization.enable_parallel is True

        # Island model configuration
        config.parallelization.island_model = True
        config.parallelization.num_islands = 4

        assert config.parallelization.island_model is True
        assert config.parallelization.num_islands == 4

    def test_config_save_and_load(self):
        """Test saving and loading configuration."""
        config = create_test_config()

        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            config.save(f.name)
            temp_path = f.name

        # Load config
        loaded_config = DarwinConfig.load(temp_path)

        assert loaded_config.evolution.population_size == config.evolution.population_size
        assert loaded_config.evolution.generations == config.evolution.generations

        # Cleanup
        Path(temp_path).unlink()

    def test_predefined_configurations(self):
        """Test predefined configuration templates."""
        # Default config
        default = create_default_config()
        assert default.evolution.population_size == 100

        # Test config (smaller, faster)
        test = create_test_config()
        assert test.evolution.population_size == 20
        assert test.evolution.generations == 10
        assert test.parallelization.enable_parallel is False

        # Production config (larger, optimized)
        prod = create_production_config()
        assert prod.evolution.population_size == 500
        assert prod.evolution.generations == 100
        assert prod.parallelization.island_model is True


class TestChromosomeRepresentation:
    """Test suite for chromosome representation."""

    def test_gene_creation(self):
        """Test creating different types of genes."""
        # Resource gene
        resource_gene = Gene(
            gene_type=GeneType.RESOURCE,
            value="app:users:*",
            mutable=True
        )
        assert resource_gene.gene_type == GeneType.RESOURCE
        assert resource_gene.value == "app:users:*"

        # Action gene
        action_gene = Gene(
            gene_type=GeneType.ACTION,
            value=["read", "write"],
            mutable=True
        )
        assert action_gene.gene_type == GeneType.ACTION
        assert "read" in action_gene.value

        # Condition gene
        condition_gene = Gene(
            gene_type=GeneType.CONDITION,
            value={
                "attribute": "department",
                "operator": "eq",
                "value": "Engineering"
            }
        )
        assert condition_gene.value["attribute"] == "department"

    def test_gene_mutation(self):
        """Test gene mutation functionality."""
        # Create mutable gene
        gene = Gene(
            gene_type=GeneType.ACTION,
            value=["read"],
            mutable=True
        )

        mutation_params = {
            "action_pool": ["read", "write", "delete", "create"],
            "change_probability": 1.0  # Force mutation
        }

        # Mutate multiple times to see changes
        original_value = gene.value.copy()
        mutations_occurred = False

        for _ in range(10):
            gene.mutate(mutation_params)
            if gene.value != original_value:
                mutations_occurred = True
                break

        assert mutations_occurred

    def test_immutable_gene(self):
        """Test that immutable genes don't mutate."""
        gene = Gene(
            gene_type=GeneType.RESOURCE,
            value="critical:resource",
            mutable=False
        )

        original_value = gene.value
        gene.mutate({"resource_pool": ["other:resource"]})

        assert gene.value == original_value

    def test_policy_rule_creation(self):
        """Test creating a policy rule."""
        rule = PolicyRule(
            resource=Gene(GeneType.RESOURCE, "app:api:*"),
            actions=[Gene(GeneType.ACTION, "read"), Gene(GeneType.ACTION, "write")],
            conditions=[
                Gene(GeneType.CONDITION, {
                    "attribute": "role",
                    "operator": "eq",
                    "value": "developer"
                })
            ],
            effect=Gene(GeneType.EFFECT, PolicyEffect.ALLOW.value),
            priority=Gene(GeneType.PRIORITY, 100)
        )

        assert rule.resource.value == "app:api:*"
        assert len(rule.actions) == 2
        assert len(rule.conditions) == 1
        assert rule.effect.value == "allow"
        assert rule.priority.value == 100
        assert rule.rule_id is not None

    def test_policy_chromosome_creation(self):
        """Test creating a policy chromosome."""
        chromosome = PolicyChromosome()

        # Add rules
        for i in range(3):
            rule = PolicyRule(
                resource=Gene(GeneType.RESOURCE, f"app:resource{i}"),
                actions=[Gene(GeneType.ACTION, "read")],
                conditions=[],
                effect=Gene(GeneType.EFFECT, PolicyEffect.ALLOW.value),
                priority=Gene(GeneType.PRIORITY, i * 10)
            )
            chromosome.add_rule(rule)

        assert len(chromosome.rules) == 3
        assert chromosome.chromosome_id is not None

    def test_chromosome_validation(self):
        """Test chromosome validation against constraints."""
        chromosome = PolicyChromosome()

        # Add a valid rule
        rule = PolicyRule(
            resource=Gene(GeneType.RESOURCE, "app:data"),
            actions=[Gene(GeneType.ACTION, "read")],
            conditions=[],
            effect=Gene(GeneType.EFFECT, PolicyEffect.ALLOW.value),
            priority=Gene(GeneType.PRIORITY, 100)
        )
        chromosome.add_rule(rule)

        constraints = {
            "min_rules": 1,
            "max_rules": 5,
            "required_resources": ["app:data"]
        }

        is_valid, errors = chromosome.validate(constraints)
        assert is_valid
        assert len(errors) == 0

        # Test with missing required resource
        constraints["required_resources"] = ["app:missing"]
        is_valid, errors = chromosome.validate(constraints)
        assert not is_valid
        assert any("Missing required resource" in error for error in errors)

    def test_chromosome_crossover(self):
        """Test chromosome crossover operation."""
        # Create parent chromosomes
        parent1 = PolicyChromosome()
        parent2 = PolicyChromosome()

        # Add different rules to each parent
        for i in range(3):
            rule1 = PolicyRule(
                resource=Gene(GeneType.RESOURCE, f"parent1:resource{i}"),
                actions=[Gene(GeneType.ACTION, "read")],
                conditions=[],
                effect=Gene(GeneType.EFFECT, PolicyEffect.ALLOW.value),
                priority=Gene(GeneType.PRIORITY, 100)
            )
            parent1.add_rule(rule1)

            rule2 = PolicyRule(
                resource=Gene(GeneType.RESOURCE, f"parent2:resource{i}"),
                actions=[Gene(GeneType.ACTION, "write")],
                conditions=[],
                effect=Gene(GeneType.EFFECT, PolicyEffect.ALLOW.value),
                priority=Gene(GeneType.PRIORITY, 200)
            )
            parent2.add_rule(rule2)

        # Perform crossover
        crossover_params = {"method": "uniform"}
        child1, child2 = parent1.crossover(parent2, crossover_params)

        # Children should have rules from both parents
        assert len(child1.rules) > 0
        assert len(child2.rules) > 0

        # Check that children are different from parents
        assert child1.chromosome_id != parent1.chromosome_id
        assert child2.chromosome_id != parent2.chromosome_id

    def test_chromosome_serialization(self):
        """Test chromosome serialization and deserialization."""
        chromosome = PolicyChromosome()

        # Add a complex rule
        rule = PolicyRule(
            resource=Gene(GeneType.RESOURCE, "app:api:users"),
            actions=[
                Gene(GeneType.ACTION, "read"),
                Gene(GeneType.ACTION, "update")
            ],
            conditions=[
                Gene(GeneType.CONDITION, {
                    "attribute": "department",
                    "operator": "in",
                    "value": ["Engineering", "IT"]
                })
            ],
            effect=Gene(GeneType.EFFECT, PolicyEffect.ALLOW.value),
            priority=Gene(GeneType.PRIORITY, 150)
        )
        chromosome.add_rule(rule)

        # Serialize
        data = chromosome.to_dict()

        # Deserialize
        loaded = PolicyChromosome.from_dict(data)

        assert len(loaded.rules) == 1
        assert loaded.rules[0].resource.value == "app:api:users"
        assert len(loaded.rules[0].actions) == 2
        assert loaded.rules[0].conditions[0].value["attribute"] == "department"


class TestPopulationManagement:
    """Test suite for population management."""

    def test_individual_creation(self):
        """Test creating an individual."""
        chromosome = PolicyChromosome()
        individual = Individual(chromosome=chromosome)

        assert individual.chromosome == chromosome
        assert individual.fitness is None
        assert individual.age == 0
        assert not individual.evaluated

    def test_individual_fitness_update(self):
        """Test updating individual fitness."""
        individual = Individual(chromosome=PolicyChromosome())

        objectives = {
            "security": 0.8,
            "productivity": 0.7,
            "compliance": 0.9
        }
        combined_fitness = 0.8

        individual.update_fitness(combined_fitness, objectives)

        assert individual.fitness == 0.8
        assert individual.multi_objective_fitness == objectives
        assert individual.evaluated

    def test_population_initialization(self):
        """Test population initialization."""
        config = create_test_config()
        population = Population(config)

        init_params = {
            "min_rules": 2,
            "max_rules": 5,
            "resource_pool": ["app:*", "data:*"],
            "action_pool": ["read", "write", "delete"]
        }

        population.initialize_random(init_params)

        assert len(population.individuals) == config.evolution.population_size
        assert all(isinstance(ind, Individual) for ind in population.individuals)
        assert all(len(ind.chromosome.rules) >= 2 for ind in population.individuals)

    def test_population_selection_methods(self):
        """Test different parent selection methods."""
        config = create_test_config()
        population = Population(config)

        # Create population with fitness values
        for i in range(10):
            chromosome = PolicyChromosome()
            individual = Individual(chromosome=chromosome)
            individual.update_fitness(i / 10.0, {"test": i / 10.0})
            population.individuals.append(individual)

        # Tournament selection
        config.evolution.selection_method = "tournament"
        parents = population.select_parents(5)
        assert len(parents) == 5

        # Roulette selection
        config.evolution.selection_method = "roulette"
        parents = population.select_parents(5)
        assert len(parents) == 5

        # Rank selection
        config.evolution.selection_method = "rank"
        parents = population.select_parents(5)
        assert len(parents) == 5

    def test_population_elite_preservation(self):
        """Test elite preservation in population."""
        config = create_test_config()
        config.evolution.elite_size = 2
        population = Population(config)

        # Create population with varying fitness
        for i in range(10):
            chromosome = PolicyChromosome()
            individual = Individual(chromosome=chromosome)
            individual.update_fitness(i / 10.0, {"test": i / 10.0})
            population.individuals.append(individual)

        elite = population.get_elite()

        assert len(elite) == 2
        assert elite[0].fitness >= elite[1].fitness
        assert elite[0].fitness == 0.9  # Best individual

    def test_population_diversity_metrics(self):
        """Test population diversity calculation."""
        config = create_test_config()
        population = Population(config)

        # Create diverse population
        for i in range(10):
            chromosome = PolicyChromosome()
            # Add different numbers of rules
            for j in range(i % 3 + 1):
                rule = PolicyRule(
                    resource=Gene(GeneType.RESOURCE, f"res{i}_{j}"),
                    actions=[Gene(GeneType.ACTION, "read")],
                    conditions=[],
                    effect=Gene(GeneType.EFFECT, PolicyEffect.ALLOW.value),
                    priority=Gene(GeneType.PRIORITY, 100)
                )
                chromosome.add_rule(rule)

            individual = Individual(chromosome=chromosome)
            individual.update_fitness(random.random(), {})
            population.individuals.append(individual)

        diversity = population.calculate_diversity()

        assert "uniqueness_ratio" in diversity
        assert "fitness_std" in diversity
        assert diversity["unique_chromosomes"] > 0

    def test_population_statistics(self):
        """Test population statistics calculation."""
        config = create_test_config()
        population = Population(config)

        # Create population with known fitness values
        fitness_values = [0.1, 0.3, 0.5, 0.7, 0.9]
        for fitness in fitness_values:
            chromosome = PolicyChromosome()
            individual = Individual(chromosome=chromosome)
            individual.update_fitness(fitness, {"test": fitness})
            population.individuals.append(individual)

        stats = population.calculate_statistics()

        assert stats["population_size"] == 5
        assert stats["best_fitness"] == 0.9
        assert stats["worst_fitness"] == 0.1
        assert stats["avg_fitness"] == pytest.approx(0.5, rel=0.01)

    def test_population_snapshot(self):
        """Test saving and loading population snapshots."""
        config = create_test_config()
        population = Population(config)

        # Create a simple population
        for i in range(5):
            chromosome = PolicyChromosome()
            rule = PolicyRule(
                resource=Gene(GeneType.RESOURCE, f"resource{i}"),
                actions=[Gene(GeneType.ACTION, "read")],
                conditions=[],
                effect=Gene(GeneType.EFFECT, PolicyEffect.ALLOW.value),
                priority=Gene(GeneType.PRIORITY, 100)
            )
            chromosome.add_rule(rule)
            individual = Individual(chromosome=chromosome)
            individual.update_fitness(i / 5.0, {"test": i / 5.0})
            population.individuals.append(individual)

        # Save snapshot
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            population.save_snapshot(f.name)
            temp_path = f.name

        # Load snapshot
        loaded_population = Population.load_snapshot(temp_path)

        assert len(loaded_population.individuals) == 5
        assert loaded_population.generation == population.generation

        # Cleanup
        Path(temp_path).unlink()


class TestGeneticAlgorithmEngine:
    """Test suite for genetic algorithm engine."""

    @pytest.fixture
    def simple_fitness_function(self):
        """Create a simple fitness function for testing."""
        def fitness_func(chromosome: PolicyChromosome) -> Dict[str, float]:
            # Simple fitness based on number of rules
            security = min(len(chromosome.rules) / 10.0, 1.0)
            productivity = 1.0 - security  # Inverse relationship
            compliance = random.random()  # Random for testing

            return {
                "security": security,
                "productivity": productivity,
                "compliance": compliance
            }
        return fitness_func

    def test_engine_initialization(self, simple_fitness_function):
        """Test genetic algorithm engine initialization."""
        config = create_test_config()
        engine = GeneticAlgorithmEngine(config, simple_fitness_function)

        assert engine.config == config
        assert engine.fitness_function == simple_fitness_function
        assert engine.current_population is None
        assert engine.checkpoint_dir.exists()

    @pytest.mark.asyncio
    async def test_basic_evolution(self, simple_fitness_function):
        """Test basic evolution process."""
        config = create_test_config()
        config.evolution.generations = 3
        config.evolution.population_size = 10

        engine = GeneticAlgorithmEngine(config, simple_fitness_function)

        # Run evolution
        final_population = await engine.evolve()

        assert final_population is not None
        assert final_population.generation == 3
        assert len(final_population.individuals) == 10
        assert final_population.best_individual is not None

    @pytest.mark.asyncio
    async def test_fitness_evaluation(self, simple_fitness_function):
        """Test fitness evaluation for population."""
        config = create_test_config()
        engine = GeneticAlgorithmEngine(config, simple_fitness_function)

        # Create a test population
        population = Population(config)
        for i in range(5):
            chromosome = PolicyChromosome()
            for j in range(i + 1):
                rule = PolicyRule(
                    resource=Gene(GeneType.RESOURCE, f"res{j}"),
                    actions=[Gene(GeneType.ACTION, "read")],
                    conditions=[],
                    effect=Gene(GeneType.EFFECT, PolicyEffect.ALLOW.value),
                    priority=Gene(GeneType.PRIORITY, 100)
                )
                chromosome.add_rule(rule)
            population.individuals.append(Individual(chromosome=chromosome))

        engine.current_population = population
        await engine._evaluate_population()

        # All individuals should be evaluated
        assert all(ind.evaluated for ind in population.individuals)
        assert all(ind.fitness is not None for ind in population.individuals)

    def test_mutation_parameters(self, simple_fitness_function):
        """Test mutation parameter generation."""
        config = create_test_config()
        engine = GeneticAlgorithmEngine(config, simple_fitness_function)

        params = engine._get_mutation_params()

        assert "rule_mutation_rate" in params
        assert params["rule_mutation_rate"] == config.evolution.mutation_rate
        assert "max_rules" in params
        assert params["max_rules"] == config.constraints.max_rules_per_policy
        assert "resource_pool" in params
        assert isinstance(params["resource_pool"], list)

    def test_objective_normalization(self, simple_fitness_function):
        """Test objective normalization methods."""
        config = create_test_config()
        engine = GeneticAlgorithmEngine(config, simple_fitness_function)

        # Create test population with known objectives
        population = Population(config)
        for i in range(5):
            ind = Individual(chromosome=PolicyChromosome())
            ind.multi_objective_fitness = {
                "security": i * 0.2,
                "productivity": 1.0 - i * 0.2,
                "compliance": 0.5
            }
            population.individuals.append(ind)

        engine.current_population = population

        # Test minmax normalization
        config.fitness.normalization_method = "minmax"
        objectives = {"security": 0.6, "productivity": 0.4, "compliance": 0.5}
        normalized = engine._normalize_objectives(objectives)

        assert 0 <= normalized["security"] <= 1
        assert 0 <= normalized["productivity"] <= 1

    def test_termination_conditions(self, simple_fitness_function):
        """Test various termination conditions."""
        config = create_test_config()
        engine = GeneticAlgorithmEngine(config, simple_fitness_function)

        # Test runtime limit
        engine.start_time = datetime.now() - timedelta(hours=2)
        config.max_runtime = timedelta(hours=1)

        assert engine._should_terminate()

        # Test fitness target
        config.max_runtime = None
        config.fitness.objectives[0].target_value = 0.9
        config.fitness.objectives[0].minimize = False

        # Create population with best individual
        population = Population(config)
        best = Individual(chromosome=PolicyChromosome())
        best.multi_objective_fitness = {"security": 0.95}
        population.best_individual = best
        engine.current_population = population

        assert engine._should_terminate()

    def test_stagnation_detection(self, simple_fitness_function):
        """Test stagnation detection and handling."""
        config = create_test_config()
        config.evolution.stagnation_generations = 3
        engine = GeneticAlgorithmEngine(config, simple_fitness_function)

        # Create population with stagnant history
        population = Population(config)
        population.history = [
            {"best_fitness": 0.7},
            {"best_fitness": 0.7},
            {"best_fitness": 0.7},
            {"best_fitness": 0.7}
        ]
        engine.current_population = population

        assert engine._detect_stagnation()
        assert engine.stagnation_counter == 1

    @pytest.mark.asyncio
    async def test_checkpoint_save_load(self, simple_fitness_function):
        """Test checkpoint saving and loading."""
        config = create_test_config()
        engine = GeneticAlgorithmEngine(config, simple_fitness_function)

        # Create and evaluate a population
        population = Population(config)
        init_params = {
            "min_rules": 1,
            "max_rules": 3,
            "resource_pool": ["app:*"],
            "action_pool": ["read", "write"]
        }
        population.initialize_random(init_params)

        engine.current_population = population
        await engine._evaluate_population()

        # Save checkpoint
        checkpoint_path = engine.checkpoint_dir / "test_checkpoint.json"
        engine.current_population.save_snapshot(str(checkpoint_path))

        # Load checkpoint
        engine2 = GeneticAlgorithmEngine(config, simple_fitness_function)
        engine2.load_checkpoint(str(checkpoint_path))

        assert engine2.current_population is not None
        assert len(engine2.current_population.individuals) == len(population.individuals)

        # Cleanup
        checkpoint_path.unlink()

    @pytest.mark.asyncio
    async def test_parallel_evaluation(self, simple_fitness_function):
        """Test parallel fitness evaluation."""
        config = create_test_config()
        config.parallelization.enable_parallel = True
        config.parallelization.num_workers = 2
        config.parallelization.chunk_size = 5

        engine = GeneticAlgorithmEngine(config, simple_fitness_function)

        # Create larger population for parallel testing
        population = Population(config)
        for i in range(20):
            chromosome = PolicyChromosome()
            rule = PolicyRule(
                resource=Gene(GeneType.RESOURCE, f"resource{i}"),
                actions=[Gene(GeneType.ACTION, "read")],
                conditions=[],
                effect=Gene(GeneType.EFFECT, PolicyEffect.ALLOW.value),
                priority=Gene(GeneType.PRIORITY, 100)
            )
            chromosome.add_rule(rule)
            population.individuals.append(Individual(chromosome=chromosome))

        engine.current_population = population
        await engine._evaluate_population()

        # All individuals should be evaluated
        assert all(ind.evaluated for ind in population.individuals)

        # Cleanup executor
        if engine.executor:
            engine.executor.shutdown(wait=True)
