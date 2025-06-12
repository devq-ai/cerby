"""
Darwin Configuration Module.

This module defines configuration classes for the Darwin genetic algorithm framework,
including evolution parameters, optimization objectives, and system settings.
"""

from typing import Optional, Dict, Any, List, Literal
from pydantic import BaseModel, Field, ConfigDict, field_validator
from datetime import timedelta
import os


class EvolutionParameters(BaseModel):
    """Parameters controlling the genetic algorithm evolution process."""

    model_config = ConfigDict(validate_assignment=True)

    # Population parameters
    population_size: int = Field(
        default=100,
        ge=10,
        le=10000,
        description="Number of individuals in the population"
    )
    generations: int = Field(
        default=50,
        ge=1,
        le=1000,
        description="Maximum number of generations to evolve"
    )

    # Genetic operators
    mutation_rate: float = Field(
        default=0.1,
        ge=0.0,
        le=1.0,
        description="Probability of mutation for each gene"
    )
    crossover_rate: float = Field(
        default=0.8,
        ge=0.0,
        le=1.0,
        description="Probability of crossover between parents"
    )

    # Selection parameters
    elite_size: int = Field(
        default=10,
        ge=0,
        description="Number of best individuals to preserve"
    )
    tournament_size: int = Field(
        default=5,
        ge=2,
        description="Number of individuals in tournament selection"
    )
    selection_method: Literal["tournament", "roulette", "rank"] = Field(
        default="tournament",
        description="Selection method for choosing parents"
    )

    # Advanced parameters
    adaptive_mutation: bool = Field(
        default=True,
        description="Enable adaptive mutation rate based on diversity"
    )
    migration_rate: float = Field(
        default=0.05,
        ge=0.0,
        le=0.5,
        description="Rate of migration between sub-populations"
    )
    stagnation_generations: int = Field(
        default=10,
        ge=1,
        description="Generations without improvement before intervention"
    )

    @field_validator('elite_size')
    def validate_elite_size(cls, v, info):
        """Ensure elite size is less than population size."""
        if 'population_size' in info.data and v >= info.data['population_size']:
            raise ValueError('Elite size must be less than population size')
        return v


class ObjectiveConfig(BaseModel):
    """Configuration for a single optimization objective."""

    name: str = Field(description="Name of the objective")
    weight: float = Field(
        default=1.0,
        ge=0.0,
        description="Weight of this objective in fitness calculation"
    )
    minimize: bool = Field(
        default=False,
        description="Whether to minimize (True) or maximize (False) this objective"
    )
    target_value: Optional[float] = Field(
        default=None,
        description="Target value for this objective (optional)"
    )
    threshold: Optional[float] = Field(
        default=None,
        description="Minimum/maximum acceptable value"
    )


class FitnessConfig(BaseModel):
    """Configuration for fitness evaluation."""

    objectives: List[ObjectiveConfig] = Field(
        default_factory=lambda: [
            ObjectiveConfig(name="security", weight=0.4),
            ObjectiveConfig(name="productivity", weight=0.4),
            ObjectiveConfig(name="compliance", weight=0.2)
        ],
        description="List of optimization objectives"
    )
    penalty_factor: float = Field(
        default=0.1,
        ge=0.0,
        le=1.0,
        description="Penalty factor for constraint violations"
    )
    normalization_method: Literal["minmax", "zscore", "none"] = Field(
        default="minmax",
        description="Method for normalizing objective values"
    )


class ConstraintConfig(BaseModel):
    """Configuration for constraint handling."""

    max_rules_per_policy: int = Field(
        default=50,
        ge=1,
        le=200,
        description="Maximum number of rules in a policy"
    )
    min_rules_per_policy: int = Field(
        default=1,
        ge=1,
        description="Minimum number of rules in a policy"
    )
    max_conditions_per_rule: int = Field(
        default=10,
        ge=1,
        le=50,
        description="Maximum conditions in a single rule"
    )
    required_attributes: List[str] = Field(
        default_factory=list,
        description="Attributes that must be present in policies"
    )
    forbidden_combinations: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Forbidden attribute combinations"
    )


class LoggingConfig(BaseModel):
    """Configuration for logging and monitoring."""

    enable_logging: bool = Field(
        default=True,
        description="Enable detailed evolution logging"
    )
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = Field(
        default="INFO",
        description="Logging level"
    )
    log_interval: int = Field(
        default=10,
        ge=1,
        description="Generations between detailed logs"
    )
    save_snapshots: bool = Field(
        default=True,
        description="Save population snapshots during evolution"
    )
    snapshot_interval: int = Field(
        default=25,
        ge=1,
        description="Generations between snapshots"
    )
    metrics_export: bool = Field(
        default=True,
        description="Export metrics to monitoring system"
    )


class ParallelizationConfig(BaseModel):
    """Configuration for parallel processing."""

    enable_parallel: bool = Field(
        default=True,
        description="Enable parallel fitness evaluation"
    )
    num_workers: Optional[int] = Field(
        default=None,
        ge=1,
        description="Number of parallel workers (None for auto)"
    )
    chunk_size: int = Field(
        default=10,
        ge=1,
        description="Individuals per parallel chunk"
    )
    island_model: bool = Field(
        default=False,
        description="Use island model for parallel evolution"
    )
    num_islands: int = Field(
        default=4,
        ge=2,
        le=16,
        description="Number of islands if using island model"
    )


class DarwinConfig(BaseModel):
    """Main configuration class for Darwin framework."""

    model_config = ConfigDict(
        validate_assignment=True,
        extra='forbid'
    )

    # Sub-configurations
    evolution: EvolutionParameters = Field(
        default_factory=EvolutionParameters,
        description="Evolution parameters"
    )
    fitness: FitnessConfig = Field(
        default_factory=FitnessConfig,
        description="Fitness evaluation configuration"
    )
    constraints: ConstraintConfig = Field(
        default_factory=ConstraintConfig,
        description="Constraint handling configuration"
    )
    logging: LoggingConfig = Field(
        default_factory=LoggingConfig,
        description="Logging and monitoring configuration"
    )
    parallelization: ParallelizationConfig = Field(
        default_factory=ParallelizationConfig,
        description="Parallel processing configuration"
    )

    # General settings
    random_seed: Optional[int] = Field(
        default=None,
        description="Random seed for reproducibility"
    )
    checkpoint_dir: str = Field(
        default="./darwin_checkpoints",
        description="Directory for saving checkpoints"
    )
    max_runtime: Optional[timedelta] = Field(
        default=None,
        description="Maximum runtime for evolution"
    )

    @classmethod
    def from_env(cls) -> "DarwinConfig":
        """Create configuration from environment variables."""
        config_dict = {}

        # Evolution parameters from env
        if pop_size := os.getenv("DARWIN_POPULATION_SIZE"):
            config_dict.setdefault("evolution", {})["population_size"] = int(pop_size)
        if generations := os.getenv("DARWIN_GENERATIONS"):
            config_dict.setdefault("evolution", {})["generations"] = int(generations)
        if mutation_rate := os.getenv("DARWIN_MUTATION_RATE"):
            config_dict.setdefault("evolution", {})["mutation_rate"] = float(mutation_rate)
        if crossover_rate := os.getenv("DARWIN_CROSSOVER_RATE"):
            config_dict.setdefault("evolution", {})["crossover_rate"] = float(crossover_rate)
        if elite_size := os.getenv("DARWIN_ELITE_SIZE"):
            config_dict.setdefault("evolution", {})["elite_size"] = int(elite_size)

        # Parallelization from env
        if num_workers := os.getenv("DARWIN_NUM_WORKERS"):
            config_dict.setdefault("parallelization", {})["num_workers"] = int(num_workers)

        # General settings
        if random_seed := os.getenv("DARWIN_RANDOM_SEED"):
            config_dict["random_seed"] = int(random_seed)
        if checkpoint_dir := os.getenv("DARWIN_CHECKPOINT_DIR"):
            config_dict["checkpoint_dir"] = checkpoint_dir

        return cls(**config_dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return self.model_dump()

    def save(self, filepath: str) -> None:
        """Save configuration to JSON file."""
        import json
        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=2, default=str)

    @classmethod
    def load(cls, filepath: str) -> "DarwinConfig":
        """Load configuration from JSON file."""
        import json
        with open(filepath, 'r') as f:
            data = json.load(f)
        return cls(**data)

    def validate_consistency(self) -> None:
        """Validate configuration consistency across components."""
        # Check elite size vs population
        if self.evolution.elite_size >= self.evolution.population_size:
            raise ValueError(
                f"Elite size ({self.evolution.elite_size}) must be less than "
                f"population size ({self.evolution.population_size})"
            )

        # Check tournament size vs population
        if self.evolution.tournament_size > self.evolution.population_size:
            raise ValueError(
                f"Tournament size ({self.evolution.tournament_size}) must not exceed "
                f"population size ({self.evolution.population_size})"
            )

        # Check objective weights sum to 1.0 (if normalized)
        total_weight = sum(obj.weight for obj in self.fitness.objectives)
        if abs(total_weight - 1.0) > 0.001 and len(self.fitness.objectives) > 1:
            # Auto-normalize weights
            for obj in self.fitness.objectives:
                obj.weight /= total_weight

        # Check constraint consistency
        if self.constraints.min_rules_per_policy > self.constraints.max_rules_per_policy:
            raise ValueError(
                f"Minimum rules ({self.constraints.min_rules_per_policy}) cannot exceed "
                f"maximum rules ({self.constraints.max_rules_per_policy})"
            )


# Convenience functions
def create_default_config() -> DarwinConfig:
    """Create a default configuration suitable for most use cases."""
    return DarwinConfig()


def create_test_config() -> DarwinConfig:
    """Create a configuration suitable for testing (smaller, faster)."""
    return DarwinConfig(
        evolution=EvolutionParameters(
            population_size=20,
            generations=10,
            mutation_rate=0.2,
            crossover_rate=0.7,
            elite_size=2
        ),
        logging=LoggingConfig(
            log_interval=1,
            snapshot_interval=5
        ),
        parallelization=ParallelizationConfig(
            enable_parallel=False  # Disable for deterministic tests
        )
    )


def create_production_config() -> DarwinConfig:
    """Create a configuration suitable for production use."""
    return DarwinConfig(
        evolution=EvolutionParameters(
            population_size=500,
            generations=100,
            mutation_rate=0.05,
            crossover_rate=0.9,
            elite_size=50,
            adaptive_mutation=True
        ),
        fitness=FitnessConfig(
            objectives=[
                ObjectiveConfig(name="security", weight=0.5),
                ObjectiveConfig(name="productivity", weight=0.3),
                ObjectiveConfig(name="compliance", weight=0.2)
            ]
        ),
        parallelization=ParallelizationConfig(
            enable_parallel=True,
            island_model=True,
            num_islands=8
        ),
        logging=LoggingConfig(
            log_interval=10,
            snapshot_interval=20,
            metrics_export=True
        )
    )
