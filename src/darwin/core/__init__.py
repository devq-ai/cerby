"""
Darwin Core Module - Genetic Algorithm Components.

This module contains the core components of the Darwin genetic algorithm framework,
including configuration, chromosome representation, population management, and the
main evolution engine.
"""

from src.darwin.core.config import (
    DarwinConfig,
    EvolutionParameters,
    ObjectiveConfig,
    FitnessConfig,
    ConstraintConfig,
    LoggingConfig,
    ParallelizationConfig,
    create_default_config,
    create_test_config,
    create_production_config
)

from src.darwin.core.chromosome import (
    PolicyChromosome,
    PolicyRule,
    Gene,
    GeneType,
    ConditionOperator,
    PolicyEffect
)

from src.darwin.core.population import (
    Population,
    Individual
)

from src.darwin.core.engine import (
    GeneticAlgorithmEngine
)

__all__ = [
    # Configuration
    "DarwinConfig",
    "EvolutionParameters",
    "ObjectiveConfig",
    "FitnessConfig",
    "ConstraintConfig",
    "LoggingConfig",
    "ParallelizationConfig",
    "create_default_config",
    "create_test_config",
    "create_production_config",

    # Chromosome representation
    "PolicyChromosome",
    "PolicyRule",
    "Gene",
    "GeneType",
    "ConditionOperator",
    "PolicyEffect",

    # Population management
    "Population",
    "Individual",

    # Engine
    "GeneticAlgorithmEngine"
]
