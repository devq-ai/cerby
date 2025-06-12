"""
Darwin Genetic Algorithm Framework for Cerby Identity Automation Platform.

This module implements genetic algorithm-based optimization for access policies,
enabling automatic evolution of security rules that balance multiple objectives:
security, productivity, and compliance.
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
from src.darwin.core.population import Population, Individual
from src.darwin.core.engine import GeneticAlgorithmEngine
from src.darwin.core.chromosome import (
    PolicyChromosome,
    PolicyRule,
    Gene,
    GeneType,
    ConditionOperator,
    PolicyEffect
)

__version__ = "1.0.0"

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
    # Population
    "Population",
    "Individual",
    # Engine
    "GeneticAlgorithmEngine",
    # Chromosome
    "PolicyChromosome",
    "PolicyRule",
    "Gene",
    "GeneType",
    "ConditionOperator",
    "PolicyEffect",
    # Default config
    "DEFAULT_CONFIG"
]

# Default configuration for quick start
DEFAULT_CONFIG = None  # Will be initialized after imports

def _create_default_config():
    """Create default configuration after all imports are resolved."""
    global DEFAULT_CONFIG
    if DEFAULT_CONFIG is None:
        DEFAULT_CONFIG = DarwinConfig()
    return DEFAULT_CONFIG

# Initialize default config
DEFAULT_CONFIG = _create_default_config()
