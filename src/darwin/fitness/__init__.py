"""
Darwin fitness functions for policy evaluation.

This module provides various fitness functions to evaluate policy chromosomes
based on different objectives like security, productivity, compliance, and risk.
"""

from typing import Dict, Any, List, Tuple, Optional
from abc import ABC, abstractmethod
import asyncio

from src.darwin.fitness.base import (
    FitnessFunction,
    FitnessMetrics,
    AsyncFitnessFunction
)

from src.darwin.fitness.security import (
    SecurityFitness,
    SecurityMetrics
)

from src.darwin.fitness.productivity import (
    ProductivityFitness,
    ProductivityMetrics
)

from src.darwin.fitness.compliance import (
    ComplianceFitness,
    ComplianceMetrics
)

from src.darwin.fitness.risk import (
    RiskFitness,
    RiskMetrics
)

from src.darwin.fitness.multi_objective import (
    MultiObjectiveFitness,
    ObjectiveWeight,
    ParetoFront
)

from src.darwin.fitness.coverage import (
    PolicyCoverageFitness,
    CoverageMetrics
)

from src.darwin.fitness.user_experience import (
    UserExperienceFitness,
    UXMetrics
)

from src.darwin.fitness.performance import (
    PerformanceFitness,
    PerformanceMetrics
)

__all__ = [
    # Base classes
    "FitnessFunction",
    "FitnessMetrics",
    "AsyncFitnessFunction",

    # Security
    "SecurityFitness",
    "SecurityMetrics",

    # Productivity
    "ProductivityFitness",
    "ProductivityMetrics",

    # Compliance
    "ComplianceFitness",
    "ComplianceMetrics",

    # Risk
    "RiskFitness",
    "RiskMetrics",

    # Multi-objective
    "MultiObjectiveFitness",
    "ObjectiveWeight",
    "ParetoFront",

    # Coverage
    "PolicyCoverageFitness",
    "CoverageMetrics",

    # User Experience
    "UserExperienceFitness",
    "UXMetrics",

    # Performance
    "PerformanceFitness",
    "PerformanceMetrics",
]
