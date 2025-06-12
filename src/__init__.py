"""
Cerby Identity Automation Platform - Source Package

This package contains all the core components for the identity automation platform
including API endpoints, data models, genetic algorithms, and analytics engines.
"""

__version__ = "1.0.0"
__author__ = "DevQ.ai Team"
__email__ = "dion@devq.ai"

# Package-level imports for convenience
from src.core.config import settings

__all__ = [
    "settings",
    "__version__",
    "__author__",
    "__email__"
]
