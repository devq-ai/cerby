"""
Ingestion package for Cerby Identity Automation Platform.

This package handles the ingestion of identity data from various SaaS applications
including synthetic data generation, SCIM endpoints, webhook processing, and batch imports.
"""

from src.ingestion.base import (
    BaseIngestionHandler,
    IngestionResult,
    IngestionError,
    IngestionStatus
)

from src.ingestion.synthetic import SyntheticDataGenerator
from src.ingestion.scim import SCIMHandler
from src.ingestion.webhook import WebhookHandler
from src.ingestion.batch import BatchImporter
from src.ingestion.streaming import StreamProcessor

__all__ = [
    # Base classes
    "BaseIngestionHandler",
    "IngestionResult",
    "IngestionError",
    "IngestionStatus",

    # Handlers
    "SyntheticDataGenerator",
    "SCIMHandler",
    "WebhookHandler",
    "BatchImporter",
    "StreamProcessor",
]
