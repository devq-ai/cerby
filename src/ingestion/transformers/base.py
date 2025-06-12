"""
Base transformer class for identity data transformation.

This module provides the foundation for all data transformers used in the
ingestion pipeline to normalize, validate, and enrich identity data.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional, Callable, Union
from datetime import datetime
import re
import logfire

from src.db.models.identity import IdentityProvider


class TransformationError(Exception):
    """Exception raised during data transformation."""

    def __init__(self, message: str, field: Optional[str] = None,
                 value: Any = None, provider: Optional[str] = None):
        self.field = field
        self.value = value
        self.provider = provider
        super().__init__(message)


class BaseTransformer(ABC):
    """
    Abstract base class for data transformers.

    Provides common functionality for transforming identity data
    from various providers into a normalized internal format.
    """

    def __init__(self, provider: Optional[IdentityProvider] = None):
        self.provider = provider
        self.transformation_rules: Dict[str, Callable] = {}
        self.field_mappings: Dict[str, str] = {}
        self.required_fields: List[str] = []
        self.validation_rules: Dict[str, Callable] = {}

    @abstractmethod
    def transform(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Transform input data to normalized format.

        Args:
            data: Raw input data

        Returns:
            Transformed data in normalized format

        Raises:
            TransformationError: If transformation fails
        """
        pass

    def add_transformation_rule(self, field: str, transformer: Callable) -> None:
        """Add a transformation rule for a specific field."""
        self.transformation_rules[field] = transformer

    def add_field_mapping(self, source_field: str, target_field: str) -> None:
        """Add a field mapping from source to target."""
        self.field_mappings[source_field] = target_field

    def add_validation_rule(self, field: str, validator: Callable) -> None:
        """Add a validation rule for a specific field."""
        self.validation_rules[field] = validator

    def apply_field_mappings(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Apply field mappings to transform field names."""
        transformed = {}

        for source_field, value in data.items():
            target_field = self.field_mappings.get(source_field, source_field)
            transformed[target_field] = value

        return transformed

    def apply_transformations(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Apply transformation rules to field values."""
        transformed = data.copy()

        for field, transformer in self.transformation_rules.items():
            if field in transformed:
                try:
                    transformed[field] = transformer(transformed[field])
                except Exception as e:
                    raise TransformationError(
                        f"Failed to transform field {field}: {str(e)}",
                        field=field,
                        value=transformed[field],
                        provider=self.provider
                    )

        return transformed

    def validate_data(self, data: Dict[str, Any]) -> None:
        """
        Validate transformed data.

        Args:
            data: Data to validate

        Raises:
            TransformationError: If validation fails
        """
        # Check required fields
        for field in self.required_fields:
            if field not in data or data[field] is None:
                raise TransformationError(
                    f"Required field '{field}' is missing or null",
                    field=field,
                    provider=self.provider
                )

        # Apply validation rules
        for field, validator in self.validation_rules.items():
            if field in data:
                try:
                    if not validator(data[field]):
                        raise TransformationError(
                            f"Validation failed for field '{field}'",
                            field=field,
                            value=data[field],
                            provider=self.provider
                        )
                except Exception as e:
                    if isinstance(e, TransformationError):
                        raise
                    raise TransformationError(
                        f"Validation error for field '{field}': {str(e)}",
                        field=field,
                        value=data.get(field),
                        provider=self.provider
                    )

    def normalize_email(self, email: Optional[str]) -> Optional[str]:
        """Normalize email address."""
        if not email:
            return None

        # Convert to lowercase and strip whitespace
        email = email.lower().strip()

        # Basic validation
        if not re.match(r'^[^@]+@[^@]+\.[^@]+$', email):
            raise TransformationError(
                f"Invalid email format: {email}",
                field="email",
                value=email
            )

        return email

    def normalize_phone(self, phone: Optional[str]) -> Optional[str]:
        """Normalize phone number."""
        if not phone:
            return None

        # Remove all non-digit characters
        digits = re.sub(r'\D', '', phone)

        # Format based on length (US numbers)
        if len(digits) == 10:
            return f"+1{digits}"
        elif len(digits) == 11 and digits.startswith('1'):
            return f"+{digits}"
        elif len(digits) > 11:
            return f"+{digits}"  # International
        else:
            return phone  # Keep original if can't normalize

    def normalize_name(self, name: Optional[str]) -> Optional[str]:
        """Normalize name (remove extra spaces, proper case)."""
        if not name:
            return None

        # Remove extra spaces
        name = ' '.join(name.split())

        # Title case
        return name.title()

    def parse_boolean(self, value: Any) -> bool:
        """Parse various boolean representations."""
        if isinstance(value, bool):
            return value

        if isinstance(value, str):
            value = value.lower().strip()
            if value in ('true', 'yes', '1', 'on', 'enabled', 'active'):
                return True
            elif value in ('false', 'no', '0', 'off', 'disabled', 'inactive'):
                return False

        if isinstance(value, (int, float)):
            return bool(value)

        raise TransformationError(
            f"Cannot parse boolean from value: {value}",
            value=value
        )

    def parse_datetime(self, value: Any) -> Optional[datetime]:
        """Parse datetime from various formats."""
        if not value:
            return None

        if isinstance(value, datetime):
            return value

        if isinstance(value, str):
            # Try common datetime formats
            formats = [
                "%Y-%m-%dT%H:%M:%S.%fZ",
                "%Y-%m-%dT%H:%M:%SZ",
                "%Y-%m-%dT%H:%M:%S",
                "%Y-%m-%d %H:%M:%S",
                "%Y-%m-%d",
                "%m/%d/%Y",
                "%d/%m/%Y",
            ]

            for fmt in formats:
                try:
                    return datetime.strptime(value, fmt)
                except ValueError:
                    continue

            # Try ISO format
            try:
                return datetime.fromisoformat(value.replace('Z', '+00:00'))
            except ValueError:
                pass

        raise TransformationError(
            f"Cannot parse datetime from value: {value}",
            value=value
        )

    def extract_nested_value(self, data: Dict[str, Any], path: str,
                           default: Any = None) -> Any:
        """
        Extract value from nested dictionary using dot notation.

        Args:
            data: Dictionary to extract from
            path: Dot-separated path (e.g., "user.profile.email")
            default: Default value if path not found

        Returns:
            Extracted value or default
        """
        keys = path.split('.')
        value = data

        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return default

        return value

    def flatten_dict(self, data: Dict[str, Any], prefix: str = '',
                    separator: str = '_') -> Dict[str, Any]:
        """
        Flatten nested dictionary.

        Args:
            data: Dictionary to flatten
            prefix: Prefix for keys
            separator: Separator between nested keys

        Returns:
            Flattened dictionary
        """
        flattened = {}

        for key, value in data.items():
            new_key = f"{prefix}{separator}{key}" if prefix else key

            if isinstance(value, dict):
                flattened.update(
                    self.flatten_dict(value, new_key, separator)
                )
            elif isinstance(value, list):
                # Convert list to indexed keys
                for i, item in enumerate(value):
                    if isinstance(item, dict):
                        flattened.update(
                            self.flatten_dict(item, f"{new_key}_{i}", separator)
                        )
                    else:
                        flattened[f"{new_key}_{i}"] = item
            else:
                flattened[new_key] = value

        return flattened

    def merge_names(self, first_name: Optional[str], last_name: Optional[str],
                   middle_name: Optional[str] = None) -> Optional[str]:
        """Merge name components into display name."""
        parts = []

        if first_name:
            parts.append(first_name.strip())
        if middle_name:
            parts.append(middle_name.strip())
        if last_name:
            parts.append(last_name.strip())

        return ' '.join(parts) if parts else None

    def extract_username_from_email(self, email: str) -> str:
        """Extract username from email address."""
        if '@' in email:
            return email.split('@')[0]
        return email

    def sanitize_string(self, value: Optional[str],
                       max_length: Optional[int] = None) -> Optional[str]:
        """
        Sanitize string value.

        Args:
            value: String to sanitize
            max_length: Maximum allowed length

        Returns:
            Sanitized string
        """
        if not value:
            return None

        # Remove control characters
        value = ''.join(char for char in value if ord(char) >= 32 or char in '\n\r\t')

        # Strip whitespace
        value = value.strip()

        # Truncate if needed
        if max_length and len(value) > max_length:
            value = value[:max_length]

        return value or None

    def transform_batch(self, data_list: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Transform a batch of records.

        Args:
            data_list: List of records to transform

        Returns:
            List of transformed records
        """
        transformed = []
        errors = []

        for i, data in enumerate(data_list):
            try:
                transformed.append(self.transform(data))
            except TransformationError as e:
                errors.append({
                    'index': i,
                    'error': str(e),
                    'field': e.field,
                    'value': e.value
                })
                logfire.error(
                    "Batch transformation error",
                    index=i,
                    error=str(e),
                    provider=self.provider
                )

        if errors:
            logfire.warning(
                "Batch transformation completed with errors",
                total=len(data_list),
                success=len(transformed),
                errors=len(errors)
            )

        return transformed
