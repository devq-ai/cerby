"""
Unit tests for Data Transformation Pipeline (Subtask 3.5).

Tests cover:
- Data normalization
- Field mapping
- Data enrichment
- Format conversions
- Validation rules
- Transformation chains
"""

import pytest
from datetime import datetime, timedelta
from typing import Dict, List, Any
import json

from src.ingestion.transformers.base import (
    DataTransformer,
    TransformationRule,
    TransformationChain,
    ValidationRule
)
from src.ingestion.transformers import (
    FieldMapper,
    DataNormalizer,
    DataEnricher,
    FormatConverter,
    DataValidator
)
from src.db.models.saas_application import SaaSProvider


class TestDataTransformation:
    """Test suite for data transformation pipeline."""

    @pytest.fixture
    def sample_identity_data(self):
        """Sample identity data for transformation."""
        return {
            "email": "John.Doe@EXAMPLE.COM",
            "full_name": "john doe",
            "phone": "+1 (555) 123-4567",
            "dept": "eng",
            "start_date": "2024-01-15",
            "manager_email": "jane.smith@example.com",
            "location": "san francisco, ca"
        }

    @pytest.fixture
    def field_mapper(self):
        """Create a field mapper instance."""
        mapping_rules = {
            "dept": "department",
            "full_name": "display_name",
            "start_date": "hire_date",
            "phone": "phone_number"
        }
        return FieldMapper(mapping_rules)

    @pytest.fixture
    def data_normalizer(self):
        """Create a data normalizer instance."""
        return DataNormalizer()

    def test_field_mapping(self, field_mapper, sample_identity_data):
        """Test field name mapping."""
        transformed = field_mapper.transform(sample_identity_data)

        assert "department" in transformed
        assert transformed["department"] == "eng"
        assert "display_name" in transformed
        assert transformed["display_name"] == "john doe"
        assert "hire_date" in transformed
        assert "dept" not in transformed  # Original field removed

    def test_nested_field_mapping(self, field_mapper):
        """Test mapping nested fields."""
        mapper = FieldMapper({
            "user.name": "profile.full_name",
            "user.email": "contact.email",
            "user.dept": "organization.department"
        })

        data = {
            "user": {
                "name": "John Doe",
                "email": "john@example.com",
                "dept": "Engineering"
            }
        }

        transformed = mapper.transform(data)

        assert transformed["profile"]["full_name"] == "John Doe"
        assert transformed["contact"]["email"] == "john@example.com"
        assert transformed["organization"]["department"] == "Engineering"

    def test_data_normalization(self, data_normalizer, sample_identity_data):
        """Test data normalization."""
        transformed = data_normalizer.transform(sample_identity_data)

        # Email should be lowercase
        assert transformed["email"] == "john.doe@example.com"

        # Name should be title case
        assert transformed["full_name"] == "John Doe"

        # Phone should be normalized
        assert transformed["phone"] == "+15551234567"

        # Location should be title case
        assert transformed["location"] == "San Francisco, CA"

    def test_date_normalization(self, data_normalizer):
        """Test date format normalization."""
        data = {
            "date1": "2024-01-15",
            "date2": "01/15/2024",
            "date3": "15-Jan-2024",
            "date4": "January 15, 2024",
            "date5": "2024-01-15T10:30:00Z"
        }

        transformed = data_normalizer.transform(data)

        # All dates should be in ISO format
        for key in data.keys():
            assert transformed[key].startswith("2024-01-15")

    def test_data_enrichment(self):
        """Test data enrichment functionality."""
        enricher = DataEnricher({
            "department_mappings": {
                "eng": "Engineering",
                "sales": "Sales",
                "mkt": "Marketing",
                "hr": "Human Resources"
            },
            "location_mappings": {
                "sf": "San Francisco",
                "nyc": "New York City",
                "la": "Los Angeles"
            }
        })

        data = {
            "email": "user@example.com",
            "department": "eng",
            "location": "sf"
        }

        enriched = enricher.transform(data)

        assert enriched["department"] == "Engineering"
        assert enriched["location"] == "San Francisco"
        assert enriched["department_code"] == "eng"  # Original preserved
        assert enriched["location_code"] == "sf"

    def test_computed_fields(self):
        """Test computed field generation."""
        enricher = DataEnricher()

        data = {
            "first_name": "John",
            "last_name": "Doe",
            "email": "john.doe@example.com",
            "department": "Engineering"
        }

        # Add computed fields
        enricher.add_computed_field(
            "username",
            lambda d: d["email"].split("@")[0]
        )
        enricher.add_computed_field(
            "full_name",
            lambda d: f"{d['first_name']} {d['last_name']}"
        )
        enricher.add_computed_field(
            "email_domain",
            lambda d: d["email"].split("@")[1]
        )

        enriched = enricher.transform(data)

        assert enriched["username"] == "john.doe"
        assert enriched["full_name"] == "John Doe"
        assert enriched["email_domain"] == "example.com"

    def test_format_conversion(self):
        """Test format conversion between providers."""
        # Okta to SCIM format converter
        converter = FormatConverter(
            source_format="okta",
            target_format="scim"
        )

        okta_user = {
            "id": "00u123456",
            "status": "ACTIVE",
            "profile": {
                "firstName": "John",
                "lastName": "Doe",
                "email": "john.doe@example.com",
                "login": "john.doe@example.com"
            }
        }

        scim_user = converter.transform(okta_user)

        assert scim_user["schemas"] == ["urn:ietf:params:scim:schemas:core:2.0:User"]
        assert scim_user["id"] == "00u123456"
        assert scim_user["userName"] == "john.doe@example.com"
        assert scim_user["name"]["givenName"] == "John"
        assert scim_user["name"]["familyName"] == "Doe"
        assert scim_user["emails"][0]["value"] == "john.doe@example.com"
        assert scim_user["active"] is True

    def test_provider_specific_transformations(self):
        """Test provider-specific transformation rules."""
        # Google Workspace transformer
        google_transformer = DataTransformer()
        google_transformer.add_rule(
            TransformationRule(
                field="primaryEmail",
                target_field="email",
                transform=lambda x: x.lower()
            )
        )
        google_transformer.add_rule(
            TransformationRule(
                field="name",
                target_field="display_name",
                transform=lambda x: f"{x.get('givenName', '')} {x.get('familyName', '')}"
            )
        )

        google_user = {
            "id": "118200000000000000000",
            "primaryEmail": "John.Doe@Example.com",
            "name": {
                "givenName": "John",
                "familyName": "Doe"
            },
            "suspended": False
        }

        transformed = google_transformer.transform(google_user)

        assert transformed["email"] == "john.doe@example.com"
        assert transformed["display_name"] == "John Doe"
        assert transformed["is_active"] is True  # Inverted suspended flag

    def test_validation_rules(self):
        """Test data validation during transformation."""
        validator = DataValidator()

        # Add validation rules
        validator.add_rule(
            ValidationRule(
                field="email",
                validator=lambda x: "@" in x and "." in x.split("@")[1],
                error_message="Invalid email format"
            )
        )
        validator.add_rule(
            ValidationRule(
                field="department",
                validator=lambda x: x in ["Engineering", "Sales", "Marketing", "HR"],
                error_message="Invalid department"
            )
        )
        validator.add_rule(
            ValidationRule(
                field="phone",
                validator=lambda x: x.startswith("+") and len(x) >= 10,
                error_message="Invalid phone number",
                required=False
            )
        )

        # Valid data
        valid_data = {
            "email": "test@example.com",
            "department": "Engineering",
            "phone": "+15551234567"
        }

        result = validator.validate(valid_data)
        assert result.is_valid
        assert len(result.errors) == 0

        # Invalid data
        invalid_data = {
            "email": "invalid-email",
            "department": "InvalidDept",
            "phone": "555-1234"
        }

        result = validator.validate(invalid_data)
        assert not result.is_valid
        assert len(result.errors) == 3

    def test_transformation_chain(self):
        """Test chaining multiple transformations."""
        chain = TransformationChain()

        # Add transformers in order
        chain.add_transformer(FieldMapper({
            "dept": "department",
            "phone": "phone_number"
        }))
        chain.add_transformer(DataNormalizer())
        chain.add_transformer(DataEnricher({
            "department_mappings": {
                "eng": "Engineering",
                "engineering": "Engineering"
            }
        }))
        chain.add_transformer(DataValidator([
            ValidationRule("email", lambda x: "@" in x),
            ValidationRule("department", lambda x: len(x) > 0)
        ]))

        data = {
            "email": "JOHN.DOE@EXAMPLE.COM",
            "dept": "eng",
            "phone": "+1 (555) 123-4567"
        }

        result = chain.transform(data)

        assert result["email"] == "john.doe@example.com"
        assert result["department"] == "Engineering"
        assert result["phone_number"] == "+15551234567"

    def test_conditional_transformations(self):
        """Test conditional transformation rules."""
        transformer = DataTransformer()

        # Add conditional rule
        def conditional_transform(data):
            if data.get("country") == "US":
                return "+1" + data.get("phone", "").replace("-", "")
            elif data.get("country") == "UK":
                return "+44" + data.get("phone", "").replace("-", "")
            return data.get("phone", "")

        transformer.add_rule(
            TransformationRule(
                field="phone",
                transform=conditional_transform,
                condition=lambda d: "country" in d
            )
        )

        # US number
        us_data = {"phone": "555-123-4567", "country": "US"}
        transformed_us = transformer.transform(us_data)
        assert transformed_us["phone"] == "+15551234567"

        # UK number
        uk_data = {"phone": "20-7123-4567", "country": "UK"}
        transformed_uk = transformer.transform(uk_data)
        assert transformed_uk["phone"] == "+442071234567"

        # No country
        no_country = {"phone": "555-123-4567"}
        transformed_no = transformer.transform(no_country)
        assert transformed_no["phone"] == "555-123-4567"

    def test_bulk_transformation_performance(self):
        """Test performance of bulk transformations."""
        import time

        transformer = TransformationChain()
        transformer.add_transformer(FieldMapper({"dept": "department"}))
        transformer.add_transformer(DataNormalizer())
        transformer.add_transformer(DataValidator([
            ValidationRule("email", lambda x: "@" in x)
        ]))

        # Generate large dataset
        data_list = []
        for i in range(1000):
            data_list.append({
                "email": f"user{i}@example.com",
                "dept": "eng" if i % 2 == 0 else "sales",
                "name": f"User {i}"
            })

        start_time = time.time()

        # Transform all data
        results = transformer.transform_batch(data_list)

        end_time = time.time()
        duration = end_time - start_time

        assert len(results) == 1000
        assert all(r["department"] in ["eng", "sales"] for r in results)
        assert duration < 1.0  # Should process 1000 records in under 1 second

    def test_error_handling_in_transformation(self):
        """Test error handling during transformation."""
        transformer = DataTransformer()

        # Add rule that might fail
        def risky_transform(value):
            return int(value) * 2  # Will fail for non-numeric strings

        transformer.add_rule(
            TransformationRule(
                field="count",
                transform=risky_transform,
                on_error="skip"  # Skip field on error
            )
        )

        data = {
            "count": "not-a-number",
            "other_field": "value"
        }

        result = transformer.transform(data)

        assert "count" not in result  # Field skipped due to error
        assert result["other_field"] == "value"

    def test_custom_transformation_functions(self):
        """Test custom transformation functions."""
        def title_case_with_exceptions(value):
            exceptions = ["of", "and", "the", "in", "on"]
            words = value.lower().split()
            result = []

            for i, word in enumerate(words):
                if i == 0 or word not in exceptions:
                    result.append(word.capitalize())
                else:
                    result.append(word)

            return " ".join(result)

        transformer = DataTransformer()
        transformer.add_rule(
            TransformationRule(
                field="title",
                transform=title_case_with_exceptions
            )
        )

        data = {
            "title": "director of engineering and product development"
        }

        result = transformer.transform(data)
        assert result["title"] == "Director of Engineering and Product Development"

    def test_transformation_metadata(self):
        """Test capturing transformation metadata."""
        transformer = DataTransformer(capture_metadata=True)

        transformer.add_rule(
            TransformationRule(
                field="email",
                transform=lambda x: x.lower(),
                metadata_key="email_normalized"
            )
        )

        data = {"email": "JOHN.DOE@EXAMPLE.COM"}
        result, metadata = transformer.transform_with_metadata(data)

        assert result["email"] == "john.doe@example.com"
        assert metadata["email_normalized"] is True
        assert metadata["transformation_timestamp"] is not None
        assert metadata["rules_applied"] == 1
