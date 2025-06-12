"""
Unit tests for Synthetic Data Generator (Subtask 3.1).

Tests cover:
- Synthetic identity generation
- Provider-specific data formats
- Bulk data generation
- Data consistency and validation
- Realistic data patterns
"""

import pytest
from datetime import datetime, timedelta
from typing import List, Dict
import json

from src.ingestion.synthetic import (
    SyntheticDataGenerator,
    IdentityGenerator,
    EventGenerator,
    PolicyGenerator,
    DataProfile
)
from src.db.models.saas_application import SaaSProvider


class TestSyntheticDataGenerator:
    """Test suite for synthetic data generation."""

    @pytest.fixture
    def generator(self):
        """Create a synthetic data generator instance."""
        return SyntheticDataGenerator(seed=42)  # Fixed seed for reproducibility

    def test_generator_initialization(self, generator):
        """Test generator initialization with seed."""
        assert generator is not None
        assert generator.seed == 42

        # Test reproducibility with same seed
        gen2 = SyntheticDataGenerator(seed=42)
        data1 = generator.generate_identity()
        data2 = gen2.generate_identity()

        assert data1["email"] == data2["email"]

    def test_generate_single_identity(self, generator):
        """Test generating a single identity."""
        identity = generator.generate_identity(provider=SaaSProvider.OKTA)

        # Validate required fields
        assert "external_id" in identity
        assert "email" in identity
        assert "username" in identity
        assert "provider" in identity
        assert identity["provider"] == SaaSProvider.OKTA

        # Validate email format
        assert "@" in identity["email"]
        assert identity["email"].count("@") == 1

        # Validate external ID format
        assert identity["external_id"].startswith("okta_")

    def test_generate_bulk_identities(self, generator):
        """Test bulk identity generation."""
        count = 100
        identities = generator.generate_identities(count=count)

        assert len(identities) == count

        # Check uniqueness
        emails = [i["email"] for i in identities]
        assert len(set(emails)) == count  # All emails should be unique

        external_ids = [i["external_id"] for i in identities]
        assert len(set(external_ids)) == count  # All external IDs should be unique

    def test_provider_specific_formats(self, generator):
        """Test provider-specific data formats."""
        providers = [
            SaaSProvider.OKTA,
            SaaSProvider.GOOGLE,
            SaaSProvider.MICROSOFT,
            SaaSProvider.GITHUB,
            SaaSProvider.SLACK
        ]

        for provider in providers:
            identity = generator.generate_identity(provider=provider)

            # Provider-specific validations
            if provider == SaaSProvider.OKTA:
                assert identity["external_id"].startswith("00u")
                assert "profile" in identity["attributes"]

            elif provider == SaaSProvider.GOOGLE:
                assert identity["external_id"].isdigit()
                assert len(identity["external_id"]) == 21
                assert "primaryEmail" in identity["attributes"]

            elif provider == SaaSProvider.MICROSOFT:
                # Microsoft uses GUIDs
                assert "-" in identity["external_id"]
                assert len(identity["external_id"]) == 36

            elif provider == SaaSProvider.GITHUB:
                assert identity["external_id"].isdigit()
                assert "login" in identity["attributes"]

            elif provider == SaaSProvider.SLACK:
                assert identity["external_id"].startswith("U")
                assert "team_id" in identity["attributes"]

    def test_realistic_attribute_generation(self, generator):
        """Test realistic attribute patterns."""
        identity = generator.generate_identity()
        attributes = identity.get("attributes", {})

        # Check common attributes
        assert "department" in attributes
        assert attributes["department"] in [
            "Engineering", "Sales", "Marketing", "HR",
            "Finance", "Operations", "Support", "Product"
        ]

        assert "title" in attributes
        assert "manager" in attributes
        assert "location" in attributes
        assert "employeeType" in attributes

    def test_identity_relationships(self, generator):
        """Test generating related identities."""
        # Generate a manager
        manager = generator.generate_identity()
        manager["attributes"]["title"] = "Engineering Manager"

        # Generate team members
        team = generator.generate_team(manager=manager["email"], size=5)

        assert len(team) == 5
        for member in team:
            assert member["attributes"]["manager"] == manager["email"]
            assert member["attributes"]["department"] == manager["attributes"]["department"]

    def test_temporal_data_generation(self, generator):
        """Test time-based data generation."""
        # Generate identities with creation dates
        identities = generator.generate_identities_over_time(
            count=50,
            start_date=datetime.utcnow() - timedelta(days=30),
            end_date=datetime.utcnow()
        )

        assert len(identities) == 50

        # Check temporal distribution
        dates = [i["created_at"] for i in identities]
        assert min(dates) >= datetime.utcnow() - timedelta(days=30)
        assert max(dates) <= datetime.utcnow()

    def test_event_generation(self, generator):
        """Test identity event generation."""
        event_gen = EventGenerator(generator)

        # Generate identity first
        identity = generator.generate_identity()

        # Generate events for the identity
        events = event_gen.generate_events_for_identity(
            identity=identity,
            event_count=10
        )

        assert len(events) == 10

        # Validate event structure
        for event in events:
            assert "event_id" in event
            assert "event_type" in event
            assert "timestamp" in event
            assert "provider" in event
            assert "external_id" in event
            assert event["external_id"] == identity["external_id"]

    def test_data_consistency(self, generator):
        """Test data consistency across generations."""
        # Generate correlated data
        org_data = generator.generate_organization(
            name="TestCorp",
            employee_count=100
        )

        assert len(org_data["identities"]) == 100
        assert len(org_data["departments"]) > 0
        assert len(org_data["managers"]) > 0

        # Verify org structure consistency
        for identity in org_data["identities"]:
            dept = identity["attributes"]["department"]
            assert dept in org_data["departments"]

            if identity["attributes"].get("manager"):
                assert identity["attributes"]["manager"] in [
                    i["email"] for i in org_data["identities"]
                ]

    def test_custom_data_profiles(self, generator):
        """Test custom data generation profiles."""
        # Create a startup profile
        startup_profile = DataProfile(
            name="startup",
            size_range=(10, 50),
            departments=["Engineering", "Product", "Sales"],
            titles=["Engineer", "Senior Engineer", "Product Manager", "Sales Rep"],
            providers=[SaaSProvider.GITHUB, SaaSProvider.SLACK, SaaSProvider.GOOGLE]
        )

        data = generator.generate_with_profile(startup_profile)

        assert 10 <= len(data["identities"]) <= 50
        for identity in data["identities"]:
            assert identity["attributes"]["department"] in startup_profile.departments

    def test_compliance_data_generation(self, generator):
        """Test generating compliance-relevant data."""
        # Generate data with compliance flags
        identity = generator.generate_identity(
            include_compliance_data=True
        )

        assert "compliance" in identity["attributes"]
        assert "gdpr_consent" in identity["attributes"]["compliance"]
        assert "sox_relevant" in identity["attributes"]["compliance"]
        assert "data_classification" in identity["attributes"]["compliance"]

    def test_error_scenarios_generation(self, generator):
        """Test generating error scenarios for testing."""
        # Generate identities with issues
        problematic_data = generator.generate_problematic_identities(count=10)

        issues_found = []
        for identity in problematic_data:
            if "issue" in identity:
                issues_found.append(identity["issue"])

        # Should include various issues
        assert "duplicate_email" in issues_found
        assert "invalid_email" in issues_found
        assert "missing_required_field" in issues_found

    def test_performance_data_generation(self, generator):
        """Test performance of bulk data generation."""
        import time

        start_time = time.time()

        # Generate large dataset
        data = generator.generate_identities(count=1000)

        end_time = time.time()
        generation_time = end_time - start_time

        assert len(data) == 1000
        assert generation_time < 5.0  # Should complete within 5 seconds

    def test_export_formats(self, generator):
        """Test exporting generated data in various formats."""
        identities = generator.generate_identities(count=10)

        # Test JSON export
        json_data = generator.export_json(identities)
        assert json.loads(json_data)  # Should be valid JSON

        # Test CSV export
        csv_data = generator.export_csv(identities)
        assert "email,username,provider" in csv_data

        # Test SCIM format export
        scim_data = generator.export_scim(identities[0])
        assert scim_data["schemas"] == ["urn:ietf:params:scim:schemas:core:2.0:User"]

    def test_data_statistics(self, generator):
        """Test generating statistics about generated data."""
        data = generator.generate_identities(count=1000)
        stats = generator.calculate_statistics(data)

        assert "total_count" in stats
        assert stats["total_count"] == 1000

        assert "providers" in stats
        assert "departments" in stats
        assert "average_attributes_per_identity" in stats

        # Verify distribution
        assert sum(stats["providers"].values()) == 1000

    def test_seed_data_for_scenarios(self, generator):
        """Test generating specific scenario data."""
        scenarios = {
            "onboarding": generator.generate_onboarding_scenario(new_hires=5),
            "offboarding": generator.generate_offboarding_scenario(departures=3),
            "reorg": generator.generate_reorg_scenario(
                from_dept="Engineering",
                to_dept="Product",
                count=10
            ),
            "security_incident": generator.generate_security_incident_scenario(
                affected_users=20
            )
        }

        # Validate onboarding scenario
        assert len(scenarios["onboarding"]["identities"]) == 5
        assert all(i["status"] == "active" for i in scenarios["onboarding"]["identities"])

        # Validate offboarding scenario
        assert len(scenarios["offboarding"]["identities"]) == 3
        assert all(i["status"] == "suspended" for i in scenarios["offboarding"]["identities"])

        # Validate reorg scenario
        assert len(scenarios["reorg"]["events"]) == 10
        assert all(
            e["data"]["old_department"] == "Engineering" and
            e["data"]["new_department"] == "Product"
            for e in scenarios["reorg"]["events"]
        )
