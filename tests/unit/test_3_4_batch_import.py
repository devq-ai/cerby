"""
Unit tests for Batch Import functionality (Subtask 3.4).

Tests cover:
- CSV import functionality
- JSON import functionality
- Excel import functionality
- Data validation and transformation
- Error handling and reporting
- Progress tracking
"""

import pytest
from datetime import datetime
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session
import json
import csv
import io
from pathlib import Path
import pandas as pd

from src.ingestion.batch import BatchImporter, ImportJob, ImportStatus, ImportFormat
from src.db.models.identity import Identity
from src.db.models.saas_application import SaaSApplication, SaaSProvider
from src.db.models.audit import AuditLog


class TestBatchImport:
    """Test suite for batch import functionality."""

    @pytest.fixture
    def batch_importer(self, db_session: Session):
        """Create a batch importer instance."""
        return BatchImporter(db_session)

    @pytest.fixture
    def sample_csv_data(self):
        """Create sample CSV data for testing."""
        csv_data = """email,username,external_id,department,title,manager
john.doe@example.com,jdoe,okta_001,Engineering,Senior Engineer,manager@example.com
jane.smith@example.com,jsmith,okta_002,Sales,Sales Manager,director@example.com
bob.johnson@example.com,bjohnson,okta_003,Marketing,Marketing Analyst,manager2@example.com
alice.williams@example.com,awilliams,okta_004,Engineering,Junior Engineer,jdoe@example.com
charlie.brown@example.com,cbrown,okta_005,HR,HR Specialist,hr_manager@example.com"""
        return csv_data

    @pytest.fixture
    def sample_json_data(self):
        """Create sample JSON data for testing."""
        return {
            "identities": [
                {
                    "email": "user1@example.com",
                    "username": "user1",
                    "external_id": "google_001",
                    "attributes": {
                        "department": "Engineering",
                        "title": "Software Engineer",
                        "location": "San Francisco"
                    }
                },
                {
                    "email": "user2@example.com",
                    "username": "user2",
                    "external_id": "google_002",
                    "attributes": {
                        "department": "Product",
                        "title": "Product Manager",
                        "location": "New York"
                    }
                }
            ]
        }

    def test_csv_import_success(self, client: TestClient, batch_importer, sample_csv_data):
        """Test successful CSV import."""
        # Create import job
        files = {"file": ("identities.csv", io.StringIO(sample_csv_data), "text/csv")}

        response = client.post(
            "/api/v1/import/csv",
            files=files,
            data={"provider": "okta"}
        )

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "completed"
        assert data["total_records"] == 5
        assert data["successful_records"] == 5
        assert data["failed_records"] == 0

    def test_json_import_success(self, client: TestClient, batch_importer, sample_json_data):
        """Test successful JSON import."""
        response = client.post(
            "/api/v1/import/json",
            json={
                "provider": "google",
                "data": sample_json_data
            }
        )

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "completed"
        assert data["total_records"] == 2
        assert data["successful_records"] == 2

    def test_excel_import_success(self, client: TestClient, batch_importer):
        """Test successful Excel import."""
        # Create Excel data
        df = pd.DataFrame({
            "email": ["excel1@example.com", "excel2@example.com"],
            "username": ["excel1", "excel2"],
            "external_id": ["ms_001", "ms_002"],
            "department": ["IT", "Finance"]
        })

        excel_buffer = io.BytesIO()
        df.to_excel(excel_buffer, index=False)
        excel_buffer.seek(0)

        files = {"file": ("identities.xlsx", excel_buffer, "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")}

        response = client.post(
            "/api/v1/import/excel",
            files=files,
            data={"provider": "microsoft"}
        )

        assert response.status_code == 200
        data = response.json()
        assert data["total_records"] == 2

    def test_import_validation_errors(self, client: TestClient):
        """Test import with validation errors."""
        invalid_csv = """email,username,external_id
invalid-email,user1,id1
,user2,id2
user3@example.com,,id3
user4@example.com,user4,"""

        files = {"file": ("invalid.csv", io.StringIO(invalid_csv), "text/csv")}

        response = client.post(
            "/api/v1/import/csv",
            files=files,
            data={"provider": "okta"}
        )

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "completed_with_errors"
        assert data["failed_records"] > 0
        assert len(data["errors"]) > 0

    def test_import_duplicate_handling(self, client: TestClient, db_session: Session):
        """Test handling of duplicate records during import."""
        # Create existing identity
        existing = Identity(
            provider=SaaSProvider.OKTA,
            external_id="okta_existing",
            email="existing@example.com",
            username="existing"
        )
        db_session.add(existing)
        db_session.commit()

        # Import CSV with duplicate
        csv_data = """email,username,external_id
existing@example.com,existing,okta_existing
new@example.com,newuser,okta_new"""

        files = {"file": ("duplicates.csv", io.StringIO(csv_data), "text/csv")}

        response = client.post(
            "/api/v1/import/csv",
            files=files,
            data={"provider": "okta", "update_existing": "false"}
        )

        assert response.status_code == 200
        data = response.json()
        assert data["successful_records"] == 1  # Only new record
        assert data["skipped_records"] == 1  # Duplicate skipped

    def test_import_update_existing(self, client: TestClient, db_session: Session):
        """Test updating existing records during import."""
        # Create existing identity
        existing = Identity(
            provider=SaaSProvider.GOOGLE,
            external_id="google_update",
            email="update@example.com",
            username="oldusername",
            attributes={"department": "OldDept"}
        )
        db_session.add(existing)
        db_session.commit()

        # Import with updates
        json_data = {
            "identities": [{
                "external_id": "google_update",
                "email": "update@example.com",
                "username": "newusername",
                "attributes": {"department": "NewDept"}
            }]
        }

        response = client.post(
            "/api/v1/import/json",
            json={
                "provider": "google",
                "data": json_data,
                "update_existing": True
            }
        )

        assert response.status_code == 200

        # Verify update
        db_session.refresh(existing)
        assert existing.username == "newusername"
        assert existing.attributes["department"] == "NewDept"

    def test_import_progress_tracking(self, client: TestClient, batch_importer):
        """Test import progress tracking for large files."""
        # Create large dataset
        large_csv = "email,username,external_id\n"
        for i in range(1000):
            large_csv += f"user{i}@example.com,user{i},id_{i}\n"

        files = {"file": ("large.csv", io.StringIO(large_csv), "text/csv")}

        # Start import asynchronously
        response = client.post(
            "/api/v1/import/csv?async=true",
            files=files,
            data={"provider": "okta"}
        )

        assert response.status_code == 202
        data = response.json()
        job_id = data["job_id"]
        assert data["status"] == "pending"

        # Check progress
        progress_response = client.get(f"/api/v1/import/status/{job_id}")
        assert progress_response.status_code == 200
        progress_data = progress_response.json()
        assert "progress_percentage" in progress_data
        assert progress_data["status"] in ["pending", "processing", "completed"]

    def test_import_transformation_rules(self, client: TestClient):
        """Test data transformation during import."""
        csv_data = """email,full_name,dept_code
john.doe@example.com,John Doe,ENG
jane.smith@example.com,Jane Smith,SLS"""

        transformation_rules = {
            "field_mappings": {
                "full_name": "display_name",
                "dept_code": "department"
            },
            "value_mappings": {
                "department": {
                    "ENG": "Engineering",
                    "SLS": "Sales"
                }
            },
            "computed_fields": {
                "username": "lambda row: row['email'].split('@')[0]"
            }
        }

        files = {"file": ("transform.csv", io.StringIO(csv_data), "text/csv")}

        response = client.post(
            "/api/v1/import/csv",
            files=files,
            data={
                "provider": "custom",
                "transformation_rules": json.dumps(transformation_rules)
            }
        )

        assert response.status_code == 200
        data = response.json()
        assert data["successful_records"] == 2

    def test_import_error_reporting(self, client: TestClient):
        """Test detailed error reporting during import."""
        problematic_csv = """email,username,external_id,department
good@example.com,good,id1,Engineering
bad-email,bad,id2,Engineering
duplicate@example.com,dup1,id3,Sales
duplicate@example.com,dup2,id4,Marketing
missing@example.com,,id5,HR"""

        files = {"file": ("errors.csv", io.StringIO(problematic_csv), "text/csv")}

        response = client.post(
            "/api/v1/import/csv",
            files=files,
            data={"provider": "okta"}
        )

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "completed_with_errors"
        assert len(data["errors"]) > 0

        # Check error details
        errors = data["errors"]
        error_types = [e["type"] for e in errors]
        assert "validation_error" in error_types
        assert "duplicate_error" in error_types

        # Check error has row information
        for error in errors:
            assert "row_number" in error
            assert "field" in error
            assert "message" in error

    def test_import_rollback_on_failure(self, client: TestClient, db_session: Session):
        """Test transaction rollback on import failure."""
        initial_count = db_session.query(Identity).count()

        # CSV that will fail partway through
        csv_data = """email,username,external_id
valid1@example.com,valid1,id1
valid2@example.com,valid2,id2
invalid@example.com,invalid,TRIGGER_ERROR
valid3@example.com,valid3,id3"""

        files = {"file": ("rollback.csv", io.StringIO(csv_data), "text/csv")}

        response = client.post(
            "/api/v1/import/csv",
            files=files,
            data={"provider": "okta", "atomic": "true"}  # All or nothing
        )

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "failed"

        # Verify no records were imported
        final_count = db_session.query(Identity).count()
        assert final_count == initial_count

    def test_import_audit_logging(self, client: TestClient, db_session: Session):
        """Test audit logging for import operations."""
        csv_data = """email,username,external_id
audit1@example.com,audit1,id1
audit2@example.com,audit2,id2"""

        files = {"file": ("audit.csv", io.StringIO(csv_data), "text/csv")}

        response = client.post(
            "/api/v1/import/csv",
            files=files,
            data={"provider": "okta"},
            headers={"Authorization": "Bearer test_token"}
        )

        assert response.status_code == 200

        # Check audit logs
        audit_logs = db_session.query(AuditLog).filter(
            AuditLog.action == "batch_import"
        ).all()

        assert len(audit_logs) > 0
        audit_log = audit_logs[0]
        assert audit_log.resource_type == "identity_import"
        assert audit_log.success is True
        assert audit_log.action_metadata["total_records"] == 2
        assert audit_log.action_metadata["format"] == "csv"

    def test_import_format_detection(self, client: TestClient):
        """Test automatic format detection."""
        # Test with JSON content but .txt extension
        json_content = '{"identities": [{"email": "test@example.com", "username": "test"}]}'

        files = {"file": ("data.txt", json_content, "text/plain")}

        response = client.post(
            "/api/v1/import/auto",
            files=files,
            data={"provider": "okta"}
        )

        assert response.status_code == 200
        data = response.json()
        assert data["detected_format"] == "json"
        assert data["successful_records"] == 1

    def test_import_scheduling(self, client: TestClient):
        """Test scheduling imports for later execution."""
        csv_data = """email,username,external_id
scheduled@example.com,scheduled,id1"""

        files = {"file": ("scheduled.csv", io.StringIO(csv_data), "text/csv")}

        response = client.post(
            "/api/v1/import/csv",
            files=files,
            data={
                "provider": "okta",
                "schedule_at": "2024-12-31T23:59:59Z"
            }
        )

        assert response.status_code == 202
        data = response.json()
        assert data["status"] == "scheduled"
        assert "job_id" in data
        assert data["scheduled_at"] == "2024-12-31T23:59:59Z"
