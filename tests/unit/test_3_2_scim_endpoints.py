"""
Unit tests for SCIM 2.0 Endpoints (Subtask 3.2).

Tests cover:
- SCIM user endpoints
- SCIM group endpoints
- SCIM schema compliance
- Filtering and pagination
- Error handling
"""

import pytest
from datetime import datetime
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session
import json

from main import app
from src.ingestion.scim import SCIMRouter, SCIMResponse, SCIMError
from src.db.models.identity import Identity
from src.db.models.saas_application import SaaSApplication, SaaSProvider


class TestSCIMUserEndpoints:
    """Test suite for SCIM user endpoints."""

    def test_scim_create_user(self, client: TestClient, db_session: Session):
        """Test creating a user via SCIM endpoint."""
        scim_user = {
            "schemas": ["urn:ietf:params:scim:schemas:core:2.0:User"],
            "externalId": "external123",
            "userName": "testuser",
            "name": {
                "givenName": "Test",
                "familyName": "User"
            },
            "emails": [
                {
                    "value": "testuser@example.com",
                    "primary": True
                }
            ],
            "active": True
        }

        response = client.post("/scim/v2/Users", json=scim_user)

        assert response.status_code == 201
        data = response.json()
        assert data["id"] is not None
        assert data["userName"] == "testuser"
        assert data["meta"]["resourceType"] == "User"
        assert data["meta"]["created"] is not None

    def test_scim_get_user(self, client: TestClient, db_session: Session):
        """Test retrieving a user via SCIM endpoint."""
        # Create an identity first
        identity = Identity(
            provider=SaaSProvider.OKTA,
            external_id="okta123",
            email="getuser@example.com",
            username="getuser"
        )
        db_session.add(identity)
        db_session.commit()

        response = client.get(f"/scim/v2/Users/{identity.id}")

        assert response.status_code == 200
        data = response.json()
        assert data["id"] == str(identity.id)
        assert data["userName"] == "getuser"
        assert data["emails"][0]["value"] == "getuser@example.com"

    def test_scim_update_user(self, client: TestClient, db_session: Session):
        """Test updating a user via SCIM endpoint."""
        # Create an identity
        identity = Identity(
            provider=SaaSProvider.OKTA,
            external_id="update123",
            email="updateuser@example.com",
            username="updateuser"
        )
        db_session.add(identity)
        db_session.commit()

        update_data = {
            "schemas": ["urn:ietf:params:scim:schemas:core:2.0:User"],
            "id": str(identity.id),
            "userName": "updateuser",
            "emails": [
                {
                    "value": "newemail@example.com",
                    "primary": True
                }
            ],
            "active": False
        }

        response = client.put(f"/scim/v2/Users/{identity.id}", json=update_data)

        assert response.status_code == 200
        data = response.json()
        assert data["emails"][0]["value"] == "newemail@example.com"
        assert data["active"] is False

    def test_scim_patch_user(self, client: TestClient, db_session: Session):
        """Test patching a user via SCIM endpoint."""
        # Create an identity
        identity = Identity(
            provider=SaaSProvider.GOOGLE,
            external_id="patch123",
            email="patchuser@example.com",
            username="patchuser",
            is_active=True
        )
        db_session.add(identity)
        db_session.commit()

        patch_data = {
            "schemas": ["urn:ietf:params:scim:api:messages:2.0:PatchOp"],
            "Operations": [
                {
                    "op": "replace",
                    "path": "active",
                    "value": False
                },
                {
                    "op": "add",
                    "path": "phoneNumbers",
                    "value": [
                        {
                            "value": "+1-555-1234",
                            "type": "work"
                        }
                    ]
                }
            ]
        }

        response = client.patch(f"/scim/v2/Users/{identity.id}", json=patch_data)

        assert response.status_code == 200
        data = response.json()
        assert data["active"] is False
        assert data["phoneNumbers"][0]["value"] == "+1-555-1234"

    def test_scim_delete_user(self, client: TestClient, db_session: Session):
        """Test deleting a user via SCIM endpoint."""
        # Create an identity
        identity = Identity(
            provider=SaaSProvider.MICROSOFT,
            external_id="delete123",
            email="deleteuser@example.com",
            username="deleteuser"
        )
        db_session.add(identity)
        db_session.commit()

        response = client.delete(f"/scim/v2/Users/{identity.id}")

        assert response.status_code == 204

        # Verify soft delete
        db_session.refresh(identity)
        assert identity.is_active is False

    def test_scim_list_users(self, client: TestClient, db_session: Session):
        """Test listing users via SCIM endpoint."""
        # Create multiple identities
        for i in range(5):
            identity = Identity(
                provider=SaaSProvider.OKTA,
                external_id=f"list{i}",
                email=f"listuser{i}@example.com",
                username=f"listuser{i}"
            )
            db_session.add(identity)
        db_session.commit()

        response = client.get("/scim/v2/Users")

        assert response.status_code == 200
        data = response.json()
        assert data["schemas"] == ["urn:ietf:params:scim:api:messages:2.0:ListResponse"]
        assert data["totalResults"] >= 5
        assert len(data["Resources"]) >= 5

    def test_scim_filter_users(self, client: TestClient, db_session: Session):
        """Test filtering users via SCIM endpoint."""
        # Create identities with different attributes
        identity1 = Identity(
            provider=SaaSProvider.OKTA,
            external_id="filter1",
            email="engineering@example.com",
            username="enguser",
            attributes={"department": "Engineering"}
        )
        identity2 = Identity(
            provider=SaaSProvider.OKTA,
            external_id="filter2",
            email="sales@example.com",
            username="salesuser",
            attributes={"department": "Sales"}
        )
        db_session.add_all([identity1, identity2])
        db_session.commit()

        # Test email filter
        response = client.get('/scim/v2/Users?filter=email eq "engineering@example.com"')

        assert response.status_code == 200
        data = response.json()
        assert data["totalResults"] == 1
        assert data["Resources"][0]["emails"][0]["value"] == "engineering@example.com"

    def test_scim_pagination(self, client: TestClient, db_session: Session):
        """Test pagination in SCIM list endpoint."""
        # Create many identities
        for i in range(20):
            identity = Identity(
                provider=SaaSProvider.GOOGLE,
                external_id=f"page{i}",
                email=f"pageuser{i}@example.com",
                username=f"pageuser{i}"
            )
            db_session.add(identity)
        db_session.commit()

        # Test pagination
        response = client.get("/scim/v2/Users?startIndex=1&count=10")

        assert response.status_code == 200
        data = response.json()
        assert data["itemsPerPage"] == 10
        assert data["startIndex"] == 1
        assert len(data["Resources"]) <= 10

    def test_scim_error_handling(self, client: TestClient):
        """Test SCIM error responses."""
        # Test 404 - User not found
        response = client.get("/scim/v2/Users/nonexistent")

        assert response.status_code == 404
        data = response.json()
        assert data["schemas"] == ["urn:ietf:params:scim:api:messages:2.0:Error"]
        assert data["status"] == "404"
        assert "Not Found" in data["detail"]

        # Test 400 - Invalid filter
        response = client.get("/scim/v2/Users?filter=invalid filter syntax")

        assert response.status_code == 400
        data = response.json()
        assert data["status"] == "400"

    def test_scim_schema_validation(self, client: TestClient):
        """Test SCIM schema validation."""
        # Invalid user data - missing required fields
        invalid_user = {
            "schemas": ["urn:ietf:params:scim:schemas:core:2.0:User"],
            # Missing userName
            "emails": [
                {
                    "value": "test@example.com",
                    "primary": True
                }
            ]
        }

        response = client.post("/scim/v2/Users", json=invalid_user)

        assert response.status_code == 400
        data = response.json()
        assert data["scimType"] == "invalidValue"

    def test_scim_attributes_parameter(self, client: TestClient, db_session: Session):
        """Test attributes parameter for sparse field sets."""
        # Create an identity
        identity = Identity(
            provider=SaaSProvider.SLACK,
            external_id="attrs123",
            email="attrs@example.com",
            username="attrsuser",
            display_name="Attrs User",
            attributes={"department": "IT", "title": "Developer"}
        )
        db_session.add(identity)
        db_session.commit()

        # Request only specific attributes
        response = client.get(f"/scim/v2/Users/{identity.id}?attributes=userName,emails")

        assert response.status_code == 200
        data = response.json()
        assert "userName" in data
        assert "emails" in data
        assert "displayName" not in data  # Should be excluded

    def test_scim_bulk_operations(self, client: TestClient):
        """Test SCIM bulk operations endpoint."""
        bulk_request = {
            "schemas": ["urn:ietf:params:scim:api:messages:2.0:BulkRequest"],
            "Operations": [
                {
                    "method": "POST",
                    "path": "/Users",
                    "bulkId": "bulk1",
                    "data": {
                        "schemas": ["urn:ietf:params:scim:schemas:core:2.0:User"],
                        "userName": "bulkuser1",
                        "emails": [{"value": "bulk1@example.com", "primary": True}]
                    }
                },
                {
                    "method": "POST",
                    "path": "/Users",
                    "bulkId": "bulk2",
                    "data": {
                        "schemas": ["urn:ietf:params:scim:schemas:core:2.0:User"],
                        "userName": "bulkuser2",
                        "emails": [{"value": "bulk2@example.com", "primary": True}]
                    }
                }
            ]
        }

        response = client.post("/scim/v2/Bulk", json=bulk_request)

        assert response.status_code == 200
        data = response.json()
        assert data["schemas"] == ["urn:ietf:params:scim:api:messages:2.0:BulkResponse"]
        assert len(data["Operations"]) == 2
        assert all(op["status"] == "201" for op in data["Operations"])


class TestSCIMGroupEndpoints:
    """Test suite for SCIM group endpoints."""

    def test_scim_create_group(self, client: TestClient):
        """Test creating a group via SCIM endpoint."""
        scim_group = {
            "schemas": ["urn:ietf:params:scim:schemas:core:2.0:Group"],
            "displayName": "Engineering Team",
            "members": [
                {
                    "value": "user1",
                    "$ref": "https://example.com/scim/v2/Users/user1",
                    "type": "User"
                }
            ]
        }

        response = client.post("/scim/v2/Groups", json=scim_group)

        assert response.status_code == 201
        data = response.json()
        assert data["displayName"] == "Engineering Team"
        assert data["meta"]["resourceType"] == "Group"

    def test_scim_service_provider_config(self, client: TestClient):
        """Test SCIM ServiceProviderConfig endpoint."""
        response = client.get("/scim/v2/ServiceProviderConfig")

        assert response.status_code == 200
        data = response.json()
        assert data["schemas"] == ["urn:ietf:params:scim:schemas:core:2.0:ServiceProviderConfig"]
        assert "patch" in data
        assert "bulk" in data
        assert "filter" in data
        assert "changePassword" in data
        assert "sort" in data
        assert "etag" in data
        assert "authenticationSchemes" in data

    def test_scim_schemas_endpoint(self, client: TestClient):
        """Test SCIM Schemas endpoint."""
        response = client.get("/scim/v2/Schemas")

        assert response.status_code == 200
        data = response.json()
        assert data["totalResults"] >= 2  # At least User and Group schemas

        # Check for User schema
        user_schema = next(
            (s for s in data["Resources"] if s["id"] == "urn:ietf:params:scim:schemas:core:2.0:User"),
            None
        )
        assert user_schema is not None
        assert "attributes" in user_schema

    def test_scim_resource_types(self, client: TestClient):
        """Test SCIM ResourceTypes endpoint."""
        response = client.get("/scim/v2/ResourceTypes")

        assert response.status_code == 200
        data = response.json()
        assert data["totalResults"] >= 2  # At least User and Group

        # Check for User resource type
        user_type = next(
            (r for r in data["Resources"] if r["name"] == "User"),
            None
        )
        assert user_type is not None
        assert user_type["endpoint"] == "/Users"
        assert user_type["schema"] == "urn:ietf:params:scim:schemas:core:2.0:User"
