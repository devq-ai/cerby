"""
SCIM 2.0 handler for Cerby Identity Automation Platform.

This module implements SCIM (System for Cross-domain Identity Management) 2.0
protocol for standardized identity provisioning and management.
"""

from typing import Dict, List, Any, Optional, Union
from datetime import datetime
from urllib.parse import urlparse, parse_qs
import re
import logfire

from fastapi import HTTPException, status
from sqlalchemy import select, and_, or_, func
from sqlalchemy.ext.asyncio import AsyncSession

from src.ingestion.base import BaseIngestionHandler, IngestionResult, IngestionStatus
from src.db.models.identity import Identity, IdentityProvider, IdentityStatus
from src.db.models.saas_app import SaaSApplication
from src.db.models.audit import IdentityEvent, EventType
from src.core.config import settings


class SCIMError(HTTPException):
    """SCIM-specific error with proper error response format."""

    def __init__(self, status_code: int, detail: str, scim_type: Optional[str] = None):
        error_response = {
            "schemas": ["urn:ietf:params:scim:api:messages:2.0:Error"],
            "status": str(status_code),
            "detail": detail
        }
        if scim_type:
            error_response["scimType"] = scim_type

        super().__init__(status_code=status_code, detail=error_response)


class SCIMHandler(BaseIngestionHandler):
    """
    Handles SCIM 2.0 protocol operations for identity management.

    Implements Users and Groups endpoints with full CRUD operations,
    filtering, sorting, and pagination support.
    """

    # SCIM schemas
    USER_SCHEMA = "urn:ietf:params:scim:schemas:core:2.0:User"
    GROUP_SCHEMA = "urn:ietf:params:scim:schemas:core:2.0:Group"
    ENTERPRISE_USER_SCHEMA = "urn:ietf:params:scim:schemas:extension:enterprise:2.0:User"
    LIST_RESPONSE_SCHEMA = "urn:ietf:params:scim:api:messages:2.0:ListResponse"
    PATCH_OP_SCHEMA = "urn:ietf:params:scim:api:messages:2.0:PatchOp"

    def __init__(self, db_session: AsyncSession, saas_app: SaaSApplication):
        super().__init__(db_session, saas_app)
        self.base_url = f"{settings.api_v1_prefix}/scim/v2"

    async def ingest(self, **kwargs) -> IngestionResult:
        """
        SCIM doesn't use batch ingestion - it's request/response based.
        This method is here for interface compliance.
        """
        raise NotImplementedError("SCIM uses individual request handlers, not batch ingestion")

    async def validate_data(self, data: Dict[str, Any]) -> bool:
        """Validate SCIM resource data."""
        if "schemas" not in data:
            return False

        if self.USER_SCHEMA in data["schemas"]:
            return self._validate_user_data(data)
        elif self.GROUP_SCHEMA in data["schemas"]:
            return self._validate_group_data(data)

        return False

    async def transform_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Transform SCIM data to internal identity format."""
        if self.USER_SCHEMA in data["schemas"]:
            return self._transform_user_data(data)
        elif self.GROUP_SCHEMA in data["schemas"]:
            return self._transform_group_data(data)

        raise ValueError("Unknown SCIM schema")

    # SCIM User Operations

    async def create_user(self, user_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create a new user via SCIM.

        POST /Users
        """
        with logfire.span("SCIM create user", provider=self.provider):
            # Validate input
            if not await self.validate_data(user_data):
                raise SCIMError(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Invalid user data",
                    scim_type="invalidValue"
                )

            # Transform to internal format
            identity_data = await self.transform_data(user_data)

            # Check if user already exists
            existing = await self._find_existing_identity(identity_data["external_id"])
            if existing:
                raise SCIMError(
                    status_code=status.HTTP_409_CONFLICT,
                    detail="User already exists",
                    scim_type="uniqueness"
                )

            # Create identity
            identity = await self._create_identity(identity_data)

            # Create event
            await self._create_identity_event(
                identity,
                EventType.USER_CREATED,
                {"scim_request": user_data}
            )

            await self.db_session.commit()

            # Return SCIM response
            return await self._identity_to_scim_user(identity)

    async def get_user(self, user_id: str) -> Dict[str, Any]:
        """
        Get a user by ID via SCIM.

        GET /Users/{id}
        """
        with logfire.span("SCIM get user", provider=self.provider, user_id=user_id):
            identity = await self._get_identity_by_external_id(user_id)
            if not identity:
                raise SCIMError(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail=f"User {user_id} not found"
                )

            return await self._identity_to_scim_user(identity)

    async def update_user(self, user_id: str, user_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Update a user via SCIM PUT.

        PUT /Users/{id}
        """
        with logfire.span("SCIM update user", provider=self.provider, user_id=user_id):
            # Validate input
            if not await self.validate_data(user_data):
                raise SCIMError(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Invalid user data",
                    scim_type="invalidValue"
                )

            # Get existing identity
            identity = await self._get_identity_by_external_id(user_id)
            if not identity:
                raise SCIMError(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail=f"User {user_id} not found"
                )

            # Transform and update
            identity_data = await self.transform_data(user_data)
            identity = await self._update_identity(identity, identity_data)

            # Create event
            await self._create_identity_event(
                identity,
                EventType.USER_UPDATED,
                {"scim_request": user_data}
            )

            await self.db_session.commit()

            return await self._identity_to_scim_user(identity)

    async def patch_user(self, user_id: str, patch_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Partially update a user via SCIM PATCH.

        PATCH /Users/{id}
        """
        with logfire.span("SCIM patch user", provider=self.provider, user_id=user_id):
            # Validate patch operation
            if "schemas" not in patch_data or self.PATCH_OP_SCHEMA not in patch_data["schemas"]:
                raise SCIMError(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Invalid patch request",
                    scim_type="invalidValue"
                )

            # Get existing identity
            identity = await self._get_identity_by_external_id(user_id)
            if not identity:
                raise SCIMError(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail=f"User {user_id} not found"
                )

            # Apply patch operations
            operations = patch_data.get("Operations", [])
            for operation in operations:
                await self._apply_patch_operation(identity, operation)

            # Create event
            await self._create_identity_event(
                identity,
                EventType.USER_UPDATED,
                {"scim_patch": patch_data}
            )

            await self.db_session.commit()

            return await self._identity_to_scim_user(identity)

    async def delete_user(self, user_id: str) -> None:
        """
        Delete (deprovision) a user via SCIM.

        DELETE /Users/{id}
        """
        with logfire.span("SCIM delete user", provider=self.provider, user_id=user_id):
            identity = await self._get_identity_by_external_id(user_id)
            if not identity:
                raise SCIMError(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail=f"User {user_id} not found"
                )

            # Soft delete - set status to deprovisioned
            identity.status = IdentityStatus.DEPROVISIONED.value
            identity.deprovisioned_at = datetime.utcnow()

            # Create event
            await self._create_identity_event(
                identity,
                EventType.USER_DELETED,
                {"scim_delete": True}
            )

            await self.db_session.commit()

    async def list_users(self, filter_query: Optional[str] = None,
                        start_index: int = 1, count: int = 100,
                        sort_by: Optional[str] = None,
                        sort_order: str = "ascending") -> Dict[str, Any]:
        """
        List users with filtering, sorting, and pagination.

        GET /Users
        """
        with logfire.span("SCIM list users", provider=self.provider):
            # Build query
            query = select(Identity).where(
                Identity.provider == self.provider,
                Identity.status != IdentityStatus.DEPROVISIONED.value
            )

            # Apply filters
            if filter_query:
                query = await self._apply_scim_filter(query, filter_query)

            # Get total count
            count_query = select(func.count()).select_from(query.subquery())
            total_results = await self.db_session.scalar(count_query)

            # Apply sorting
            if sort_by:
                query = self._apply_sorting(query, sort_by, sort_order)

            # Apply pagination
            query = query.offset(start_index - 1).limit(count)

            # Execute query
            result = await self.db_session.execute(query)
            identities = result.scalars().all()

            # Convert to SCIM format
            resources = []
            for identity in identities:
                resources.append(await self._identity_to_scim_user(identity))

            # Build list response
            return {
                "schemas": [self.LIST_RESPONSE_SCHEMA],
                "totalResults": total_results,
                "startIndex": start_index,
                "itemsPerPage": len(resources),
                "Resources": resources
            }

    # SCIM Group Operations (placeholder for future implementation)

    async def create_group(self, group_data: Dict[str, Any]) -> Dict[str, Any]:
        """Create a new group via SCIM."""
        raise SCIMError(
            status_code=status.HTTP_501_NOT_IMPLEMENTED,
            detail="Group operations not yet implemented"
        )

    async def get_group(self, group_id: str) -> Dict[str, Any]:
        """Get a group by ID via SCIM."""
        raise SCIMError(
            status_code=status.HTTP_501_NOT_IMPLEMENTED,
            detail="Group operations not yet implemented"
        )

    # Helper Methods

    def _validate_user_data(self, data: Dict[str, Any]) -> bool:
        """Validate SCIM user data."""
        # Required fields
        if "userName" not in data:
            return False

        # At least one name component or displayName
        name = data.get("name", {})
        if not data.get("displayName") and not (name.get("givenName") or name.get("familyName")):
            return False

        return True

    def _validate_group_data(self, data: Dict[str, Any]) -> bool:
        """Validate SCIM group data."""
        return "displayName" in data

    def _transform_user_data(self, scim_user: Dict[str, Any]) -> Dict[str, Any]:
        """Transform SCIM user to internal identity format."""
        # Extract name components
        name = scim_user.get("name", {})

        # Extract primary email
        email = None
        emails = scim_user.get("emails", [])
        for e in emails:
            if e.get("primary"):
                email = e.get("value")
                break
        if not email and emails:
            email = emails[0].get("value")

        # Extract enterprise attributes
        enterprise = {}
        if self.ENTERPRISE_USER_SCHEMA in scim_user.get("schemas", []):
            enterprise = scim_user.get(self.ENTERPRISE_USER_SCHEMA, {})

        # Build identity data
        identity_data = {
            "external_id": scim_user.get("id", scim_user.get("externalId", scim_user["userName"])),
            "email": email or f"{scim_user['userName']}@example.com",
            "username": scim_user["userName"],
            "display_name": scim_user.get("displayName", f"{name.get('givenName', '')} {name.get('familyName', '')}".strip()),
            "first_name": name.get("givenName"),
            "last_name": name.get("familyName"),
            "department": enterprise.get("department"),
            "job_title": scim_user.get("title"),
            "manager_email": enterprise.get("manager", {}).get("value"),
            "employee_id": enterprise.get("employeeNumber"),
            "location": scim_user.get("locale"),
            "is_active": scim_user.get("active", True),
            "provider_attributes": {
                "scim": scim_user
            }
        }

        return identity_data

    def _transform_group_data(self, scim_group: Dict[str, Any]) -> Dict[str, Any]:
        """Transform SCIM group to internal format."""
        # Placeholder for group transformation
        return {}

    async def _get_identity_by_external_id(self, external_id: str) -> Optional[Identity]:
        """Get identity by external ID."""
        stmt = select(Identity).where(
            Identity.provider == self.provider,
            Identity.external_id == external_id
        )
        result = await self.db_session.execute(stmt)
        return result.scalar_one_or_none()

    async def _identity_to_scim_user(self, identity: Identity) -> Dict[str, Any]:
        """Convert internal identity to SCIM user format."""
        # Build SCIM user
        scim_user = {
            "schemas": [self.USER_SCHEMA],
            "id": identity.external_id,
            "externalId": str(identity.uuid),
            "userName": identity.username or identity.email,
            "name": {
                "formatted": identity.display_name,
                "givenName": identity.first_name,
                "familyName": identity.last_name
            },
            "displayName": identity.display_name,
            "active": identity.status == IdentityStatus.ACTIVE.value,
            "emails": [
                {
                    "value": identity.email,
                    "type": "work",
                    "primary": True
                }
            ],
            "meta": {
                "resourceType": "User",
                "created": identity.created_at.isoformat(),
                "lastModified": identity.updated_at.isoformat(),
                "location": f"{self.base_url}/Users/{identity.external_id}",
                "version": f'W/"{identity.version}"'
            }
        }

        # Add enterprise extension if we have data
        if identity.department or identity.job_title or identity.manager_email or identity.employee_id:
            scim_user["schemas"].append(self.ENTERPRISE_USER_SCHEMA)
            scim_user[self.ENTERPRISE_USER_SCHEMA] = {}

            if identity.department:
                scim_user[self.ENTERPRISE_USER_SCHEMA]["department"] = identity.department
            if identity.employee_id:
                scim_user[self.ENTERPRISE_USER_SCHEMA]["employeeNumber"] = identity.employee_id
            if identity.manager_email:
                scim_user[self.ENTERPRISE_USER_SCHEMA]["manager"] = {
                    "value": identity.manager_email
                }

        if identity.job_title:
            scim_user["title"] = identity.job_title

        if identity.location:
            scim_user["locale"] = identity.location

        return scim_user

    async def _apply_scim_filter(self, query, filter_string: str):
        """
        Apply SCIM filter to query.

        Supports basic filters like:
        - userName eq "john.doe"
        - emails[type eq "work" and value co "@example.com"]
        - name.familyName sw "Smith"
        - active eq true
        """
        # Simple filter parser - handles basic attribute filters
        filter_pattern = r'(\w+(?:\.\w+)?)\s+(eq|ne|co|sw|ew|gt|lt|ge|le)\s+"?([^"]+)"?'
        matches = re.findall(filter_pattern, filter_string)

        for attribute, operator, value in matches:
            # Map SCIM attributes to model fields
            field_map = {
                "userName": Identity.username,
                "name.givenName": Identity.first_name,
                "name.familyName": Identity.last_name,
                "displayName": Identity.display_name,
                "active": Identity.status,
                "externalId": Identity.uuid,
                "emails.value": Identity.email,
                "email": Identity.email
            }

            field = field_map.get(attribute)
            if not field:
                continue

            # Apply operator
            if operator == "eq":
                if attribute == "active":
                    status = IdentityStatus.ACTIVE.value if value.lower() == "true" else IdentityStatus.SUSPENDED.value
                    query = query.where(field == status)
                else:
                    query = query.where(field == value)
            elif operator == "ne":
                query = query.where(field != value)
            elif operator == "co":  # contains
                query = query.where(field.contains(value))
            elif operator == "sw":  # starts with
                query = query.where(field.startswith(value))
            elif operator == "ew":  # ends with
                query = query.where(field.endswith(value))

        return query

    def _apply_sorting(self, query, sort_by: str, sort_order: str):
        """Apply sorting to query."""
        # Map SCIM attributes to model fields
        field_map = {
            "userName": Identity.username,
            "name.givenName": Identity.first_name,
            "name.familyName": Identity.last_name,
            "displayName": Identity.display_name,
            "created": Identity.created_at,
            "lastModified": Identity.updated_at
        }

        field = field_map.get(sort_by, Identity.created_at)

        if sort_order.lower() == "descending":
            query = query.order_by(field.desc())
        else:
            query = query.order_by(field.asc())

        return query

    async def _apply_patch_operation(self, identity: Identity, operation: Dict[str, Any]) -> None:
        """Apply a single SCIM patch operation to an identity."""
        op = operation.get("op", "").lower()
        path = operation.get("path", "")
        value = operation.get("value")

        if op == "replace":
            # Handle path-based updates
            if path == "active":
                identity.status = IdentityStatus.ACTIVE.value if value else IdentityStatus.SUSPENDED.value
            elif path == "name.givenName":
                identity.first_name = value
            elif path == "name.familyName":
                identity.last_name = value
            elif path == "displayName":
                identity.display_name = value
            elif path == "emails[type eq \"work\"].value":
                identity.email = value
            elif path == "userName":
                identity.username = value
            # Handle attribute-based updates
            elif not path and isinstance(value, dict):
                if "active" in value:
                    identity.status = IdentityStatus.ACTIVE.value if value["active"] else IdentityStatus.SUSPENDED.value
                if "name" in value:
                    name = value["name"]
                    if "givenName" in name:
                        identity.first_name = name["givenName"]
                    if "familyName" in name:
                        identity.last_name = name["familyName"]
                if "displayName" in value:
                    identity.display_name = value["displayName"]
                if "userName" in value:
                    identity.username = value["userName"]

        elif op == "add":
            # Handle add operations (mainly for multi-valued attributes)
            pass

        elif op == "remove":
            # Handle remove operations
            if path == "manager":
                identity.manager_email = None
