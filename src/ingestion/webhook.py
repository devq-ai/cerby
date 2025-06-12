"""
Webhook handler for Cerby Identity Automation Platform.

This module processes real-time identity events from various SaaS providers
via webhooks, supporting different authentication methods and event formats.
"""

import hashlib
import hmac
import json
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from enum import Enum
import asyncio
from collections import defaultdict
import logfire

from fastapi import Request, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession

from src.ingestion.base import BaseIngestionHandler, IngestionResult, IngestionStatus, IngestionError
from src.db.models.identity import Identity, IdentityProvider, IdentityStatus
from src.db.models.saas_app import SaaSApplication
from src.db.models.audit import IdentityEvent, EventType
from src.core.config import settings


class WebhookAuthType(str, Enum):
    """Types of webhook authentication."""
    HMAC_SHA256 = "hmac_sha256"
    HMAC_SHA1 = "hmac_sha1"
    BEARER_TOKEN = "bearer_token"
    BASIC_AUTH = "basic_auth"
    CUSTOM_HEADER = "custom_header"
    IP_WHITELIST = "ip_whitelist"
    NONE = "none"


class WebhookHandler(BaseIngestionHandler):
    """
    Handles incoming webhooks from various SaaS providers.

    Supports multiple authentication methods, event deduplication,
    and provider-specific event parsing.
    """

    def __init__(self, db_session: AsyncSession, saas_app: SaaSApplication):
        super().__init__(db_session, saas_app)

        # Event type mapping by provider
        self.event_mappings = {
            IdentityProvider.OKTA: self._map_okta_events,
            IdentityProvider.AZURE_AD: self._map_azure_ad_events,
            IdentityProvider.GOOGLE_WORKSPACE: self._map_google_events,
            IdentityProvider.SLACK: self._map_slack_events,
            IdentityProvider.GITHUB: self._map_github_events,
            IdentityProvider.JIRA: self._map_atlassian_events,
            IdentityProvider.CONFLUENCE: self._map_atlassian_events,
            IdentityProvider.SALESFORCE: self._map_salesforce_events,
            IdentityProvider.BOX: self._map_box_events,
            IdentityProvider.DROPBOX: self._map_dropbox_events,
        }

        # Event parsers by provider
        self.event_parsers = {
            IdentityProvider.OKTA: self._parse_okta_event,
            IdentityProvider.AZURE_AD: self._parse_azure_ad_event,
            IdentityProvider.GOOGLE_WORKSPACE: self._parse_google_event,
            IdentityProvider.SLACK: self._parse_slack_event,
            IdentityProvider.GITHUB: self._parse_github_event,
            IdentityProvider.JIRA: self._parse_atlassian_event,
            IdentityProvider.CONFLUENCE: self._parse_atlassian_event,
            IdentityProvider.SALESFORCE: self._parse_salesforce_event,
            IdentityProvider.BOX: self._parse_box_event,
            IdentityProvider.DROPBOX: self._parse_dropbox_event,
        }

        # Event deduplication cache (in production, use Redis)
        self._event_cache = defaultdict(set)
        self._cache_ttl = timedelta(minutes=5)
        self._last_cache_cleanup = datetime.utcnow()

    async def ingest(self, **kwargs) -> IngestionResult:
        """
        Process webhook events.

        This is typically called from the webhook endpoint handler.
        """
        request: Request = kwargs.get("request")
        if not request:
            raise IngestionError("Request object required for webhook processing")

        self.result.status = IngestionStatus.IN_PROGRESS

        with logfire.span("Process webhook", provider=self.provider):
            try:
                # Verify webhook authenticity
                if not await self._verify_webhook(request):
                    raise IngestionError(
                        "Webhook verification failed",
                        provider=self.provider,
                        error_code="INVALID_SIGNATURE"
                    )

                # Parse webhook payload
                payload = await self._parse_payload(request)

                # Extract events from payload
                events = await self._extract_events(payload)
                self.result.total_records = len(events)

                # Process each event
                for event in events:
                    await self._process_webhook_event(event)

                # Update SaaS app stats
                self.saas_app.record_sync_success(
                    identities_count=self.result.processed_records,
                    duration_seconds=int((datetime.utcnow() - self.result.started_at).total_seconds())
                )

                self.result.complete()
                await self.db_session.commit()

                logfire.info(
                    "Webhook processing completed",
                    provider=self.provider,
                    events_processed=self.result.processed_records,
                    events_failed=self.result.failed_records
                )

            except Exception as e:
                logfire.error(
                    "Webhook processing failed",
                    provider=self.provider,
                    error=str(e)
                )
                self.result.status = IngestionStatus.FAILED
                self.result.add_error(str(e))
                self.saas_app.record_sync_error(str(e))
                await self.db_session.rollback()
                raise

        return self.result

    async def validate_data(self, data: Dict[str, Any]) -> bool:
        """Validate webhook event data."""
        # Basic validation - provider-specific validation in parsers
        return "event_type" in data and "identity_data" in data

    async def transform_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Transform webhook event data to internal format."""
        # Already transformed by provider-specific parsers
        return data["identity_data"]

    async def _verify_webhook(self, request: Request) -> bool:
        """Verify webhook authenticity based on provider configuration."""
        auth_type = self.saas_app.provider_config.get("webhook_auth_type", WebhookAuthType.NONE)

        if auth_type == WebhookAuthType.NONE:
            return True

        elif auth_type == WebhookAuthType.HMAC_SHA256:
            return await self._verify_hmac_sha256(request)

        elif auth_type == WebhookAuthType.HMAC_SHA1:
            return await self._verify_hmac_sha1(request)

        elif auth_type == WebhookAuthType.BEARER_TOKEN:
            return await self._verify_bearer_token(request)

        elif auth_type == WebhookAuthType.CUSTOM_HEADER:
            return await self._verify_custom_header(request)

        elif auth_type == WebhookAuthType.IP_WHITELIST:
            return await self._verify_ip_whitelist(request)

        return False

    async def _verify_hmac_sha256(self, request: Request) -> bool:
        """Verify HMAC-SHA256 signature."""
        signature_header = self.saas_app.provider_config.get("signature_header", "X-Webhook-Signature")
        expected_signature = request.headers.get(signature_header)

        if not expected_signature:
            return False

        # Get webhook secret
        secret = self.saas_app.webhook_secret
        if not secret:
            return False

        # Calculate signature
        body = await request.body()
        calculated_signature = hmac.new(
            secret.encode(),
            body,
            hashlib.sha256
        ).hexdigest()

        # Compare signatures
        return hmac.compare_digest(calculated_signature, expected_signature)

    async def _verify_hmac_sha1(self, request: Request) -> bool:
        """Verify HMAC-SHA1 signature (used by GitHub)."""
        signature_header = request.headers.get("X-Hub-Signature")
        if not signature_header or not signature_header.startswith("sha1="):
            return False

        expected_signature = signature_header[5:]  # Remove "sha1=" prefix

        secret = self.saas_app.webhook_secret
        if not secret:
            return False

        body = await request.body()
        calculated_signature = hmac.new(
            secret.encode(),
            body,
            hashlib.sha1
        ).hexdigest()

        return hmac.compare_digest(calculated_signature, expected_signature)

    async def _verify_bearer_token(self, request: Request) -> bool:
        """Verify Bearer token in Authorization header."""
        auth_header = request.headers.get("Authorization", "")
        if not auth_header.startswith("Bearer "):
            return False

        token = auth_header[7:]  # Remove "Bearer " prefix
        expected_token = self.saas_app.provider_config.get("webhook_token")

        return token == expected_token

    async def _verify_custom_header(self, request: Request) -> bool:
        """Verify custom header value."""
        header_name = self.saas_app.provider_config.get("custom_header_name")
        header_value = self.saas_app.provider_config.get("custom_header_value")

        if not header_name or not header_value:
            return False

        return request.headers.get(header_name) == header_value

    async def _verify_ip_whitelist(self, request: Request) -> bool:
        """Verify request comes from whitelisted IP."""
        client_ip = request.client.host if request.client else None
        if not client_ip:
            return False

        whitelist = self.saas_app.provider_config.get("ip_whitelist", [])
        return client_ip in whitelist

    async def _parse_payload(self, request: Request) -> Dict[str, Any]:
        """Parse webhook payload from request."""
        content_type = request.headers.get("content-type", "").lower()

        if "application/json" in content_type:
            return await request.json()

        elif "application/x-www-form-urlencoded" in content_type:
            form_data = await request.form()
            return dict(form_data)

        else:
            # Try to parse as JSON anyway
            body = await request.body()
            return json.loads(body.decode())

    async def _extract_events(self, payload: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract individual events from webhook payload."""
        # Provider-specific event extraction
        if self.provider == IdentityProvider.OKTA:
            # Okta sends events in "events" array
            return payload.get("data", {}).get("events", [])

        elif self.provider == IdentityProvider.GITHUB:
            # GitHub sends single event
            return [payload]

        elif self.provider == IdentityProvider.SLACK:
            # Slack sends event in "event" object
            event = payload.get("event")
            return [event] if event else []

        elif self.provider in [IdentityProvider.JIRA, IdentityProvider.CONFLUENCE]:
            # Atlassian sends single event
            return [payload]

        # Default: assume single event
        return [payload]

    async def _process_webhook_event(self, event: Dict[str, Any]) -> None:
        """Process a single webhook event."""
        try:
            # Check for duplicate
            event_id = self._get_event_id(event)
            if await self._is_duplicate_event(event_id):
                self.result.skipped_records += 1
                return

            # Map event type
            event_type = await self._map_event_type(event)
            if not event_type:
                self.result.skipped_records += 1
                return

            # Parse event data
            parser = self.event_parsers.get(self.provider)
            if not parser:
                raise IngestionError(f"No parser for provider {self.provider}")

            parsed_data = await parser(event, event_type)

            if not parsed_data:
                self.result.skipped_records += 1
                return

            # Validate parsed data
            if not await self.validate_data(parsed_data):
                raise IngestionError("Invalid event data")

            # Process based on event type
            if event_type in [EventType.USER_CREATED, EventType.USER_UPDATED]:
                identity_data = await self.transform_data(parsed_data)
                await self.process_identity(identity_data)

            elif event_type == EventType.USER_DELETED:
                await self._process_deletion(parsed_data["identity_data"])

            elif event_type == EventType.USER_SUSPENDED:
                await self._process_suspension(parsed_data["identity_data"], True)

            elif event_type == EventType.USER_ACTIVATED:
                await self._process_suspension(parsed_data["identity_data"], False)

            else:
                # Just record the event
                await self._record_event_only(parsed_data, event_type)

        except Exception as e:
            self.result.add_error(
                IngestionError(
                    str(e),
                    provider=self.provider,
                    details={"event": event}
                )
            )

    def _get_event_id(self, event: Dict[str, Any]) -> str:
        """Extract or generate unique event ID."""
        # Try common event ID fields
        event_id = (
            event.get("id") or
            event.get("eventId") or
            event.get("event_id") or
            event.get("uuid") or
            event.get("messageId")
        )

        if not event_id:
            # Generate ID from event content
            event_str = json.dumps(event, sort_keys=True)
            event_id = hashlib.sha256(event_str.encode()).hexdigest()

        return f"{self.provider}:{event_id}"

    async def _is_duplicate_event(self, event_id: str) -> bool:
        """Check if event is duplicate using cache."""
        # Clean up old cache entries
        await self._cleanup_event_cache()

        # Check cache
        if event_id in self._event_cache[self.provider]:
            return True

        # Add to cache
        self._event_cache[self.provider].add(event_id)

        # Also check database for recent events
        from sqlalchemy import select
        stmt = select(IdentityEvent).where(
            IdentityEvent.provider == self.provider,
            IdentityEvent.provider_event_id == event_id,
            IdentityEvent.created_at > datetime.utcnow() - timedelta(hours=24)
        )
        result = await self.db_session.execute(stmt)
        return result.scalar_one_or_none() is not None

    async def _cleanup_event_cache(self) -> None:
        """Clean up old entries from event cache."""
        if datetime.utcnow() - self._last_cache_cleanup > self._cache_ttl:
            self._event_cache.clear()
            self._last_cache_cleanup = datetime.utcnow()

    async def _map_event_type(self, event: Dict[str, Any]) -> Optional[EventType]:
        """Map provider event to internal event type."""
        mapper = self.event_mappings.get(self.provider)
        if not mapper:
            return None

        return await mapper(event)

    async def _process_deletion(self, identity_data: Dict[str, Any]) -> None:
        """Process identity deletion event."""
        identity = await self._find_existing_identity(identity_data["external_id"])
        if identity:
            identity.status = IdentityStatus.DEPROVISIONED.value
            identity.deprovisioned_at = datetime.utcnow()

            await self._create_identity_event(
                identity,
                EventType.USER_DELETED,
                identity_data
            )

            self.result.processed_records += 1

    async def _process_suspension(self, identity_data: Dict[str, Any], suspended: bool) -> None:
        """Process identity suspension/activation event."""
        identity = await self._find_existing_identity(identity_data["external_id"])
        if identity:
            identity.status = IdentityStatus.SUSPENDED.value if suspended else IdentityStatus.ACTIVE.value

            event_type = EventType.USER_SUSPENDED if suspended else EventType.USER_ACTIVATED
            await self._create_identity_event(
                identity,
                event_type,
                identity_data
            )

            self.result.processed_records += 1

    async def _record_event_only(self, parsed_data: Dict[str, Any], event_type: EventType) -> None:
        """Record event without processing identity changes."""
        identity = await self._find_existing_identity(parsed_data["identity_data"]["external_id"])
        if identity:
            await self._create_identity_event(
                identity,
                event_type,
                parsed_data["identity_data"]
            )
            self.result.processed_records += 1

    # Provider-specific event mapping methods

    async def _map_okta_events(self, event: Dict[str, Any]) -> Optional[EventType]:
        """Map Okta event types."""
        event_type = event.get("eventType", "")

        mapping = {
            "user.lifecycle.create": EventType.USER_CREATED,
            "user.lifecycle.update": EventType.USER_UPDATED,
            "user.lifecycle.deactivate": EventType.USER_DELETED,
            "user.lifecycle.suspend": EventType.USER_SUSPENDED,
            "user.lifecycle.unsuspend": EventType.USER_ACTIVATED,
            "user.session.start": EventType.LOGIN_SUCCESS,
            "user.authentication.auth_via_mfa": EventType.LOGIN_SUCCESS,
            "user.mfa.factor.activate": EventType.MFA_ENABLED,
            "user.mfa.factor.deactivate": EventType.MFA_DISABLED,
        }

        return mapping.get(event_type)

    async def _map_azure_ad_events(self, event: Dict[str, Any]) -> Optional[EventType]:
        """Map Azure AD event types."""
        # Azure AD webhook events vary by configuration
        operation = event.get("Operation", "")

        mapping = {
            "Add user": EventType.USER_CREATED,
            "Update user": EventType.USER_UPDATED,
            "Delete user": EventType.USER_DELETED,
            "Disable account": EventType.USER_SUSPENDED,
            "Enable account": EventType.USER_ACTIVATED,
        }

        return mapping.get(operation)

    async def _map_google_events(self, event: Dict[str, Any]) -> Optional[EventType]:
        """Map Google Workspace event types."""
        event_name = event.get("name", "")

        mapping = {
            "admin.user.create": EventType.USER_CREATED,
            "admin.user.update": EventType.USER_UPDATED,
            "admin.user.delete": EventType.USER_DELETED,
            "admin.user.suspend": EventType.USER_SUSPENDED,
            "admin.user.unsuspend": EventType.USER_ACTIVATED,
            "login.success": EventType.LOGIN_SUCCESS,
            "login.failure": EventType.LOGIN_FAILURE,
        }

        return mapping.get(event_name)

    async def _map_slack_events(self, event: Dict[str, Any]) -> Optional[EventType]:
        """Map Slack event types."""
        event_type = event.get("type", "")

        mapping = {
            "team_join": EventType.USER_CREATED,
            "user_change": EventType.USER_UPDATED,
            "user_profile_changed": EventType.USER_UPDATED,
        }

        return mapping.get(event_type)

    async def _map_github_events(self, event: Dict[str, Any]) -> Optional[EventType]:
        """Map GitHub event types."""
        # GitHub uses X-GitHub-Event header, but we'll check action field
        action = event.get("action", "")

        if "member" in event:
            if action == "added":
                return EventType.USER_CREATED
            elif action == "removed":
                return EventType.USER_DELETED

        return None

    async def _map_atlassian_events(self, event: Dict[str, Any]) -> Optional[EventType]:
        """Map Atlassian (Jira/Confluence) event types."""
        webhook_event = event.get("webhookEvent", "")

        mapping = {
            "user_created": EventType.USER_CREATED,
            "user_updated": EventType.USER_UPDATED,
            "user_deleted": EventType.USER_DELETED,
        }

        return mapping.get(webhook_event)

    async def _map_salesforce_events(self, event: Dict[str, Any]) -> Optional[EventType]:
        """Map Salesforce event types."""
        # Salesforce Platform Events
        event_type = event.get("Type__c", "")

        mapping = {
            "UserCreated": EventType.USER_CREATED,
            "UserUpdated": EventType.USER_UPDATED,
            "UserDeactivated": EventType.USER_DELETED,
        }

        return mapping.get(event_type)

    async def _map_box_events(self, event: Dict[str, Any]) -> Optional[EventType]:
        """Map Box event types."""
        event_type = event.get("event_type", "")

        mapping = {
            "USER_CREATED": EventType.USER_CREATED,
            "USER_MODIFIED": EventType.USER_UPDATED,
            "USER_DELETED": EventType.USER_DELETED,
        }

        return mapping.get(event_type)

    async def _map_dropbox_events(self, event: Dict[str, Any]) -> Optional[EventType]:
        """Map Dropbox event types."""
        # Dropbox Business API events
        event_type = event.get(".tag", "")

        mapping = {
            "member_add": EventType.USER_CREATED,
            "member_change_status": EventType.USER_UPDATED,
            "member_remove": EventType.USER_DELETED,
        }

        return mapping.get(event_type)

    # Provider-specific event parsing methods

    async def _parse_okta_event(self, event: Dict[str, Any], event_type: EventType) -> Optional[Dict[str, Any]]:
        """Parse Okta event data."""
        target = event.get("target", [])
        if not target:
            return None

        # Get user from target
        user_target = next((t for t in target if t.get("type") == "User"), None)
        if not user_target:
            return None

        # Extract user data
        user_data = {
            "external_id": user_target.get("id"),
            "email": user_target.get("alternateId"),
            "display_name": user_target.get("displayName"),
            "provider_attributes": event
        }

        return {
            "event_type": event_type,
            "identity_data": user_data
        }

    async def _parse_azure_ad_event(self, event: Dict[str, Any], event_type: EventType) -> Optional[Dict[str, Any]]:
        """Parse Azure AD event data."""
        user_data = {
            "external_id": event.get("ObjectId"),
            "email": event.get("UserPrincipalName"),
            "display_name": event.get("DisplayName"),
            "first_name": event.get("GivenName"),
            "last_name": event.get("Surname"),
            "department": event.get("Department"),
            "job_title": event.get("JobTitle"),
            "provider_attributes": event
        }

        return {
            "event_type": event_type,
            "identity_data": user_data
        }

    async def _parse_google_event(self, event: Dict[str, Any], event_type: EventType) -> Optional[Dict[str, Any]]:
        """Parse Google Workspace event data."""
        parameters = event.get("parameters", {})

        user_data = {
            "external_id": parameters.get("USER_ID"),
            "email": parameters.get("USER_EMAIL"),
            "provider_attributes": event
        }

        return {
            "event_type": event_type,
            "identity_data": user_data
        }

    async def _parse_slack_event(self, event: Dict[str, Any], event_type: EventType) -> Optional[Dict[str, Any]]:
        """Parse Slack event data."""
        user = event.get("user", {})
        profile = user.get("profile", {})

        user_data = {
            "external_id": user.get("id"),
            "email": profile.get("email"),
            "username": user.get("name"),
            "display_name": profile.get("real_name"),
            "first_name": profile.get("first_name"),
            "last_name": profile.get("last_name"),
            "job_title": profile.get("title"),
            "provider_attributes": event
        }

        return {
            "event_type": event_type,
            "identity_data": user_data
        }

    async def _parse_github_event(self, event: Dict[str, Any], event_type: EventType) -> Optional[Dict[str, Any]]:
        """Parse GitHub event data."""
        member = event.get("member", {})

        user_data = {
            "external_id": str(member.get("id")),
            "username": member.get("login"),
            "email": member.get("email", f"{member.get('login')}@users.noreply.github.com"),
            "display_name": member.get("name") or member.get("login"),
            "provider_attributes": event
        }

        return {
            "event_type": event_type,
            "identity_data": user_data
        }

    async def _parse_atlassian_event(self, event: Dict[str, Any], event_type: EventType) -> Optional[Dict[str, Any]]:
        """Parse Atlassian (Jira/Confluence) event data."""
        user = event.get("user", {})

        user_data = {
            "external_id": user.get("accountId"),
            "email": user.get("emailAddress"),
            "display_name": user.get("displayName"),
            "provider_attributes": event
        }

        return {
            "event_type": event_type,
            "identity_data": user_data
        }

    async def _parse_salesforce_event(self, event: Dict[str, Any], event_type: EventType) -> Optional[Dict[str, Any]]:
        """Parse Salesforce event data."""
        user_data = {
            "external_id": event.get("UserId__c"),
            "email": event.get("Email__c"),
            "username": event.get("Username__c"),
            "display_name": event.get("Name__c"),
            "first_name": event.get("FirstName__c"),
            "last_name": event.get("LastName__c"),
            "department": event.get("Department__c"),
            "job_title": event.get("Title__c"),
            "provider_attributes": event
        }

        return {
            "event_type": event_type,
            "identity_data": user_data
        }

    async def _parse_box_event(self, event: Dict[str, Any], event_type: EventType) -> Optional[Dict[str, Any]]:
        """Parse Box event data."""
        source = event.get("source", {})

        user_data = {
            "external_id": source.get("id"),
            "email": source.get("login"),
            "display_name": source.get("name"),
            "job_title": source.get("job_title"),
            "provider_attributes": event
        }

        return {
            "event_type": event_type,
            "identity_data": user_data
        }

    async def _parse_dropbox_event(self, event: Dict[str, Any], event_type: EventType) -> Optional[Dict[str, Any]]:
        """Parse Dropbox event data."""
        user_data = {
            "external_id": event.get("account_id"),
            "email": event.get("email"),
            "display_name": event.get("display_name"),
            "first_name": event.get("given_name"),
            "last_name": event.get("surname"),
            "provider_attributes": event
        }

        return {
            "event_type": event_type,
            "identity_data": user_data
        }
