"""
Synthetic data generator for Cerby Identity Automation Platform.

This module generates realistic synthetic identity data for all supported
SaaS providers to simulate real-world identity management scenarios.
"""

import random
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from faker import Faker
import logfire

from src.ingestion.base import BaseIngestionHandler, IngestionResult, IngestionStatus
from src.db.models.identity import IdentityProvider
from src.db.models.saas_app import SaaSApplication
from src.core.config import settings


class SyntheticDataGenerator(BaseIngestionHandler):
    """
    Generates synthetic identity data for testing and demonstration.

    Supports generating realistic data for all 10+ configured SaaS providers
    with appropriate attributes and patterns for each provider type.
    """

    def __init__(self, db_session, saas_app: SaaSApplication):
        super().__init__(db_session, saas_app)
        self.faker = Faker()
        Faker.seed(42)  # For reproducible data

        # Department distribution
        self.departments = [
            ("Engineering", 0.25),
            ("Sales", 0.20),
            ("Marketing", 0.15),
            ("Product", 0.10),
            ("Finance", 0.10),
            ("HR", 0.08),
            ("Operations", 0.07),
            ("Legal", 0.05)
        ]

        # Job titles by department
        self.job_titles = {
            "Engineering": [
                "Software Engineer", "Senior Software Engineer", "Staff Engineer",
                "Engineering Manager", "DevOps Engineer", "Data Engineer",
                "Security Engineer", "QA Engineer", "Tech Lead"
            ],
            "Sales": [
                "Sales Representative", "Account Executive", "Sales Manager",
                "Business Development Representative", "Sales Director",
                "Account Manager", "Sales Engineer"
            ],
            "Marketing": [
                "Marketing Manager", "Content Strategist", "Product Marketing Manager",
                "Digital Marketing Specialist", "Marketing Director", "SEO Specialist"
            ],
            "Product": [
                "Product Manager", "Senior Product Manager", "Product Designer",
                "UX Designer", "Product Director", "Product Analyst"
            ],
            "Finance": [
                "Financial Analyst", "Controller", "Accountant", "CFO",
                "Finance Manager", "Budget Analyst"
            ],
            "HR": [
                "HR Manager", "Recruiter", "HR Business Partner", "HR Director",
                "Talent Acquisition Specialist", "HR Coordinator"
            ],
            "Operations": [
                "Operations Manager", "Operations Analyst", "Supply Chain Manager",
                "Operations Director", "Process Improvement Specialist"
            ],
            "Legal": [
                "Legal Counsel", "Compliance Manager", "Contract Manager",
                "General Counsel", "Paralegal"
            ]
        }

        # Location distribution
        self.locations = [
            ("San Francisco, CA", 0.20),
            ("New York, NY", 0.18),
            ("Austin, TX", 0.12),
            ("Seattle, WA", 0.10),
            ("Chicago, IL", 0.08),
            ("Boston, MA", 0.08),
            ("Denver, CO", 0.06),
            ("Los Angeles, CA", 0.06),
            ("Remote", 0.12)
        ]

        # Provider-specific attribute generators
        self.provider_generators = {
            IdentityProvider.OKTA: self._generate_okta_attributes,
            IdentityProvider.AZURE_AD: self._generate_azure_ad_attributes,
            IdentityProvider.GOOGLE_WORKSPACE: self._generate_google_workspace_attributes,
            IdentityProvider.SLACK: self._generate_slack_attributes,
            IdentityProvider.GITHUB: self._generate_github_attributes,
            IdentityProvider.JIRA: self._generate_jira_attributes,
            IdentityProvider.CONFLUENCE: self._generate_confluence_attributes,
            IdentityProvider.SALESFORCE: self._generate_salesforce_attributes,
            IdentityProvider.BOX: self._generate_box_attributes,
            IdentityProvider.DROPBOX: self._generate_dropbox_attributes,
        }

    async def ingest(self, count: int = 100, **kwargs) -> IngestionResult:
        """
        Generate and ingest synthetic identity data.

        Args:
            count: Number of identities to generate
            **kwargs: Additional options (e.g., department_filter)

        Returns:
            IngestionResult with generation statistics
        """
        self.result.status = IngestionStatus.IN_PROGRESS

        with logfire.span("Generate synthetic data", provider=self.provider, count=count):
            try:
                # Generate identities
                identities = await self._generate_identities(count, **kwargs)

                # Process batch
                await self.process_batch(identities, batch_size=50)

                # Update SaaS app sync stats
                self.saas_app.record_sync_success(
                    identities_count=self.result.processed_records,
                    duration_seconds=int((datetime.utcnow() - self.result.started_at).total_seconds())
                )

                self.result.complete()
                logfire.info(
                    "Synthetic data generation completed",
                    provider=self.provider,
                    generated=self.result.processed_records,
                    failed=self.result.failed_records
                )

            except Exception as e:
                logfire.error(
                    "Synthetic data generation failed",
                    provider=self.provider,
                    error=str(e)
                )
                self.result.status = IngestionStatus.FAILED
                self.result.add_error(str(e))
                self.saas_app.record_sync_error(str(e))

            finally:
                await self.db_session.commit()

        return self.result

    async def validate_data(self, data: Dict[str, Any]) -> bool:
        """Validate synthetic data (always valid for synthetic)."""
        required_fields = ["external_id", "email"]
        return all(field in data for field in required_fields)

    async def transform_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Transform synthetic data (no transformation needed)."""
        return data

    async def _generate_identities(self, count: int, **kwargs) -> List[Dict[str, Any]]:
        """Generate list of synthetic identities."""
        identities = []
        department_filter = kwargs.get("department_filter")

        # Generate manager pool first (10% of total)
        manager_count = max(1, count // 10)
        managers = []

        for i in range(manager_count):
            manager = self._generate_single_identity(
                is_manager=True,
                department_filter=department_filter
            )
            managers.append(manager)
            identities.append(manager)

        # Generate remaining employees
        for i in range(count - manager_count):
            # Assign a manager from the same department if possible
            identity = self._generate_single_identity(
                is_manager=False,
                department_filter=department_filter
            )

            # Find manager in same department
            dept_managers = [m for m in managers if m["department"] == identity["department"]]
            if dept_managers:
                identity["manager_email"] = random.choice(dept_managers)["email"]

            identities.append(identity)

        return identities

    def _generate_single_identity(self, is_manager: bool = False,
                                department_filter: Optional[str] = None) -> Dict[str, Any]:
        """Generate a single synthetic identity."""
        # Basic profile
        first_name = self.faker.first_name()
        last_name = self.faker.last_name()
        username = f"{first_name.lower()}.{last_name.lower()}"
        email = f"{username}@example.com"

        # Department selection
        if department_filter:
            department = department_filter
        else:
            department = self._weighted_choice(self.departments)

        # Job title selection
        available_titles = self.job_titles[department]
        if is_manager:
            # Select manager titles
            manager_titles = [t for t in available_titles if "Manager" in t or "Director" in t or "Lead" in t]
            job_title = random.choice(manager_titles) if manager_titles else available_titles[0]
        else:
            job_title = random.choice(available_titles)

        # Location selection
        location = self._weighted_choice(self.locations)

        # Employment dates
        hire_date = self.faker.date_between(start_date='-5y', end_date='-1m')
        provisioned_at = hire_date + timedelta(days=random.randint(0, 7))

        # Base identity data
        identity_data = {
            "external_id": f"{self.provider}_{uuid.uuid4().hex[:12]}",
            "email": email,
            "username": username,
            "display_name": f"{first_name} {last_name}",
            "first_name": first_name,
            "last_name": last_name,
            "department": department,
            "job_title": job_title,
            "employee_id": f"EMP{random.randint(10000, 99999)}",
            "location": location,
            "provisioned_at": provisioned_at,
            "is_privileged": is_manager or "Director" in job_title or "Manager" in job_title,
            "is_service_account": False,
            "provider_attributes": {}
        }

        # Add provider-specific attributes
        provider_func = self.provider_generators.get(self.provider)
        if provider_func:
            provider_attrs = provider_func(identity_data)
            identity_data["provider_attributes"] = provider_attrs

        return identity_data

    def _weighted_choice(self, choices: List[tuple]) -> str:
        """Make a weighted random choice."""
        items, weights = zip(*choices)
        return random.choices(items, weights=weights)[0]

    def _generate_okta_attributes(self, identity: Dict[str, Any]) -> Dict[str, Any]:
        """Generate Okta-specific attributes."""
        return {
            "login": identity["username"],
            "status": random.choice(["ACTIVE", "ACTIVE", "ACTIVE", "PROVISIONED"]),
            "created": identity["provisioned_at"].isoformat(),
            "activated": (identity["provisioned_at"] + timedelta(hours=1)).isoformat(),
            "statusChanged": identity["provisioned_at"].isoformat(),
            "lastLogin": self.faker.date_time_between(start_date='-7d', end_date='now').isoformat(),
            "lastUpdated": self.faker.date_time_between(start_date='-30d', end_date='now').isoformat(),
            "passwordChanged": self.faker.date_time_between(start_date='-90d', end_date='now').isoformat(),
            "provider": {
                "type": "OKTA",
                "name": "OKTA"
            },
            "credentials": {
                "password": {},
                "provider": {
                    "type": "OKTA",
                    "name": "OKTA"
                }
            },
            "profile": {
                "firstName": identity["first_name"],
                "lastName": identity["last_name"],
                "email": identity["email"],
                "login": identity["username"],
                "mobilePhone": self.faker.phone_number() if random.random() > 0.3 else None,
                "department": identity["department"],
                "title": identity["job_title"],
                "employeeNumber": identity["employee_id"],
                "manager": identity.get("manager_email"),
                "city": identity["location"].split(",")[0] if "," in identity["location"] else identity["location"],
                "costCenter": f"CC{random.randint(100, 999)}"
            },
            "_links": {
                "self": {"href": f"https://example.okta.com/api/v1/users/{identity['external_id']}"}
            }
        }

    def _generate_azure_ad_attributes(self, identity: Dict[str, Any]) -> Dict[str, Any]:
        """Generate Azure AD-specific attributes."""
        return {
            "objectId": str(uuid.uuid4()),
            "userPrincipalName": f"{identity['username']}@example.onmicrosoft.com",
            "displayName": identity["display_name"],
            "givenName": identity["first_name"],
            "surname": identity["last_name"],
            "mail": identity["email"],
            "mailNickname": identity["username"],
            "accountEnabled": True,
            "usageLocation": "US",
            "department": identity["department"],
            "jobTitle": identity["job_title"],
            "companyName": "Example Corp",
            "manager": identity.get("manager_email"),
            "employeeId": identity["employee_id"],
            "city": identity["location"].split(",")[0] if "," in identity["location"] else identity["location"],
            "country": "United States",
            "createdDateTime": identity["provisioned_at"].isoformat(),
            "onPremisesSyncEnabled": False,
            "userType": "Member",
            "assignedLicenses": [
                {
                    "skuId": "c42b9cae-ea4f-4ab7-9717-81576235ccac",
                    "disabledPlans": []
                }
            ] if random.random() > 0.2 else [],
            "proxyAddresses": [
                f"SMTP:{identity['email']}",
                f"smtp:{identity['username']}@example.onmicrosoft.com"
            ]
        }

    def _generate_google_workspace_attributes(self, identity: Dict[str, Any]) -> Dict[str, Any]:
        """Generate Google Workspace-specific attributes."""
        return {
            "id": str(random.randint(100000000000000000, 999999999999999999)),
            "primaryEmail": identity["email"],
            "name": {
                "givenName": identity["first_name"],
                "familyName": identity["last_name"],
                "fullName": identity["display_name"]
            },
            "isAdmin": identity["is_privileged"],
            "isDelegatedAdmin": False,
            "lastLoginTime": self.faker.date_time_between(start_date='-7d', end_date='now').isoformat(),
            "creationTime": identity["provisioned_at"].isoformat(),
            "agreedToTerms": True,
            "suspended": False,
            "archived": False,
            "changePasswordAtNextLogin": False,
            "ipWhitelisted": False,
            "emails": [
                {
                    "address": identity["email"],
                    "type": "work",
                    "primary": True
                }
            ],
            "organizations": [
                {
                    "title": identity["job_title"],
                    "primary": True,
                    "customType": "",
                    "department": identity["department"],
                    "description": identity["job_title"],
                    "costCenter": f"CC{random.randint(100, 999)}"
                }
            ],
            "phones": [
                {
                    "value": self.faker.phone_number(),
                    "type": "work"
                }
            ] if random.random() > 0.5 else [],
            "locations": [
                {
                    "type": "desk",
                    "area": identity["location"]
                }
            ],
            "orgUnitPath": f"/{identity['department']}",
            "isMailboxSetup": True,
            "includeInGlobalAddressList": True
        }

    def _generate_slack_attributes(self, identity: Dict[str, Any]) -> Dict[str, Any]:
        """Generate Slack-specific attributes."""
        return {
            "id": f"U{uuid.uuid4().hex[:10].upper()}",
            "team_id": "T1234567890",
            "name": identity["username"],
            "deleted": False,
            "color": self.faker.hex_color(),
            "real_name": identity["display_name"],
            "tz": "America/Los_Angeles",
            "tz_label": "Pacific Standard Time",
            "tz_offset": -28800,
            "profile": {
                "title": identity["job_title"],
                "phone": self.faker.phone_number() if random.random() > 0.6 else "",
                "skype": "",
                "real_name": identity["display_name"],
                "real_name_normalized": identity["display_name"],
                "display_name": identity["username"],
                "display_name_normalized": identity["username"],
                "fields": {},
                "status_text": random.choice(["", "In a meeting", "Working from home", "Out of office"]),
                "status_emoji": random.choice(["", ":house_with_garden:", ":coffee:", ":calendar:"]),
                "status_expiration": 0,
                "avatar_hash": uuid.uuid4().hex[:12],
                "email": identity["email"],
                "first_name": identity["first_name"],
                "last_name": identity["last_name"],
                "image_24": f"https://avatars.slack-edge.com/24.png",
                "image_32": f"https://avatars.slack-edge.com/32.png",
                "image_48": f"https://avatars.slack-edge.com/48.png",
                "image_72": f"https://avatars.slack-edge.com/72.png",
                "image_192": f"https://avatars.slack-edge.com/192.png",
                "image_512": f"https://avatars.slack-edge.com/512.png",
                "team": "T1234567890"
            },
            "is_admin": identity["is_privileged"],
            "is_owner": False,
            "is_primary_owner": False,
            "is_restricted": False,
            "is_ultra_restricted": False,
            "is_bot": False,
            "is_app_user": False,
            "updated": int(datetime.utcnow().timestamp()),
            "has_2fa": random.random() > 0.3,
            "locale": "en-US"
        }

    def _generate_github_attributes(self, identity: Dict[str, Any]) -> Dict[str, Any]:
        """Generate GitHub-specific attributes."""
        return {
            "login": identity["username"],
            "id": random.randint(1000000, 9999999),
            "node_id": f"MDQ6VXNlcg{uuid.uuid4().hex[:10]}",
            "avatar_url": f"https://avatars.githubusercontent.com/u/{random.randint(1000000, 9999999)}",
            "gravatar_id": "",
            "url": f"https://api.github.com/users/{identity['username']}",
            "html_url": f"https://github.com/{identity['username']}",
            "type": "User",
            "site_admin": identity["is_privileged"],
            "name": identity["display_name"],
            "company": "Example Corp",
            "blog": "",
            "location": identity["location"],
            "email": identity["email"],
            "hireable": None,
            "bio": f"{identity['job_title']} at Example Corp",
            "twitter_username": None,
            "public_repos": random.randint(0, 50) if identity["department"] == "Engineering" else random.randint(0, 5),
            "public_gists": random.randint(0, 20) if identity["department"] == "Engineering" else 0,
            "followers": random.randint(0, 100),
            "following": random.randint(0, 50),
            "created_at": identity["provisioned_at"].isoformat(),
            "updated_at": self.faker.date_time_between(start_date='-30d', end_date='now').isoformat(),
            "private_gists": random.randint(0, 10),
            "total_private_repos": random.randint(0, 100) if identity["department"] == "Engineering" else random.randint(0, 10),
            "owned_private_repos": random.randint(0, 20) if identity["department"] == "Engineering" else random.randint(0, 5),
            "disk_usage": random.randint(100, 10000),
            "collaborators": random.randint(0, 20),
            "two_factor_authentication": random.random() > 0.2,
            "plan": {
                "name": "enterprise",
                "space": 976562499,
                "collaborators": 0,
                "private_repos": 9999
            }
        }

    def _generate_jira_attributes(self, identity: Dict[str, Any]) -> Dict[str, Any]:
        """Generate Jira-specific attributes."""
        return {
            "self": f"https://example.atlassian.net/rest/api/3/user?accountId={uuid.uuid4().hex[:24]}",
            "accountId": uuid.uuid4().hex[:24],
            "accountType": "atlassian",
            "emailAddress": identity["email"],
            "avatarUrls": {
                "48x48": f"https://avatar-management.services.atlassian.com/48.png",
                "24x24": f"https://avatar-management.services.atlassian.com/24.png",
                "16x16": f"https://avatar-management.services.atlassian.com/16.png",
                "32x32": f"https://avatar-management.services.atlassian.com/32.png"
            },
            "displayName": identity["display_name"],
            "active": True,
            "timeZone": "America/Los_Angeles",
            "locale": "en_US",
            "groups": {
                "size": random.randint(1, 5),
                "items": []
            },
            "applicationRoles": {
                "size": 1,
                "items": []
            },
            "expand": "groups,applicationRoles"
        }

    def _generate_confluence_attributes(self, identity: Dict[str, Any]) -> Dict[str, Any]:
        """Generate Confluence-specific attributes (similar to Jira)."""
        return {
            "type": "known",
            "username": identity["username"],
            "userKey": uuid.uuid4().hex[:24],
            "accountId": uuid.uuid4().hex[:24],
            "accountType": "atlassian",
            "email": identity["email"],
            "publicName": identity["display_name"],
            "profilePicture": {
                "path": f"/wiki/aa-avatar/{uuid.uuid4().hex[:8]}.png",
                "width": 48,
                "height": 48,
                "isDefault": False
            },
            "displayName": identity["display_name"],
            "_expandable": {
                "operations": "",
                "details": "",
                "personalSpace": ""
            },
            "_links": {
                "self": f"https://example.atlassian.net/wiki/rest/api/user?accountId={uuid.uuid4().hex[:24]}"
            }
        }

    def _generate_salesforce_attributes(self, identity: Dict[str, Any]) -> Dict[str, Any]:
        """Generate Salesforce-specific attributes."""
        return {
            "Id": f"005{uuid.uuid4().hex[:15].upper()}",
            "Username": f"{identity['username']}@example.com.salesforce",
            "LastName": identity["last_name"],
            "FirstName": identity["first_name"],
            "Name": identity["display_name"],
            "CompanyName": "Example Corp",
            "Division": identity["department"],
            "Department": identity["department"],
            "Title": identity["job_title"],
            "Email": identity["email"],
            "EmailEncodingKey": "UTF-8",
            "TimeZoneSidKey": "America/Los_Angeles",
            "LocaleSidKey": "en_US",
            "LanguageLocaleKey": "en_US",
            "UserType": "Standard",
            "Profile": {
                "Name": "Standard User" if not identity["is_privileged"] else "System Administrator"
            },
            "UserRole": {
                "Name": identity["job_title"]
            },
            "IsActive": True,
            "FederationIdentifier": identity["external_id"],
            "AboutMe": f"{identity['job_title']} in {identity['department']}",
            "EmployeeNumber": identity["employee_id"],
            "Manager": {
                "Email": identity.get("manager_email")
            } if identity.get("manager_email") else None,
            "MobilePhone": self.faker.phone_number() if random.random() > 0.5 else None,
            "LastLoginDate": self.faker.date_time_between(start_date='-7d', end_date='now').isoformat(),
            "LastPasswordChangeDate": self.faker.date_time_between(start_date='-90d', end_date='now').isoformat(),
            "CreatedDate": identity["provisioned_at"].isoformat(),
            "SystemModstamp": self.faker.date_time_between(start_date='-30d', end_date='now').isoformat()
        }

    def _generate_box_attributes(self, identity: Dict[str, Any]) -> Dict[str, Any]:
        """Generate Box-specific attributes."""
        return {
            "type": "user",
            "id": str(random.randint(10000000000, 99999999999)),
            "name": identity["display_name"],
            "login": identity["email"],
            "created_at": identity["provisioned_at"].isoformat(),
            "modified_at": self.faker.date_time_between(start_date='-30d', end_date='now').isoformat(),
            "language": "en",
            "timezone": "America/Los_Angeles",
            "space_amount": 10737418240,  # 10GB
            "space_used": random.randint(0, 5368709120),  # 0-5GB
            "max_upload_size": 5368709120,  # 5GB
            "status": "active",
            "job_title": identity["job_title"],
            "phone": self.faker.phone_number() if random.random() > 0.6 else None,
            "address": identity["location"],
            "avatar_url": f"https://app.box.com/api/avatar/large/{identity['username']}",
            "is_sync_enabled": True,
            "is_exempt_from_device_limits": False,
            "is_exempt_from_login_verification": False,
            "enterprise": {
                "type": "enterprise",
                "id": "12345678",
                "name": "Example Corp"
            },
            "role": "user" if not identity["is_privileged"] else "admin",
            "can_see_managed_users": identity["is_privileged"],
            "is_external_collab_restricted": False,
            "tracking_codes": [
                {
                    "type": "tracking_code",
                    "name": "department",
                    "value": identity["department"]
                },
                {
                    "type": "tracking_code",
                    "name": "employee_id",
                    "value": identity["employee_id"]
                }
            ]
        }

    def _generate_dropbox_attributes(self, identity: Dict[str, Any]) -> Dict[str, Any]:
        """Generate Dropbox-specific attributes."""
        return {
            "account_id": f"dbid:{uuid.uuid4().hex[:22]}",
            "email": identity["email"],
            "email_verified": True,
            "disabled": False,
            "given_name": identity["first_name"],
            "surname": identity["last_name"],
            "familiar_name": identity["first_name"],
            "display_name": identity["display_name"],
            "abbreviated_name": f"{identity['first_name'][0]}{identity['last_name'][0]}",
            "member_folder_id": str(random.randint(1000000000, 9999999999)),
            "groups": [],
            "joined_on": identity["provisioned_at"].isoformat(),
            "role": {
                ".tag": "team_admin" if identity["is_privileged"] else "user_management_admin"
                if "Manager" in identity["job_title"] else "member_only"
            },
            "profile_photo_url": f"https://dropbox.com/avatar/{identity['username']}.jpg",
            "status": {
                ".tag": "active"
            },
            "secondary_emails": [],
            "is_directory_restricted": False,
            "account_type": {
                ".tag": "full"
            }
        }
