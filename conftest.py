"""
PyTest configuration and fixtures for Cerby Identity Automation Platform.

This module provides shared test fixtures, database setup/teardown,
and test client configuration for comprehensive testing.
"""

import os
import sys
import asyncio
from typing import Generator, AsyncGenerator
from datetime import datetime, timedelta
import pytest
import pytest_asyncio
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.pool import StaticPool
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine, async_sessionmaker
from fastapi.testclient import TestClient
from httpx import AsyncClient
import logfire

# Add project root to Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from main import app
from src.db.database import Base, BaseModel, get_db, db_manager
from src.core.config import settings


# Override settings for testing
settings.environment = "testing"
settings.database_url = "sqlite:///./test.db"
settings.redis_url = "redis://localhost:6379/15"  # Use different Redis DB for testing
settings.logfire_environment = "testing"


# Configure pytest-asyncio
@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


# Test database setup
TEST_DATABASE_URL = "sqlite+aiosqlite:///./test.db"
TEST_SYNC_DATABASE_URL = "sqlite:///./test.db"


@pytest.fixture(scope="session")
def test_engine():
    """Create test database engine for the session."""
    engine = create_engine(
        TEST_SYNC_DATABASE_URL,
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )

    # Create all tables
    Base.metadata.create_all(bind=engine)

    yield engine

    # Drop all tables after tests
    Base.metadata.drop_all(bind=engine)
    engine.dispose()


@pytest.fixture(scope="session")
async def async_test_engine():
    """Create async test database engine for the session."""
    engine = create_async_engine(
        TEST_DATABASE_URL,
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )

    # Create all tables
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    yield engine

    # Drop all tables after tests
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)

    await engine.dispose()


@pytest.fixture
def db_session(test_engine) -> Generator[Session, None, None]:
    """Create a test database session."""
    TestingSessionLocal = sessionmaker(
        autocommit=False,
        autoflush=False,
        bind=test_engine
    )

    session = TestingSessionLocal()

    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


@pytest_asyncio.fixture
async def async_db_session(async_test_engine) -> AsyncGenerator[AsyncSession, None]:
    """Create an async test database session."""
    async_session_maker = async_sessionmaker(
        async_test_engine,
        class_=AsyncSession,
        expire_on_commit=False
    )

    async with async_session_maker() as session:
        yield session
        await session.rollback()


@pytest.fixture
def client(db_session) -> Generator[TestClient, None, None]:
    """Create a test client with database override."""

    def override_get_db():
        try:
            yield db_session
        finally:
            pass

    app.dependency_overrides[get_db] = override_get_db

    with TestClient(app) as test_client:
        yield test_client

    app.dependency_overrides.clear()


@pytest_asyncio.fixture
async def async_client(async_db_session) -> AsyncGenerator[AsyncClient, None]:
    """Create an async test client with database override."""

    async def override_get_db():
        yield async_db_session

    app.dependency_overrides[get_db] = override_get_db

    async with AsyncClient(app=app, base_url="http://test") as test_client:
        yield test_client

    app.dependency_overrides.clear()


# Test data fixtures
@pytest.fixture
def sample_user_data():
    """Sample user data for testing."""
    return {
        "email": "test@example.com",
        "full_name": "Test User",
        "is_active": True,
        "is_superuser": False
    }


@pytest.fixture
def sample_identity_data():
    """Sample identity data for testing."""
    return {
        "provider": "okta",
        "external_id": "00u1234567890",
        "email": "test@example.com",
        "username": "test.user",
        "display_name": "Test User",
        "attributes": {
            "department": "Engineering",
            "title": "Software Engineer",
            "manager": "manager@example.com"
        }
    }


@pytest.fixture
def sample_policy_data():
    """Sample access policy data for testing."""
    return {
        "name": "Engineering Access Policy",
        "description": "Default access policy for engineering team",
        "rules": [
            {
                "resource": "github",
                "action": "read",
                "condition": "user.department == 'Engineering'"
            },
            {
                "resource": "jira",
                "action": "write",
                "condition": "user.department == 'Engineering'"
            }
        ],
        "priority": 100,
        "is_active": True
    }


@pytest.fixture
def sample_saas_app_data():
    """Sample SaaS application data for testing."""
    return {
        "name": "GitHub Enterprise",
        "provider": "github",
        "api_endpoint": "https://api.github.com",
        "auth_type": "oauth2",
        "config": {
            "client_id": "test_client_id",
            "client_secret": "test_client_secret",
            "scopes": ["read:user", "repo"]
        }
    }


@pytest.fixture
def sample_identity_event_data():
    """Sample identity event data for testing."""
    return {
        "event_type": "user.created",
        "provider": "okta",
        "external_id": "00u1234567890",
        "timestamp": datetime.utcnow().isoformat(),
        "data": {
            "email": "new.user@example.com",
            "username": "new.user",
            "department": "Sales"
        }
    }


@pytest.fixture
def auth_headers():
    """Sample authentication headers for testing."""
    # This would normally use a real JWT token
    return {"Authorization": "Bearer test_token_123"}


# Genetic Algorithm test fixtures
@pytest.fixture
def ga_test_config():
    """Genetic algorithm test configuration."""
    return {
        "population_size": 10,
        "generations": 5,
        "mutation_rate": 0.1,
        "crossover_rate": 0.8,
        "elite_size": 2
    }


# Mock external services
@pytest.fixture
def mock_redis(mocker):
    """Mock Redis client for testing."""
    redis_mock = mocker.MagicMock()
    redis_mock.get.return_value = None
    redis_mock.set.return_value = True
    redis_mock.delete.return_value = True
    redis_mock.exists.return_value = False
    return redis_mock


@pytest.fixture
def mock_logfire(mocker):
    """Mock Logfire for testing."""
    logfire_mock = mocker.patch("logfire")
    return logfire_mock


# Utility functions for testing
def create_test_user(db_session: Session, **kwargs):
    """Create a test user in the database."""
    from src.db.models.user import User  # Import will be added later

    user_data = {
        "email": "test@example.com",
        "full_name": "Test User",
        "hashed_password": "hashed_password_here",
        **kwargs
    }

    user = User(**user_data)
    db_session.add(user)
    db_session.commit()
    db_session.refresh(user)
    return user


async def create_test_user_async(db_session: AsyncSession, **kwargs):
    """Create a test user in the async database."""
    from src.db.models.user import User  # Import will be added later

    user_data = {
        "email": "test@example.com",
        "full_name": "Test User",
        "hashed_password": "hashed_password_here",
        **kwargs
    }

    user = User(**user_data)
    db_session.add(user)
    await db_session.commit()
    await db_session.refresh(user)
    return user


# Performance testing utilities
@pytest.fixture
def performance_timer():
    """Simple performance timer for tests."""
    import time

    class Timer:
        def __init__(self):
            self.start_time = None
            self.end_time = None

        def start(self):
            self.start_time = time.time()

        def stop(self):
            self.end_time = time.time()

        def elapsed(self):
            if self.start_time and self.end_time:
                return self.end_time - self.start_time
            return None

    return Timer()


# Cleanup fixtures
@pytest.fixture(autouse=True)
def cleanup_test_files():
    """Clean up test files after each test."""
    yield

    # Clean up any test files created
    test_files = ["test.db", "test.db-shm", "test.db-wal"]
    for file in test_files:
        if os.path.exists(file):
            try:
                os.remove(file)
            except Exception:
                pass


# Test markers
pytest.mark.slow = pytest.mark.slow
pytest.mark.integration = pytest.mark.integration
pytest.mark.unit = pytest.mark.unit
pytest.mark.requires_redis = pytest.mark.requires_redis
pytest.mark.requires_db = pytest.mark.requires_db
