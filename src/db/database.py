"""
Database configuration and session management for Cerby Identity Automation Platform.

This module sets up SQLAlchemy with async support, manages database sessions,
and provides base classes for all database models.
"""

from typing import AsyncGenerator, Optional
from contextlib import asynccontextmanager

from sqlalchemy import create_engine, MetaData, event, Engine
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine, async_sessionmaker
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.pool import StaticPool, QueuePool
import logfire

from src.core.config import settings


# Database metadata with naming conventions
metadata = MetaData(
    naming_convention={
        "ix": "ix_%(column_0_label)s",
        "uq": "uq_%(table_name)s_%(column_0_name)s",
        "ck": "ck_%(table_name)s_%(constraint_name)s",
        "fk": "fk_%(table_name)s_%(column_0_name)s_%(referred_table_name)s",
        "pk": "pk_%(table_name)s"
    }
)

# Base class for all models
Base = declarative_base(metadata=metadata)


class DatabaseManager:
    """
    Manages database connections and sessions with support for both sync and async operations.
    """

    def __init__(self):
        self._engine: Optional[Engine] = None
        self._async_engine: Optional[AsyncEngine] = None
        self._session_factory: Optional[sessionmaker] = None
        self._async_session_factory: Optional[async_sessionmaker] = None
        self._initialized = False

    def get_database_url(self, async_mode: bool = False) -> str:
        """
        Get the appropriate database URL for sync or async operations.

        Args:
            async_mode: Whether to return async-compatible URL

        Returns:
            Database URL string
        """
        url = settings.database_url

        if async_mode:
            # Convert sync URLs to async equivalents
            if url.startswith("postgresql://"):
                url = url.replace("postgresql://", "postgresql+asyncpg://")
            elif url.startswith("mysql://"):
                url = url.replace("mysql://", "mysql+aiomysql://")
            elif url.startswith("sqlite:///"):
                url = url.replace("sqlite:///", "sqlite+aiosqlite:///")

        return url

    def _create_pool_config(self):
        """Create connection pool configuration based on database type."""
        if settings.database_url.startswith("sqlite"):
            return {
                "poolclass": StaticPool,
                "connect_args": {"check_same_thread": False}
            }
        else:
            return {
                "poolclass": QueuePool,
                "pool_size": settings.database_pool_size,
                "max_overflow": settings.database_max_overflow,
                "pool_pre_ping": True,
                "pool_recycle": 3600  # Recycle connections after 1 hour
            }

    def initialize_sync(self):
        """Initialize synchronous database engine and session factory."""
        if self._engine is not None:
            return

        pool_config = self._create_pool_config()

        self._engine = create_engine(
            self.get_database_url(async_mode=False),
            echo=settings.database_echo,
            **pool_config
        )

        # Add event listeners for logging
        @event.listens_for(self._engine, "connect")
        def receive_connect(dbapi_connection, connection_record):
            logfire.info("Database connection established",
                        connection_id=id(dbapi_connection))

        @event.listens_for(self._engine, "close")
        def receive_close(dbapi_connection, connection_record):
            logfire.info("Database connection closed",
                        connection_id=id(dbapi_connection))

        self._session_factory = sessionmaker(
            bind=self._engine,
            autocommit=False,
            autoflush=False,
            expire_on_commit=False
        )

        # Instrument SQLAlchemy with Logfire
        logfire.instrument_sqlalchemy(engine=self._engine)

    async def initialize_async(self):
        """Initialize asynchronous database engine and session factory."""
        if self._async_engine is not None:
            return

        pool_config = self._create_pool_config()

        self._async_engine = create_async_engine(
            self.get_database_url(async_mode=True),
            echo=settings.database_echo,
            **pool_config
        )

        self._async_session_factory = async_sessionmaker(
            bind=self._async_engine,
            class_=AsyncSession,
            autocommit=False,
            autoflush=False,
            expire_on_commit=False
        )

    def get_sync_session(self) -> Session:
        """
        Get a synchronous database session.

        Returns:
            SQLAlchemy Session instance
        """
        if self._session_factory is None:
            self.initialize_sync()
        return self._session_factory()

    async def get_async_session(self) -> AsyncSession:
        """
        Get an asynchronous database session.

        Returns:
            SQLAlchemy AsyncSession instance
        """
        if self._async_session_factory is None:
            await self.initialize_async()
        return self._async_session_factory()

    @asynccontextmanager
    async def async_session_scope(self) -> AsyncGenerator[AsyncSession, None]:
        """
        Provide a transactional scope for async database operations.

        Yields:
            AsyncSession instance
        """
        async with self._async_session_factory() as session:
            try:
                yield session
                await session.commit()
            except Exception:
                await session.rollback()
                raise
            finally:
                await session.close()

    def create_all_tables(self):
        """Create all database tables (sync)."""
        if self._engine is None:
            self.initialize_sync()
        Base.metadata.create_all(bind=self._engine)
        logfire.info("Database tables created")

    async def create_all_tables_async(self):
        """Create all database tables (async)."""
        if self._async_engine is None:
            await self.initialize_async()
        async with self._async_engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
        logfire.info("Database tables created (async)")

    def drop_all_tables(self):
        """Drop all database tables (sync). Use with caution!"""
        if self._engine is None:
            self.initialize_sync()
        Base.metadata.drop_all(bind=self._engine)
        logfire.warning("All database tables dropped!")

    async def drop_all_tables_async(self):
        """Drop all database tables (async). Use with caution!"""
        if self._async_engine is None:
            await self.initialize_async()
        async with self._async_engine.begin() as conn:
            await conn.run_sync(Base.metadata.drop_all)
        logfire.warning("All database tables dropped! (async)")

    def dispose(self):
        """Dispose of all database connections."""
        if self._engine is not None:
            self._engine.dispose()
            logfire.info("Sync database engine disposed")

        if self._async_engine is not None:
            # Note: async engine disposal should be done in async context
            logfire.info("Async database engine marked for disposal")

    async def dispose_async(self):
        """Dispose of async database connections."""
        if self._async_engine is not None:
            await self._async_engine.dispose()
            logfire.info("Async database engine disposed")


# Global database manager instance
db_manager = DatabaseManager()


# Dependency functions for FastAPI
async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """
    FastAPI dependency to get database session.

    Yields:
        AsyncSession instance
    """
    async with db_manager.async_session_scope() as session:
        yield session


def get_sync_db() -> Session:
    """
    Get synchronous database session (for migrations, scripts, etc).

    Returns:
        Session instance
    """
    session = db_manager.get_sync_session()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


# Base model class with common fields
from datetime import datetime
from sqlalchemy import Column, DateTime, Integer
from sqlalchemy.ext.declarative import declared_attr


class BaseModel(Base):
    """
    Abstract base model with common fields for all database models.
    """
    __abstract__ = True

    id = Column(Integer, primary_key=True, index=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)

    @declared_attr
    def __tablename__(cls):
        """Generate table name from class name."""
        # Convert CamelCase to snake_case
        name = cls.__name__
        return ''.join(['_' + c.lower() if c.isupper() else c for c in name]).lstrip('_')

    def __repr__(self):
        """Default string representation."""
        return f"<{self.__class__.__name__}(id={self.id})>"

    def to_dict(self):
        """Convert model to dictionary."""
        return {
            column.name: getattr(self, column.name)
            for column in self.__table__.columns
        }


# Export key components
__all__ = [
    "Base",
    "BaseModel",
    "db_manager",
    "get_db",
    "get_sync_db",
    "DatabaseManager"
]
