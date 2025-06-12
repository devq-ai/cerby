"""
Unit tests for Task 2.1 - Create base database configuration.

Tests verify that SQLAlchemy is properly configured with async support,
database session management, and base model class with common fields.
"""

import pytest
import asyncio
from datetime import datetime
from unittest.mock import patch, MagicMock, AsyncMock
import sys
from pathlib import Path
from sqlalchemy import inspect, create_engine
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import Session
from sqlalchemy.pool import StaticPool, QueuePool

# Add project root to Python path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.db.database import (
    Base, BaseModel, DatabaseManager, db_manager,
    get_db, get_sync_db, metadata
)
from src.core.config import settings


class TestDatabaseConfiguration:
    """Test database configuration and setup."""

    def test_base_imports(self):
        """Test that all database components can be imported."""
        assert Base is not None
        assert BaseModel is not None
        assert DatabaseManager is not None
        assert db_manager is not None
        assert metadata is not None

    def test_metadata_naming_conventions(self):
        """Test that metadata has proper naming conventions."""
        conventions = metadata.naming_convention

        # Check all expected naming conventions
        assert "ix" in conventions
        assert conventions["ix"] == "ix_%(column_0_label)s"
        assert "uq" in conventions
        assert conventions["uq"] == "uq_%(table_name)s_%(column_0_name)s"
        assert "ck" in conventions
        assert conventions["ck"] == "ck_%(table_name)s_%(constraint_name)s"
        assert "fk" in conventions
        assert conventions["fk"] == "fk_%(table_name)s_%(column_0_name)s_%(referred_table_name)s"
        assert "pk" in conventions
        assert conventions["pk"] == "pk_%(table_name)s"

    def test_database_manager_initialization(self):
        """Test DatabaseManager initialization."""
        manager = DatabaseManager()

        assert manager._engine is None
        assert manager._async_engine is None
        assert manager._session_factory is None
        assert manager._async_session_factory is None
        assert manager._initialized is False

    def test_database_url_conversion(self):
        """Test database URL conversion for async support."""
        manager = DatabaseManager()

        # Test PostgreSQL conversion
        sync_url = "postgresql://user:pass@localhost/db"
        async_url = manager.get_database_url(async_mode=True)
        if sync_url in settings.database_url:
            assert "postgresql+asyncpg://" in async_url

        # Test SQLite conversion
        sync_url = "sqlite:///test.db"
        with patch.object(settings, 'database_url', sync_url):
            async_url = manager.get_database_url(async_mode=True)
            assert "sqlite+aiosqlite:///" in async_url

        # Test MySQL conversion
        sync_url = "mysql://user:pass@localhost/db"
        with patch.object(settings, 'database_url', sync_url):
            async_url = manager.get_database_url(async_mode=True)
            assert "mysql+aiomysql://" in async_url

    def test_pool_configuration(self):
        """Test connection pool configuration based on database type."""
        manager = DatabaseManager()

        # Test SQLite pool config
        with patch.object(settings, 'database_url', 'sqlite:///test.db'):
            config = manager._create_pool_config()
            assert config['poolclass'] == StaticPool
            assert 'check_same_thread' in config['connect_args']
            assert config['connect_args']['check_same_thread'] is False

        # Test PostgreSQL pool config
        with patch.object(settings, 'database_url', 'postgresql://localhost/db'):
            config = manager._create_pool_config()
            assert config['poolclass'] == QueuePool
            assert config['pool_size'] == settings.database_pool_size
            assert config['max_overflow'] == settings.database_max_overflow
            assert config['pool_pre_ping'] is True
            assert config['pool_recycle'] == 3600

    def test_sync_engine_initialization(self):
        """Test synchronous engine initialization."""
        manager = DatabaseManager()

        with patch('src.db.database.create_engine') as mock_create_engine:
            mock_engine = MagicMock()
            mock_create_engine.return_value = mock_engine

            manager.initialize_sync()

            # Check engine was created
            assert mock_create_engine.called
            assert manager._engine == mock_engine
            assert manager._session_factory is not None

    @pytest.mark.asyncio
    async def test_async_engine_initialization(self):
        """Test asynchronous engine initialization."""
        manager = DatabaseManager()

        with patch('src.db.database.create_async_engine') as mock_create_engine:
            mock_engine = AsyncMock()
            mock_create_engine.return_value = mock_engine

            await manager.initialize_async()

            # Check async engine was created
            assert mock_create_engine.called
            assert manager._async_engine == mock_engine
            assert manager._async_session_factory is not None

    def test_get_sync_session(self):
        """Test getting synchronous database session."""
        manager = DatabaseManager()

        with patch.object(manager, 'initialize_sync') as mock_init:
            with patch.object(manager, '_session_factory') as mock_factory:
                mock_session = MagicMock(spec=Session)
                mock_factory.return_value = mock_session

                session = manager.get_sync_session()

                assert session == mock_session

    @pytest.mark.asyncio
    async def test_get_async_session(self):
        """Test getting asynchronous database session."""
        manager = DatabaseManager()

        with patch.object(manager, 'initialize_async') as mock_init:
            with patch.object(manager, '_async_session_factory') as mock_factory:
                mock_session = AsyncMock(spec=AsyncSession)
                mock_factory.return_value = mock_session

                session = await manager.get_async_session()

                assert session == mock_session

    @pytest.mark.asyncio
    async def test_async_session_scope(self):
        """Test async session context manager."""
        manager = DatabaseManager()

        mock_session = AsyncMock(spec=AsyncSession)
        mock_factory = AsyncMock(return_value=mock_session)
        manager._async_session_factory = mock_factory

        # Test successful transaction
        async with manager.async_session_scope() as session:
            assert session == mock_session

        mock_session.commit.assert_called_once()
        mock_session.close.assert_called_once()

        # Test rollback on exception
        mock_session.reset_mock()

        with pytest.raises(Exception):
            async with manager.async_session_scope() as session:
                raise Exception("Test error")

        mock_session.rollback.assert_called_once()
        mock_session.close.assert_called_once()

    def test_create_all_tables(self):
        """Test creating all database tables."""
        manager = DatabaseManager()

        with patch.object(manager, 'initialize_sync'):
            with patch.object(Base.metadata, 'create_all') as mock_create:
                mock_engine = MagicMock()
                manager._engine = mock_engine

                manager.create_all_tables()

                mock_create.assert_called_once_with(bind=mock_engine)

    @pytest.mark.asyncio
    async def test_create_all_tables_async(self):
        """Test creating all database tables asynchronously."""
        manager = DatabaseManager()

        with patch.object(manager, 'initialize_async'):
            mock_engine = AsyncMock()
            mock_conn = AsyncMock()
            mock_engine.begin.return_value.__aenter__.return_value = mock_conn
            manager._async_engine = mock_engine

            await manager.create_all_tables_async()

            mock_conn.run_sync.assert_called_once()

    def test_dispose_engines(self):
        """Test disposing database engines."""
        manager = DatabaseManager()

        mock_engine = MagicMock()
        manager._engine = mock_engine

        manager.dispose()

        mock_engine.dispose.assert_called_once()

    @pytest.mark.asyncio
    async def test_dispose_engines_async(self):
        """Test disposing async database engines."""
        manager = DatabaseManager()

        mock_engine = AsyncMock()
        manager._async_engine = mock_engine

        await manager.dispose_async()

        mock_engine.dispose.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_db_dependency(self):
        """Test FastAPI dependency for getting database session."""
        with patch.object(db_manager, 'async_session_scope') as mock_scope:
            mock_session = AsyncMock(spec=AsyncSession)
            mock_scope.return_value.__aenter__.return_value = mock_session
            mock_scope.return_value.__aexit__.return_value = None

            # Test the dependency
            async_gen = get_db()
            session = await async_gen.__anext__()

            assert session == mock_session

    def test_get_sync_db_dependency(self):
        """Test synchronous database session dependency."""
        with patch.object(db_manager, 'get_sync_session') as mock_get_session:
            mock_session = MagicMock(spec=Session)
            mock_get_session.return_value = mock_session

            # Test the dependency
            gen = get_sync_db()
            session = next(gen)

            assert session == mock_session


class TestBaseModel:
    """Test BaseModel class functionality."""

    def test_base_model_abstract(self):
        """Test that BaseModel is abstract."""
        assert BaseModel.__abstract__ is True

    def test_base_model_fields(self):
        """Test that BaseModel has required fields."""
        # Create a concrete model for testing
        from sqlalchemy import Column, String

        class TestModel(BaseModel):
            __tablename__ = "test_model"
            name = Column(String(50))

        # Check fields exist
        assert hasattr(TestModel, 'id')
        assert hasattr(TestModel, 'created_at')
        assert hasattr(TestModel, 'updated_at')

    def test_base_model_tablename(self):
        """Test automatic table name generation."""
        from sqlalchemy import Column, String

        class TestUserModel(BaseModel):
            __abstract__ = False
            name = Column(String(50))

        assert TestUserModel.__tablename__ == "test_user_model"

        class ComplexModelName(BaseModel):
            __abstract__ = False
            name = Column(String(50))

        assert ComplexModelName.__tablename__ == "complex_model_name"

    def test_base_model_repr(self):
        """Test BaseModel string representation."""
        from sqlalchemy import Column, String

        class TestModel(BaseModel):
            __tablename__ = "test_model"
            name = Column(String(50))

        instance = TestModel()
        instance.id = 123

        repr_str = repr(instance)
        assert "<TestModel(id=123)>" == repr_str

    def test_base_model_to_dict(self):
        """Test BaseModel to_dict method."""
        from sqlalchemy import Column, String

        class TestModel(BaseModel):
            __tablename__ = "test_model"
            name = Column(String(50))
            value = Column(String(100))

        # Create test instance
        instance = TestModel()
        instance.id = 1
        instance.name = "test"
        instance.value = "value123"
        instance.created_at = datetime.utcnow()
        instance.updated_at = datetime.utcnow()

        # Mock the table columns
        with patch.object(instance, '__table__') as mock_table:
            mock_columns = [
                MagicMock(name='id'),
                MagicMock(name='name'),
                MagicMock(name='value'),
                MagicMock(name='created_at'),
                MagicMock(name='updated_at')
            ]
            for col in mock_columns:
                col.name = col._mock_name

            mock_table.columns = mock_columns

            result = instance.to_dict()

            assert 'id' in result
            assert 'name' in result
            assert 'value' in result
            assert 'created_at' in result
            assert 'updated_at' in result

    def test_base_model_timestamps(self):
        """Test that timestamps have proper defaults."""
        from sqlalchemy import Column, String

        class TestModel(BaseModel):
            __tablename__ = "test_model"
            name = Column(String(50))

        # Check column properties
        created_col = TestModel.created_at.property.columns[0]
        updated_col = TestModel.updated_at.property.columns[0]

        assert created_col.nullable is False
        assert updated_col.nullable is False
        assert created_col.default is not None
        assert updated_col.onupdate is not None

    def test_global_db_manager(self):
        """Test that global db_manager is available."""
        assert db_manager is not None
        assert isinstance(db_manager, DatabaseManager)

    def test_database_settings_helper(self):
        """Test database settings helper method."""
        db_settings = settings.get_database_settings()

        assert isinstance(db_settings, dict)
        assert 'url' in db_settings
        assert 'echo' in db_settings
        assert 'pool_size' in db_settings
        assert 'max_overflow' in db_settings
        assert 'pool_pre_ping' in db_settings
        assert 'pool_recycle' in db_settings

    def test_logfire_instrumentation(self):
        """Test that Logfire instrumentation is called."""
        with patch('logfire.instrument_sqlalchemy') as mock_instrument:
            manager = DatabaseManager()

            with patch('src.db.database.create_engine') as mock_create_engine:
                mock_engine = MagicMock()
                mock_create_engine.return_value = mock_engine

                manager.initialize_sync()

                mock_instrument.assert_called_once_with(engine=mock_engine)

    @pytest.mark.parametrize("db_url,expected_check", [
        ("sqlite:///test.db", "check_same_thread=False"),
        ("sqlite:///test.db?foo=bar", "check_same_thread=False"),
    ])
    def test_sqlite_url_validation(self, db_url, expected_check):
        """Test SQLite URL validation for async compatibility."""
        from src.core.config import Settings

        validated = Settings().validate_database_url(db_url)
        assert expected_check in validated
