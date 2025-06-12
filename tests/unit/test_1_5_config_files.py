"""
Unit tests for Task 1.5 - Configure development environment files.

Tests verify that all necessary configuration files are created and properly
configured including .env, requirements.txt, .gitignore, README.md, and logging.
"""

import pytest
import sys
from pathlib import Path
import json
import os
from unittest.mock import patch, mock_open
import logging

# Add project root to Python path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.core.config import settings


class TestConfigurationFiles:
    """Test configuration files setup and content."""

    def test_env_example_exists(self):
        """Test that .env.example exists with all required variables."""
        env_example_path = project_root / ".env.example"
        assert env_example_path.exists(), ".env.example file not found"
        assert env_example_path.is_file(), ".env.example is not a file"

    def test_env_example_content(self):
        """Test that .env.example contains all required environment variables."""
        env_example_path = project_root / ".env.example"
        with open(env_example_path, 'r') as f:
            content = f.read()

        # Required environment variables
        required_vars = [
            # FastAPI Configuration
            "DEBUG=",
            "ENVIRONMENT=",
            "SECRET_KEY=",

            # Logfire Configuration
            "LOGFIRE_TOKEN=",
            "LOGFIRE_PROJECT_NAME=",
            "LOGFIRE_SERVICE_NAME=",
            "LOGFIRE_ENVIRONMENT=",

            # TaskMaster AI Configuration
            "ANTHROPIC_API_KEY=",
            "MODEL=",
            "MAX_TOKENS=",
            "TEMPERATURE=",
            "DEFAULT_SUBTASKS=",
            "DEFAULT_PRIORITY=",

            # Database Configuration
            "DATABASE_URL=",

            # Redis Configuration
            "REDIS_URL=",

            # Darwin Genetic Algorithm Configuration
            "DARWIN_POPULATION_SIZE=",
            "DARWIN_GENERATIONS=",
            "DARWIN_MUTATION_RATE=",
            "DARWIN_CROSSOVER_RATE=",
            "DARWIN_ELITE_SIZE=",

            # Identity Provider Simulation
            "SIMULATE_PROVIDERS=",
            "SIMULATION_INTERVAL=",
            "SIMULATION_BATCH_SIZE=",

            # Compliance Configuration
            "ENABLE_SOX_COMPLIANCE=",
            "ENABLE_GDPR_COMPLIANCE=",
            "COMPLIANCE_AUDIT_RETENTION_DAYS=",

            # Panel Dashboard Configuration
            "PANEL_PORT=",
            "PANEL_WEBSOCKET_ORIGIN=",

            # Feature Flags
            "ENABLE_GENETIC_OPTIMIZATION=",
            "ENABLE_REALTIME_PROCESSING=",
            "ENABLE_ADAPTIVE_LEARNING=",
        ]

        missing_vars = []
        for var in required_vars:
            if var not in content:
                missing_vars.append(var.split('=')[0])

        assert not missing_vars, f"Missing environment variables in .env.example: {missing_vars}"

    def test_env_example_has_descriptions(self):
        """Test that .env.example has comments describing variables."""
        env_example_path = project_root / ".env.example"
        with open(env_example_path, 'r') as f:
            content = f.read()

        # Should have section headers
        assert "# FastAPI Configuration" in content
        assert "# Database Configuration" in content
        assert "# Logfire Configuration" in content

    def test_requirements_txt_exists(self):
        """Test that requirements.txt exists."""
        req_path = project_root / "requirements.txt"
        assert req_path.exists(), "requirements.txt not found"
        assert req_path.is_file(), "requirements.txt is not a file"

    def test_requirements_txt_content(self):
        """Test that requirements.txt contains all necessary packages."""
        req_path = project_root / "requirements.txt"
        with open(req_path, 'r') as f:
            content = f.read().lower()

        # Core packages that must be present
        required_packages = [
            # FastAPI Foundation
            "fastapi",
            "uvicorn",
            "pydantic",
            "pydantic-settings",

            # Logfire Observability
            "logfire",

            # Database
            "sqlalchemy",
            "alembic",

            # Testing Framework
            "pytest",
            "pytest-asyncio",
            "pytest-cov",
            "httpx",

            # Data Processing
            "pandas",
            "numpy",

            # Development Tools
            "black",
            "isort",
            "flake8",
            "mypy",

            # Authentication & Security
            "python-jose",
            "passlib",
            "python-dotenv",

            # Utilities
            "redis",
        ]

        missing_packages = []
        for package in required_packages:
            if package not in content:
                missing_packages.append(package)

        assert not missing_packages, f"Missing packages in requirements.txt: {missing_packages}"

    def test_requirements_have_versions(self):
        """Test that requirements have version specifications."""
        req_path = project_root / "requirements.txt"
        with open(req_path, 'r') as f:
            lines = f.readlines()

        # Count packages with version specs
        versioned = 0
        total = 0
        for line in lines:
            line = line.strip()
            if line and not line.startswith('#'):
                total += 1
                if '>=' in line or '==' in line or '~=' in line:
                    versioned += 1

        # At least 80% should have version specs
        if total > 0:
            version_ratio = versioned / total
            assert version_ratio >= 0.8, f"Only {version_ratio*100:.1f}% of packages have version specs"

    def test_gitignore_exists(self):
        """Test that .gitignore exists."""
        gitignore_path = project_root / ".gitignore"
        assert gitignore_path.exists(), ".gitignore not found"
        assert gitignore_path.is_file(), ".gitignore is not a file"

    def test_gitignore_content(self):
        """Test that .gitignore contains necessary patterns."""
        gitignore_path = project_root / ".gitignore"
        with open(gitignore_path, 'r') as f:
            content = f.read()

        # Essential patterns
        required_patterns = [
            # Python
            "__pycache__/",
            "*.py[cod]",
            "*.so",
            ".Python",

            # Virtual Environment
            "venv/",
            "env/",
            ".venv/",

            # IDE
            ".vscode/",
            ".idea/",
            "*.swp",

            # Environment Variables
            ".env",
            ".env.local",

            # Database
            "*.db",
            "*.sqlite",

            # Logs
            "*.log",
            "logs/",

            # Testing
            ".coverage",
            ".pytest_cache/",
            "htmlcov/",

            # OS files
            ".DS_Store",
            "Thumbs.db",

            # Security
            "*.pem",
            "*.key",

            # Logfire
            ".logfire/",
        ]

        missing_patterns = []
        for pattern in required_patterns:
            if pattern not in content:
                missing_patterns.append(pattern)

        assert not missing_patterns, f"Missing patterns in .gitignore: {missing_patterns}"

    def test_readme_exists(self):
        """Test that README.md exists."""
        readme_path = project_root / "README.md"
        assert readme_path.exists(), "README.md not found"
        assert readme_path.is_file(), "README.md is not a file"

    def test_readme_structure(self):
        """Test that README.md has proper structure and content."""
        readme_path = project_root / "README.md"
        with open(readme_path, 'r') as f:
            content = f.read()

        # Required sections (case-insensitive check)
        required_sections = [
            "# Cerby Identity Automation Platform",
            "## Overview",
            "## Technology Stack",
            "## Prerequisites",
            "## Installation",
            "## Running the Application",
            "## Testing",
            "## Project Structure",
            "## API",
            "## Contributing",
        ]

        missing_sections = []
        for section in required_sections:
            if section not in content:
                # Try case-insensitive search
                if section.lower() not in content.lower():
                    missing_sections.append(section)

        assert not missing_sections, f"Missing sections in README.md: {missing_sections}"

    def test_readme_has_badges(self):
        """Test that README.md includes status badges."""
        readme_path = project_root / "README.md"
        with open(readme_path, 'r') as f:
            content = f.read()

        # Should have some indication of test coverage or build status
        # Could be badges or text indicators
        indicators = ["coverage", "test", "build", "status"]
        has_indicator = any(indicator in content.lower() for indicator in indicators)
        assert has_indicator, "README.md should include project status indicators"

    def test_config_module_loads(self):
        """Test that the config module loads without errors."""
        try:
            from src.core.config import settings
            assert settings is not None
            assert hasattr(settings, 'debug')
            assert hasattr(settings, 'database_url')
            assert hasattr(settings, 'logfire_token')
        except ImportError as e:
            pytest.fail(f"Failed to import config module: {e}")

    def test_settings_validation(self):
        """Test that settings have proper validation."""
        from src.core.config import Settings

        # Test creating settings with defaults
        test_settings = Settings()

        # Check defaults are set
        assert test_settings.app_name == "Cerby Identity Automation Platform"
        assert test_settings.environment in ["development", "testing", "production"]
        assert isinstance(test_settings.debug, bool)
        assert isinstance(test_settings.port, int)

    def test_settings_from_env(self, monkeypatch):
        """Test that settings can be loaded from environment variables."""
        # Set test environment variables
        monkeypatch.setenv("DEBUG", "true")
        monkeypatch.setenv("DATABASE_URL", "postgresql://test:test@localhost/test")
        monkeypatch.setenv("REDIS_URL", "redis://localhost:6379/1")

        # Reload settings
        from src.core.config import Settings
        test_settings = Settings()

        assert test_settings.debug is True
        assert test_settings.database_url == "postgresql://test:test@localhost/test"
        assert test_settings.redis_url == "redis://localhost:6379/1"

    def test_settings_type_conversion(self, monkeypatch):
        """Test that settings properly convert types from strings."""
        monkeypatch.setenv("PORT", "8080")
        monkeypatch.setenv("DARWIN_POPULATION_SIZE", "200")
        monkeypatch.setenv("DARWIN_MUTATION_RATE", "0.15")
        monkeypatch.setenv("ENABLE_GENETIC_OPTIMIZATION", "true")

        from src.core.config import Settings
        test_settings = Settings()

        assert test_settings.port == 8080
        assert isinstance(test_settings.port, int)
        assert test_settings.darwin_population_size == 200
        assert test_settings.darwin_mutation_rate == 0.15
        assert test_settings.enable_genetic_optimization is True

    def test_logging_configuration(self):
        """Test that logging is properly configured."""
        # Check if logging is configured in main.py or config
        main_path = project_root / "main.py"
        config_path = project_root / "src" / "core" / "config.py"

        logging_configured = False

        for file_path in [main_path, config_path]:
            if file_path.exists():
                with open(file_path, 'r') as f:
                    content = f.read()
                    if "logging" in content or "logfire" in content:
                        logging_configured = True
                        break

        assert logging_configured, "Logging configuration not found"

    def test_settings_validators(self):
        """Test that settings have proper validators."""
        config_path = project_root / "src" / "core" / "config.py"
        with open(config_path, 'r') as f:
            content = f.read()

        # Check for validators
        assert "@validator" in content or "field_validator" in content
        assert "parse_cors_origins" in content  # CORS origins parser
        assert "validate_database_url" in content  # Database URL validator

    def test_sensitive_data_not_in_repo(self):
        """Test that sensitive data is not committed to repository."""
        # Check that .env file is not in repo (should be in .gitignore)
        env_path = project_root / ".env"
        gitignore_path = project_root / ".gitignore"

        if env_path.exists():
            # If .env exists, make sure it's in .gitignore
            with open(gitignore_path, 'r') as f:
                gitignore_content = f.read()
                assert ".env" in gitignore_content, ".env file exists but not in .gitignore!"

    def test_config_helper_methods(self):
        """Test that config has helper methods."""
        from src.core.config import settings

        # Test helper methods
        assert hasattr(settings, 'get_redis_url_with_password')
        assert hasattr(settings, 'get_database_settings')
        assert hasattr(settings, 'get_logfire_settings')
        assert hasattr(settings, 'is_production')
        assert hasattr(settings, 'is_development')

        # Test they work
        assert isinstance(settings.is_production(), bool)
        assert isinstance(settings.is_development(), bool)
        assert isinstance(settings.get_database_settings(), dict)

    def test_pre_commit_config(self):
        """Test that pre-commit configuration exists (optional but recommended)."""
        pre_commit_path = project_root / ".pre-commit-config.yaml"
        if pre_commit_path.exists():
            with open(pre_commit_path, 'r') as f:
                content = f.read()
                # Should have basic hooks
                assert "black" in content or "ruff" in content
                assert "isort" in content or "imports" in content

    @pytest.mark.parametrize("config_file", [
        ".env.example",
        "requirements.txt",
        ".gitignore",
        "README.md",
    ])
    def test_config_files_not_empty(self, config_file):
        """Test that configuration files are not empty."""
        file_path = project_root / config_file
        assert file_path.exists(), f"{config_file} not found"
        assert file_path.stat().st_size > 0, f"{config_file} is empty"
