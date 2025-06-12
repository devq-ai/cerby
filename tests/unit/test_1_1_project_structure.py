"""
Unit tests for Task 1.1 - Initialize Python project structure.

Tests verify that all required directories and files are created correctly
and that the project structure meets DevQ.ai standards.
"""

import os
import sys
from pathlib import Path
import pytest
import tempfile
import shutil

# Add project root to Python path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))


class TestProjectStructure:
    """Test project directory structure and initialization."""

    @pytest.fixture
    def project_paths(self):
        """Get all expected project paths."""
        return {
            'directories': [
                'src',
                'src/api',
                'src/api/v1',
                'src/api/v1/endpoints',
                'src/core',
                'src/db',
                'src/db/models',
                'src/schemas',
                'src/services',
                'src/genetic_algorithm',
                'src/ingestion',
                'src/ingestion/generators',
                'src/ingestion/providers',
                'src/ingestion/transformers',
                'src/analytics',
                'tests',
                'tests/unit',
                'tests/integration',
                'docs',
                '.taskmaster',
            ],
            'files': [
                'main.py',
                'requirements.txt',
                '.gitignore',
                'README.md',
                'conftest.py',
                '.env.example',
            ],
            'optional_dirs': [
                '.git',
                '.logfire',
                '.zed',
                'mcp',
                'venv',
            ]
        }

    def test_required_directories_exist(self, project_paths):
        """Test that all required directories exist."""
        missing_dirs = []

        for directory in project_paths['directories']:
            dir_path = project_root / directory
            if not dir_path.exists() or not dir_path.is_dir():
                missing_dirs.append(directory)

        assert not missing_dirs, f"Missing directories: {missing_dirs}"

    def test_required_files_exist(self, project_paths):
        """Test that all required files exist."""
        missing_files = []

        for file in project_paths['files']:
            file_path = project_root / file
            if not file_path.exists() or not file_path.is_file():
                missing_files.append(file)

        assert not missing_files, f"Missing files: {missing_files}"

    def test_python_package_structure(self):
        """Test that Python packages have __init__.py files."""
        packages = [
            'src',
            'src/core',
            'src/db',
            'src/db/models',
            'src/ingestion',
            'src/ingestion/transformers',
        ]

        missing_init = []
        for package in packages:
            init_path = project_root / package / '__init__.py'
            if not init_path.exists():
                missing_init.append(f"{package}/__init__.py")

        assert not missing_init, f"Missing __init__.py files: {missing_init}"

    def test_git_configuration(self):
        """Test Git repository configuration."""
        git_dir = project_root / '.git'
        assert git_dir.exists(), "Git repository not initialized"

        # Check git config
        git_config = git_dir / 'config'
        if git_config.exists():
            with open(git_config, 'r') as f:
                config_content = f.read()
                assert 'user' in config_content, "Git user configuration missing"

    def test_taskmaster_configuration(self):
        """Test TaskMaster AI configuration."""
        taskmaster_dir = project_root / '.taskmaster'
        assert taskmaster_dir.exists(), "TaskMaster directory missing"

        # Check required TaskMaster files
        tasks_json = taskmaster_dir / 'tasks.json'
        config_json = taskmaster_dir / 'config.json'

        assert tasks_json.exists(), "tasks.json missing"
        assert config_json.exists(), "config.json missing"

        # Validate JSON structure
        import json
        with open(tasks_json, 'r') as f:
            tasks_data = json.load(f)
            assert 'tasks' in tasks_data, "Invalid tasks.json structure"
            assert 'metadata' in tasks_data, "Missing metadata in tasks.json"

        with open(config_json, 'r') as f:
            config_data = json.load(f)
            assert 'models' in config_data, "Invalid config.json structure"

    def test_readme_content(self):
        """Test README.md has proper content."""
        readme_path = project_root / 'README.md'
        assert readme_path.exists(), "README.md missing"

        with open(readme_path, 'r') as f:
            content = f.read()

            # Check for required sections
            required_sections = [
                '# Cerby Identity Automation Platform',
                '## Overview',
                '## Technology Stack',
                '## Installation',
                '## Running the Application',
                '## Testing',
                '## Project Structure',
            ]

            for section in required_sections:
                assert section in content, f"README missing section: {section}"

    def test_gitignore_content(self):
        """Test .gitignore has proper patterns."""
        gitignore_path = project_root / '.gitignore'
        assert gitignore_path.exists(), ".gitignore missing"

        with open(gitignore_path, 'r') as f:
            content = f.read()

            # Check for required patterns
            required_patterns = [
                '__pycache__',
                '*.pyc',
                'venv/',
                '.env',
                '*.db',
                '.coverage',
                '.pytest_cache/',
                '*.log',
            ]

            for pattern in required_patterns:
                assert pattern in content, f".gitignore missing pattern: {pattern}"

    def test_environment_template(self):
        """Test .env.example has all required variables."""
        env_example = project_root / '.env.example'
        assert env_example.exists(), ".env.example missing"

        with open(env_example, 'r') as f:
            content = f.read()

            # Check for required environment variables
            required_vars = [
                'DEBUG=',
                'ENVIRONMENT=',
                'SECRET_KEY=',
                'LOGFIRE_TOKEN=',
                'DATABASE_URL=',
                'REDIS_URL=',
                'ANTHROPIC_API_KEY=',
                'DARWIN_POPULATION_SIZE=',
                'ENABLE_GENETIC_OPTIMIZATION=',
            ]

            for var in required_vars:
                assert var in content, f".env.example missing variable: {var}"

    def test_project_imports(self):
        """Test that main modules can be imported."""
        try:
            import main
            assert hasattr(main, 'app'), "FastAPI app not found in main.py"
        except ImportError as e:
            pytest.fail(f"Failed to import main module: {e}")

        try:
            from src.core import settings
            assert settings is not None, "Settings not properly initialized"
        except ImportError as e:
            pytest.fail(f"Failed to import settings: {e}")

    def test_requirements_file(self):
        """Test requirements.txt has all necessary packages."""
        req_path = project_root / 'requirements.txt'
        assert req_path.exists(), "requirements.txt missing"

        with open(req_path, 'r') as f:
            content = f.read()

            # Check for core packages
            required_packages = [
                'fastapi',
                'uvicorn',
                'pydantic',
                'logfire',
                'sqlalchemy',
                'pytest',
                'pandas',
                'redis',
                'httpx',
            ]

            for package in required_packages:
                assert package in content, f"requirements.txt missing package: {package}"

    @pytest.mark.parametrize("directory", [
        "src/api/v1/endpoints",
        "src/db/models",
        "src/schemas",
        "src/services",
        "src/ingestion/generators",
    ])
    def test_directory_is_empty_or_has_init(self, directory):
        """Test that directories are either empty or have __init__.py."""
        dir_path = project_root / directory
        if dir_path.exists():
            files = list(dir_path.iterdir())
            # Directory should either be empty or have __init__.py
            if files:
                init_exists = any(f.name == '__init__.py' for f in files)
                assert init_exists, f"{directory} has files but no __init__.py"
