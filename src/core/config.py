"""
Core configuration module for Cerby Identity Automation Platform.

This module manages all application settings using Pydantic Settings,
providing type-safe configuration with environment variable support.
"""

from typing import List, Optional, Dict, Any
from pydantic_settings import BaseSettings
from pydantic import Field, validator
import os


class Settings(BaseSettings):
    """
    Application settings with environment variable support.

    All settings can be overridden via environment variables.
    """

    # Application settings
    app_name: str = "Cerby Identity Automation Platform"
    app_version: str = "1.0.0"
    debug: bool = Field(default=False, env="DEBUG")
    environment: str = Field(default="development", env="ENVIRONMENT")
    secret_key: str = Field(default="your-secret-key-here-change-in-production", env="SECRET_KEY")

    # API settings
    api_v1_prefix: str = "/api/v1"
    api_docs_url: str = "/api/docs"
    api_redoc_url: str = "/api/redoc"
    api_openapi_url: str = "/api/openapi.json"

    # Server settings
    host: str = Field(default="0.0.0.0", env="HOST")
    port: int = Field(default=8000, env="PORT")
    workers: int = Field(default=1, env="WORKERS")

    # Database settings
    database_url: str = Field(
        default="sqlite:///./cerby.db",
        env="DATABASE_URL"
    )
    database_pool_size: int = Field(default=5, env="DATABASE_POOL_SIZE")
    database_max_overflow: int = Field(default=10, env="DATABASE_MAX_OVERFLOW")
    database_echo: bool = Field(default=False, env="DATABASE_ECHO")

    # Redis settings
    redis_url: str = Field(default="redis://localhost:6379", env="REDIS_URL")
    redis_password: Optional[str] = Field(default=None, env="REDIS_PASSWORD")
    redis_db: int = Field(default=0, env="REDIS_DB")
    redis_decode_responses: bool = True

    # Logfire settings
    logfire_token: str = Field(default="", env="LOGFIRE_TOKEN")
    logfire_project_name: str = Field(
        default="cerby-identity-automation",
        env="LOGFIRE_PROJECT_NAME"
    )
    logfire_service_name: str = Field(default="cerby-api", env="LOGFIRE_SERVICE_NAME")
    logfire_environment: str = Field(default="development", env="LOGFIRE_ENVIRONMENT")

    # CORS settings
    cors_origins: List[str] = Field(
        default=["*"],
        env="CORS_ORIGINS"
    )
    cors_allow_credentials: bool = Field(default=True, env="CORS_ALLOW_CREDENTIALS")
    cors_allow_methods: List[str] = ["*"]
    cors_allow_headers: List[str] = ["*"]

    # Authentication settings
    jwt_algorithm: str = Field(default="HS256", env="JWT_ALGORITHM")
    access_token_expire_minutes: int = Field(default=30, env="ACCESS_TOKEN_EXPIRE_MINUTES")
    refresh_token_expire_days: int = Field(default=7, env="REFRESH_TOKEN_EXPIRE_DAYS")

    # Rate limiting
    rate_limit_enabled: bool = Field(default=True, env="RATE_LIMIT_ENABLED")
    rate_limit_per_minute: int = Field(default=60, env="RATE_LIMIT_PER_MINUTE")
    rate_limit_per_hour: int = Field(default=1000, env="RATE_LIMIT_PER_HOUR")

    # Darwin Genetic Algorithm settings
    darwin_population_size: int = Field(default=100, env="DARWIN_POPULATION_SIZE")
    darwin_generations: int = Field(default=50, env="DARWIN_GENERATIONS")
    darwin_mutation_rate: float = Field(default=0.1, env="DARWIN_MUTATION_RATE")
    darwin_crossover_rate: float = Field(default=0.8, env="DARWIN_CROSSOVER_RATE")
    darwin_elite_size: int = Field(default=10, env="DARWIN_ELITE_SIZE")
    darwin_tournament_size: int = Field(default=3, env="DARWIN_TOURNAMENT_SIZE")
    darwin_max_policy_rules: int = Field(default=50, env="DARWIN_MAX_POLICY_RULES")

    # Identity Provider Simulation settings
    simulate_providers: List[str] = Field(
        default=[
            "okta", "azure_ad", "google_workspace", "slack",
            "github", "jira", "confluence", "salesforce",
            "box", "dropbox"
        ],
        env="SIMULATE_PROVIDERS"
    )
    simulation_interval: int = Field(default=5, env="SIMULATION_INTERVAL")
    simulation_batch_size: int = Field(default=100, env="SIMULATION_BATCH_SIZE")
    simulation_max_events_per_minute: int = Field(default=10000, env="SIMULATION_MAX_EVENTS_PER_MINUTE")

    # Compliance settings
    enable_sox_compliance: bool = Field(default=True, env="ENABLE_SOX_COMPLIANCE")
    enable_gdpr_compliance: bool = Field(default=True, env="ENABLE_GDPR_COMPLIANCE")
    compliance_audit_retention_days: int = Field(default=365, env="COMPLIANCE_AUDIT_RETENTION_DAYS")
    compliance_report_schedule: str = Field(default="0 2 * * *", env="COMPLIANCE_REPORT_SCHEDULE")  # Cron format

    # Panel Dashboard settings
    panel_port: int = Field(default=5006, env="PANEL_PORT")
    panel_websocket_origin: str = Field(default="*", env="PANEL_WEBSOCKET_ORIGIN")
    panel_allow_websocket_origin: List[str] = Field(
        default=["localhost:5006", "127.0.0.1:5006"],
        env="PANEL_ALLOW_WEBSOCKET_ORIGIN"
    )
    panel_num_threads: int = Field(default=4, env="PANEL_NUM_THREADS")
    panel_session_token_expiry: int = Field(default=3600, env="PANEL_SESSION_TOKEN_EXPIRY")

    # Data Lake settings
    data_lake_path: str = Field(default="./data_lake", env="DATA_LAKE_PATH")
    data_lake_partition_by: List[str] = Field(
        default=["year", "month", "day", "provider"],
        env="DATA_LAKE_PARTITION_BY"
    )
    data_lake_file_format: str = Field(default="parquet", env="DATA_LAKE_FILE_FORMAT")
    data_lake_compression: str = Field(default="snappy", env="DATA_LAKE_COMPRESSION")

    # Performance settings
    enable_performance_monitoring: bool = Field(default=True, env="ENABLE_PERFORMANCE_MONITORING")
    slow_query_threshold_ms: int = Field(default=1000, env="SLOW_QUERY_THRESHOLD_MS")
    request_timeout_seconds: int = Field(default=30, env="REQUEST_TIMEOUT_SECONDS")
    max_request_size_mb: int = Field(default=100, env="MAX_REQUEST_SIZE_MB")

    # Feature flags
    enable_genetic_optimization: bool = Field(default=True, env="ENABLE_GENETIC_OPTIMIZATION")
    enable_realtime_processing: bool = Field(default=True, env="ENABLE_REALTIME_PROCESSING")
    enable_adaptive_learning: bool = Field(default=False, env="ENABLE_ADAPTIVE_LEARNING")
    enable_a_b_testing: bool = Field(default=False, env="ENABLE_A_B_TESTING")
    enable_debug_toolbar: bool = Field(default=True, env="ENABLE_DEBUG_TOOLBAR")
    enable_sql_echo: bool = Field(default=False, env="ENABLE_SQL_ECHO")
    enable_request_logging: bool = Field(default=True, env="ENABLE_REQUEST_LOGGING")

    # External API Keys (for future integrations)
    anthropic_api_key: Optional[str] = Field(default=None, env="ANTHROPIC_API_KEY")
    openai_api_key: Optional[str] = Field(default=None, env="OPENAI_API_KEY")
    perplexity_api_key: Optional[str] = Field(default=None, env="PERPLEXITY_API_KEY")

    # Monitoring & Alerts
    alert_email: Optional[str] = Field(default=None, env="ALERT_EMAIL")
    alert_webhook_url: Optional[str] = Field(default=None, env="ALERT_WEBHOOK_URL")
    alert_threshold_error_rate: float = Field(default=0.05, env="ALERT_THRESHOLD_ERROR_RATE")
    alert_threshold_response_time_ms: int = Field(default=5000, env="ALERT_THRESHOLD_RESPONSE_TIME_MS")

    @validator("cors_origins", pre=True)
    def parse_cors_origins(cls, v):
        """Parse CORS origins from comma-separated string."""
        if isinstance(v, str):
            return [origin.strip() for origin in v.split(",")]
        return v

    @validator("simulate_providers", pre=True)
    def parse_simulate_providers(cls, v):
        """Parse provider list from comma-separated string."""
        if isinstance(v, str):
            return [provider.strip() for provider in v.split(",")]
        return v

    @validator("panel_allow_websocket_origin", pre=True)
    def parse_websocket_origins(cls, v):
        """Parse websocket origins from comma-separated string."""
        if isinstance(v, str):
            return [origin.strip() for origin in v.split(",")]
        return v

    @validator("data_lake_partition_by", pre=True)
    def parse_partition_by(cls, v):
        """Parse partition columns from comma-separated string."""
        if isinstance(v, str):
            return [col.strip() for col in v.split(",")]
        return v

    @validator("database_url")
    def validate_database_url(cls, v):
        """Validate and potentially modify database URL."""
        if v.startswith("sqlite"):
            # Ensure SQLite URL has check_same_thread=False for async
            if "?" not in v:
                v += "?check_same_thread=False"
            elif "check_same_thread" not in v:
                v += "&check_same_thread=False"
        return v

    def get_redis_url_with_password(self) -> str:
        """Get Redis URL with password included if set."""
        if self.redis_password:
            # Parse URL and insert password
            parts = self.redis_url.split("://")
            if len(parts) == 2:
                return f"{parts[0]}://:{self.redis_password}@{parts[1]}"
        return self.redis_url

    def get_database_settings(self) -> Dict[str, Any]:
        """Get database configuration for SQLAlchemy."""
        return {
            "url": self.database_url,
            "echo": self.database_echo or self.enable_sql_echo,
            "pool_size": self.database_pool_size,
            "max_overflow": self.database_max_overflow,
            "pool_pre_ping": True,
            "pool_recycle": 3600,  # Recycle connections after 1 hour
        }

    def get_logfire_settings(self) -> Dict[str, Any]:
        """Get Logfire configuration."""
        return {
            "token": self.logfire_token,
            "project_name": self.logfire_project_name,
            "service_name": self.logfire_service_name,
            "environment": self.logfire_environment,
        }

    def is_production(self) -> bool:
        """Check if running in production environment."""
        return self.environment.lower() == "production"

    def is_development(self) -> bool:
        """Check if running in development environment."""
        return self.environment.lower() == "development"

    class Config:
        """Pydantic configuration."""
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False
        # Allow extra fields for forward compatibility
        extra = "allow"


# Create global settings instance
settings = Settings()


# Export commonly used settings
DEBUG = settings.debug
ENVIRONMENT = settings.environment
DATABASE_URL = settings.database_url
REDIS_URL = settings.redis_url
SECRET_KEY = settings.secret_key
