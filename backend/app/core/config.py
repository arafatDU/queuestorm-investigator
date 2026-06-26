"""Application configuration loaded from environment variables.

Centralizes runtime configuration so the rest of the codebase doesn't read
``os.environ`` directly. Values come from ``.env`` (loaded automatically via
``pydantic-settings`` when present) and real environment variables.
"""

from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime settings for the QueueStorm Investigator backend."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ----- Application -----
    app_env: str = Field(default="development", alias="APP_ENV")
    app_host: str = Field(default="0.0.0.0", alias="APP_HOST")
    app_port: int = Field(default=8000, alias="APP_PORT")
    app_log_level: str = Field(default="info", alias="APP_LOG_LEVEL")

    # ----- LLM Provider -----
    gemini_api_key: str = Field(default="", alias="GEMINI_API_KEY")
    gemini_model: str = Field(default="gemma-3-27b-it", alias="GEMINI_MODEL")
    gemini_api_base_url: str = Field(
        default="https://generativelanguage.googleapis.com/v1beta",
        alias="GEMINI_API_BASE_URL",
    )

    # ----- Safety -----
    safety_sanitizer_enabled: bool = Field(
        default=True, alias="SAFETY_SANITIZER_ENABLED"
    )

    # ----- Observability -----
    service_name: str = Field(
        default="queuestorm-investigator-backend", alias="SERVICE_NAME"
    )


@lru_cache
def get_settings() -> Settings:
    """Return a cached :class:`Settings` instance."""

    return Settings()