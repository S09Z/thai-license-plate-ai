"""Application configuration loaded from the environment."""

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime settings read from environment variables (prefix ``APP_``).

    Attributes:
        app_name: Human-readable service name.
        env: Deployment environment (``dev``, ``staging`` or ``prod``).
        log_level: Root logging level (e.g. ``INFO``, ``DEBUG``).
        version: Application version, mirrored from ``pyproject.toml``.
    """

    model_config = SettingsConfigDict(
        env_prefix="APP_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    app_name: str = "thai-license-plate-ai"
    env: str = "dev"
    log_level: str = "INFO"
    version: str = "0.1.0"


@lru_cache
def get_settings() -> Settings:
    """Return a cached :class:`Settings` instance.

    Returns:
        The process-wide settings, instantiated once and memoized.
    """
    return Settings()
