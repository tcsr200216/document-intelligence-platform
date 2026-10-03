from __future__ import annotations

from typing import Literal

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime configuration loaded from environment variables."""

    database_url: str | None = None
    embedding_provider: Literal["hashing", "openai"] = "hashing"
    embedding_dimensions: int = Field(default=256, ge=1, le=3_072)
    openai_api_key: SecretStr | None = None
    openai_embedding_model: str = "text-embedding-3-small"
    openai_base_url: str = "https://api.openai.com/v1"
    embedding_timeout_seconds: float = Field(default=20.0, gt=0, le=120)

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")


settings = Settings()
