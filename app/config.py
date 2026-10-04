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
    answer_provider: Literal["extractive", "openai"] = "extractive"
    openai_answer_model: str = "gpt-4.1-mini"
    answer_timeout_seconds: float = Field(default=30.0, gt=0, le=120)
    answer_min_relevance: float = Field(default=0.05, ge=-1.0, le=1.0)

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")


settings = Settings()
