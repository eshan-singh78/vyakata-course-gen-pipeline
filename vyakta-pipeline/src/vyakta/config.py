"""Application settings loaded from environment."""

from enum import Enum
from pathlib import Path

from pydantic import Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class LLMProvider(str, Enum):
    ANTHROPIC = "anthropic"
    OPENAI = "openai"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    llm_provider: LLMProvider = Field(default=LLMProvider.ANTHROPIC)
    anthropic_api_key: str | None = Field(default=None)
    openai_api_key: str | None = Field(default=None)

    model_name: str = Field(default="claude-sonnet-4-6")
    max_retries: int = Field(default=3)
    timeout_seconds: int = Field(default=120)
    temperature: float = Field(default=0.2)

    output_dir: Path = Field(default=Path("./output"))

    # v3 production settings
    max_concurrent_llm_calls: int = Field(default=5)
    batch_size: int = Field(default=5)
    enable_cost_tracking: bool = Field(default=True)

    def get_api_key(self) -> str:
        if self.llm_provider == LLMProvider.ANTHROPIC:
            if not self.anthropic_api_key:
                raise ValueError("ANTHROPIC_API_KEY is required when provider is anthropic")
            return self.anthropic_api_key
        if self.llm_provider == LLMProvider.OPENAI:
            if not self.openai_api_key:
                raise ValueError("OPENAI_API_KEY is required when provider is openai")
            return self.openai_api_key
        raise ValueError(f"Unsupported provider: {self.llm_provider}")

    @field_validator("batch_size", "max_concurrent_llm_calls", "timeout_seconds", mode="after")
    @classmethod
    def _validate_positive(cls, v: int) -> int:
        if v <= 0:
            raise ValueError("must be a positive integer")
        return v

    @field_validator("max_retries", mode="after")
    @classmethod
    def _validate_non_negative(cls, v: int) -> int:
        if v < 0:
            raise ValueError("must be a non-negative integer")
        return v

    @field_validator("temperature", mode="after")
    @classmethod
    def _validate_temperature(cls, v: float) -> float:
        if v < 0 or v > 2:
            raise ValueError("must be between 0 and 2")
        return v

    @model_validator(mode="after")
    def _validate_model_provider(self) -> "Settings":
        if self.llm_provider == LLMProvider.OPENAI and self.model_name.startswith("claude-"):
            raise ValueError(f"Model '{self.model_name}' is not valid for OpenAI provider")
        if self.llm_provider == LLMProvider.ANTHROPIC and self.model_name.startswith("gpt-"):
            raise ValueError(f"Model '{self.model_name}' is not valid for Anthropic provider")
        return self


def get_settings() -> Settings:
    return Settings()
