"""Media pipeline configuration."""

from enum import Enum
from pathlib import Path

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class TTSProvider(str, Enum):
    OPENAI = "openai"
    KYUTAI = "kyutai"
    ELEVENLABS = "elevenlabs"


class MediaSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # ---- Voice ----
    tts_provider: TTSProvider = Field(default=TTSProvider.OPENAI)
    openai_api_key: str | None = Field(default=None)
    kyutai_api_key: str | None = Field(default=None)
    elevenlabs_api_key: str | None = Field(default=None)

    tts_model: str = Field(default="tts-1")
    tts_voice: str = Field(default="alloy")
    tts_speed: float = Field(default=1.0)
    tts_format: str = Field(default="mp3")
    # Kyutai-specific
    kyutai_base_url: str = Field(default="https://api.kyutai.org/v1")

    # ---- Visuals ----
    slide_width: int = Field(default=1920)
    slide_height: int = Field(default=1080)
    slide_template_dir: Path = Field(
        default=Path(__file__).parent / "visuals" / "templates"
    )
    puppeteer_executable: str | None = Field(default=None)

    # ---- Assembly ----
    ffmpeg_path: str = Field(default="ffmpeg")
    ffprobe_path: str = Field(default="ffprobe")
    output_fps: int = Field(default=30)

    # ---- Pipeline ----
    media_output_dir: Path = Field(default=Path("./media_output"))
    max_concurrent_videos: int = Field(default=3)
    skip_existing: bool = Field(default=True)

    @field_validator("tts_speed", mode="after")
    @classmethod
    def _validate_speed(cls, v: float) -> float:
        if v < 0.25 or v > 4.0:
            raise ValueError("tts_speed must be between 0.25 and 4.0")
        return v

    @field_validator(
        "slide_width", "slide_height", "output_fps", "max_concurrent_videos",
        mode="after",
    )
    @classmethod
    def _validate_positive(cls, v: int) -> int:
        if v <= 0:
            raise ValueError("must be a positive integer")
        return v

    def get_tts_api_key(self) -> str:
        if self.tts_provider == TTSProvider.OPENAI:
            if not self.openai_api_key:
                raise ValueError("OPENAI_API_KEY is required when TTS provider is openai")
            return self.openai_api_key
        if self.tts_provider == TTSProvider.KYUTAI:
            if not self.kyutai_api_key:
                raise ValueError("KYUTAI_API_KEY is required when TTS provider is kyutai")
            return self.kyutai_api_key
        if self.tts_provider == TTSProvider.ELEVENLABS:
            if not self.elevenlabs_api_key:
                raise ValueError("ELEVENLABS_API_KEY is required when TTS provider is elevenlabs")
            return self.elevenlabs_api_key
        raise ValueError(f"Unsupported TTS provider: {self.tts_provider}")
