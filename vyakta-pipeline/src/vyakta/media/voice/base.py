"""Abstract TTS client interface."""

from abc import ABC, abstractmethod
from pathlib import Path

from vyakta.media.config import MediaSettings


class TTSError(Exception):
    """Raised when TTS generation fails."""


class TTSClient(ABC):
    """Async text-to-speech client."""

    def __init__(self, settings: MediaSettings):
        self._settings = settings

    @abstractmethod
    async def synthesize(self, text: str, output_path: Path) -> float:
        """Generate audio file from text. Return duration in seconds."""
